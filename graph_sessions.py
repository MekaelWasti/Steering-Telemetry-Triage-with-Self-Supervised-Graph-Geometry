"""Graph-based sessionization for host telemetry.

A *session* is a connected component in an event graph:
  - nodes  = telemetry events (e.g. one process execution)
  - edges  = evidence that two events belong to the same activity, supplied by
             pluggable EdgeExtractors (temporal proximity, causal spawn links,
             shared rare resources, shared identity, ...)

Swap extractors in/out per dataset; the sessionizer itself is dataset-agnostic.
Every extractor must guard against "god components" (one hub merging an entire
host) — e.g. SharedResourceEdges drops resources touched by too many events.

Example
-------
    sess = GraphSessionizer([
        TemporalChainEdges(host_col="hostname", time_col="t", max_gap_s=120),
        ParentChildEdges(id_col="pid_hash", parent_col="parent_pid_hash"),
        SharedResourceEdges(file_links, resource_cols=["Hostname", "filename"],
                            id_col="pid_hash", max_degree=10),
    ], id_col="pid_hash")
    session_id = sess.fit_transform(df)        # pd.Series aligned to df
    print(sess.report_)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

__all__ = [
    "TemporalChainEdges",
    "ParentChildEdges",
    "SharedResourceEdges",
    "IdentityEdges",
    "GraphSessionizer",
]


class TemporalChainEdges:
    """Link consecutive events in the same scope (e.g. host) when the gap <= max_gap_s.

    Weak, full-coverage backbone: equivalent to idle-gap burst sessionization.
    """

    name = "temporal"

    def __init__(self, host_col: str, time_col: str, max_gap_s: float, weight: float = 0.3):
        self.host_col, self.time_col = host_col, time_col
        self.max_gap_s, self.weight = float(max_gap_s), weight

    def edges(self, df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        d = df[[self.host_col, self.time_col]].copy()
        d["_id"] = ids.values
        d = d.sort_values([self.host_col, self.time_col])
        prev_id = d.groupby(self.host_col)["_id"].shift(1)
        gap = d[self.time_col] - d.groupby(self.host_col)[self.time_col].shift(1)
        m = prev_id.notna() & (gap <= self.max_gap_s)
        return pd.DataFrame({"src": prev_id[m], "dst": d.loc[m, "_id"],
                             "weight": self.weight, "etype": self.name})


class ParentChildEdges:
    """Link an event to its causal parent when the parent is present in the data.

    Strong evidence; survives arbitrary time gaps (stitches paused activity).

    God-component guard: parents with more than ``max_children`` resolved
    children are *hubs* (service managers, telemetry agents — in ACME4 test,
    ``wintap.exe`` parents ~10K collector children) and their edges are dropped;
    spawning thousands of children is host infrastructure, not one activity.
    """

    name = "parent"

    def __init__(self, id_col: str, parent_col: str, max_children: int = 25,
                 weight: float = 1.0):
        self.id_col, self.parent_col = id_col, parent_col
        self.max_children, self.weight = max_children, weight

    def edges(self, df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        present = set(ids)
        m = df[self.parent_col].notna() & df[self.parent_col].isin(present) \
            & (df[self.parent_col] != ids.values)
        e = pd.DataFrame({"src": df.loc[m, self.parent_col], "dst": ids[m].values})
        fanout = e.groupby("src")["dst"].transform("size")
        e = e[fanout <= self.max_children]
        return e.assign(weight=self.weight, etype=self.name)


class SharedResourceEdges:
    """Link events that touched the same *rare* resource (file, registry key,
    socket, ...), supplied as a long-format link table: one row = (event id, resource).

    God-component guard: resources touched by more than ``max_degree`` distinct
    events are dropped (agent logs, OS hives, DNS servers...). Each kept resource
    contributes a star (to its first toucher) rather than a full clique — same
    connected components, O(degree) instead of O(degree^2) edges.
    """

    name = "resource"

    def __init__(self, links: pd.DataFrame, resource_cols: list[str] | str,
                 id_col: str, min_degree: int = 2, max_degree: int = 10,
                 weight: float = 0.6):
        self.links = links
        self.resource_cols = [resource_cols] if isinstance(resource_cols, str) else list(resource_cols)
        self.id_col = id_col
        self.min_degree, self.max_degree, self.weight = min_degree, max_degree, weight

    def edges(self, df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        present = set(ids)
        lk = self.links[self.links[self.id_col].isin(present)]
        lk = lk[[self.id_col, *self.resource_cols]].dropna().drop_duplicates()
        # degree computed *within this split* to keep train/test hygiene
        deg = lk.groupby(self.resource_cols)[self.id_col].transform("nunique")
        lk = lk[(deg >= self.min_degree) & (deg <= self.max_degree)]
        # star per resource: hub = first event id seen for that resource
        hub = lk.groupby(self.resource_cols)[self.id_col].transform("first")
        m = hub != lk[self.id_col]
        return pd.DataFrame({"src": hub[m], "dst": lk.loc[m, self.id_col],
                             "weight": self.weight, "etype": self.name})


class IdentityEdges(SharedResourceEdges):
    """Shared-identity linkage (same user/account/token...) — a SharedResourceEdges
    where the 'resource' is the identity column on the event table itself.
    Scope with a host/time column in ``resource_cols`` to avoid god components.
    """

    name = "identity"

    def __init__(self, resource_cols, id_col: str, max_degree: int = 50, weight: float = 0.5):
        # links table is the event table itself; bound at fit time
        super().__init__(links=None, resource_cols=resource_cols, id_col=id_col,
                         min_degree=2, max_degree=max_degree, weight=weight)

    def edges(self, df: pd.DataFrame, ids: pd.Series) -> pd.DataFrame:
        self.links = df.assign(**{self.id_col: ids.values})
        return super().edges(df, ids)


class GraphSessionizer:
    """Union of all extractor edges -> connected components -> session ids.

    Parameters
    ----------
    extractors : list of edge extractors (see above)
    id_col     : column holding a unique event id (rows with duplicate ids are
                 collapsed onto the same node)

    After ``fit_transform``: ``self.report_`` has per-extractor edge counts and
    component-size stats — check ``max_component`` for god components.
    """

    def __init__(self, extractors: list, id_col: str):
        self.extractors, self.id_col = extractors, id_col

    def fit_transform(self, df: pd.DataFrame) -> pd.Series:
        ids = df[self.id_col].astype(str)
        uniq = pd.Index(ids.unique())
        idx = pd.Series(np.arange(len(uniq)), index=uniq)

        frames, counts = [], {}
        for ex in self.extractors:
            e = ex.edges(df, ids)
            counts[ex.name] = len(e)
            frames.append(e)
        E = pd.concat(frames, ignore_index=True)

        src = idx[E["src"].astype(str)].to_numpy()
        dst = idx[E["dst"].astype(str)].to_numpy()
        n = len(uniq)
        adj = coo_matrix((np.ones(len(E)), (src, dst)), shape=(n, n))
        _, labels = connected_components(adj, directed=False)

        comp = pd.Series(labels, index=uniq)
        sizes = ids.map(comp).value_counts()
        self.report_ = {
            "edges_per_type": counts,
            "n_nodes": n,
            "n_sessions": int(comp.nunique()),
            "median_component": float(sizes.median()),
            "p95_component": float(sizes.quantile(0.95)),
            "max_component": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
        }
        return ids.map(comp).rename("session_id").set_axis(df.index)
