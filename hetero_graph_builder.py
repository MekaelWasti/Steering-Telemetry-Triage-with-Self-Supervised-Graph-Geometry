"""PyTorch Geometric heterogeneous graph builder for process telemetry.

This is the PyG refactor of the notebook's NetworkX-only
``GeometricGraphBuilder``.  The important design shift is that shared artifacts
become their own node type instead of creating process-process cliques.

Typical use from the notebook:

    from hetero_graph_builder import HeterogeneousGeometricGraphBuilder

    builder = HeterogeneousGeometricGraphBuilder(
        rare_file_max_degree=10,
        add_same_user_edges=False,
    )
    data = builder.build_graph(process_df, file_links=process_file_df)
    print(data)
    print(builder.report_)

The returned object is a ``torch_geometric.data.HeteroData`` with relation
types such as:

    ("process", "spawns", "process")
    ("process", "touches", "file")
    ("process", "ran_as", "user")
    ("process", "ran_on", "host")

That shape can feed HGT, HeteroConv+GraphSAGE, HeteroConv+GAT, and a later GRU
over time-ordered process embeddings inside each session.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
import torch.nn.functional as F
from torch_geometric.data import HeteroData
from torch_geometric.nn import GATConv, HeteroConv, SAGEConv


DEFAULT_PROCESS_FEATURES = (
    "duration_seconds",
    "cpu_cycle_count",
    "cpu_utilization",
    "commit_charge",
    "commit_peak",
    "read_operation_count",
    "write_operation_count",
    "read_transfer_kilobytes",
    "write_transfer_kilobytes",
    "hard_fault_count",
    "reg_totals",
    "reg_reads",
    "reg_writes",
    "reg_createkeys",
    "reg_deletekeys",
    "reg_deletevalues",
    "Close_Events",
    "Create_Events",
    "Delete_Events",
    "Rename_Events",
    "SetInfo_Events",
    "Read_Bytes",
    "Read_Events",
    "Write_Bytes",
    "Write_Events",
    "file_num_raw_rows",
    "num_uniq_file_hash",
    "conn_id_count",
    "net_total_events",
    "net_total_size",
    "tcp_accept_count",
    "tcp_connect_count",
    "tcp_disconnect_count",
    "tcp_recv_count",
    "tcp_recv_size",
    "tcp_send_count",
    "tcp_send_size",
    "udp_recv_count",
    "udp_recv_size",
    "udp_send_count",
    "udp_send_size",
    "dll_num_uniq_files",
)


@dataclass(frozen=True)
class HeteroGraphColumns:
    """Column names used by ``HeterogeneousGeometricGraphBuilder``."""

    id_col: str = "pid_hash"
    parent_col: str = "parent_pid_hash"
    process_name_col: str = "process_name"
    host_col: str = "hostname"
    file_host_col: str = "Hostname"
    user_col: str = "user_name"
    file_col: str = "filename"
    time_col: str = "process_started"
    label_col: str = "red_team"
    bad_user_col: str = "bad_user"
    remote_ip_col: str = "remote_ip_addr"
    remote_port_col: str = "remote_port"
    protocol_col: str = "protocol"


class UnionFind:
    """Tiny union-find for deriving process sessions from selected relations."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

    def labels(self) -> np.ndarray:
        roots = [self.find(i) for i in range(len(self.parent))]
        root_to_label: dict[int, int] = {}
        labels = np.empty(len(roots), dtype=np.int64)
        for i, root in enumerate(roots):
            labels[i] = root_to_label.setdefault(root, len(root_to_label))
        return labels


class HeterogeneousGeometricGraphBuilder:
    """Build a PyG ``HeteroData`` graph from process telemetry.

    Node types
    ----------
    process
        One node per unique ``pid_hash``.
    file
        Rare touched file/artifact nodes.  These replace the old rare-artifact
        process clique.
    remote
        Optional rare network destination nodes from ``process_net_conn``.
    user, host
        Context nodes.  These are useful for a hetero GNN, but they are not used
        for session connected-components by default because they can become hubs.

    Session extraction
    ------------------
    ``data["process"].session_id`` is computed from strong relations only:
    parent-child, rare-file co-touch, optional rare-network co-destination, and
    optional same-user temporal chains.  Host/user context edges are intentionally
    excluded from sessionization.
    """

    def __init__(
        self,
        columns: HeteroGraphColumns | None = None,
        process_feature_cols: Iterable[str] = DEFAULT_PROCESS_FEATURES,
        rare_file_min_degree: int = 2,
        rare_file_max_degree: int = 10,
        rare_remote_min_degree: int = 2,
        rare_remote_max_degree: int = 20,
        add_host_edges: bool = True,
        add_user_edges: bool = True,
        add_file_edges: bool = True,
        add_network_edges: bool = True,
        add_same_user_edges: bool = False,
        same_user_window: str | pd.Timedelta = "1h",
        normalize_artifacts: bool = True,
        standardize_numeric: bool = True,
        log1p_numeric: bool = True,
    ):
        self.columns = columns or HeteroGraphColumns()
        self.process_feature_cols = tuple(process_feature_cols)
        self.rare_file_min_degree = rare_file_min_degree
        self.rare_file_max_degree = rare_file_max_degree
        self.rare_remote_min_degree = rare_remote_min_degree
        self.rare_remote_max_degree = rare_remote_max_degree
        self.add_host_edges = add_host_edges
        self.add_user_edges = add_user_edges
        self.add_file_edges = add_file_edges
        self.add_network_edges = add_network_edges
        self.add_same_user_edges = add_same_user_edges
        self.same_user_window = pd.Timedelta(same_user_window)
        self.normalize_artifacts = normalize_artifacts
        self.standardize_numeric = standardize_numeric
        self.log1p_numeric = log1p_numeric
        self.reset()

    def reset(self) -> None:
        self.data: HeteroData | None = None
        self.node_maps: dict[str, dict[str, int]] = {}
        self.category_maps: dict[str, dict[str, int]] = {}
        self.feature_columns_: list[str] = []
        self.sessions: list[list[str]] = []
        self.session_process_indices: list[torch.Tensor] = []
        self.report_: dict[str, object] = {}
        self._uf: UnionFind | None = None

    def build_graph(
        self,
        df: pd.DataFrame,
        file_links: pd.DataFrame | None = None,
        net_links: pd.DataFrame | None = None,
    ) -> HeteroData:
        """Return a PyG heterogeneous graph for the given telemetry.

        ``df`` should be process-level rows, such as ``process_uber_summary``.
        ``file_links`` can be ``process_file.parquet`` for richer process-file
        edges; if omitted, the builder falls back to ``df[["pid_hash",
        "filename"]]``.  ``net_links`` can be ``process_net_conn.parquet``.
        """

        self.reset()
        process_df = self._prepare_process_df(df)
        data = HeteroData()
        self._uf = UnionFind(len(process_df))

        self._add_process_nodes(data, process_df)
        self._add_parent_child_edges(data, process_df)

        if self.add_host_edges:
            self._add_categorical_entity_edges(
                data=data,
                df=process_df,
                node_type="host",
                col=self.columns.host_col,
                forward=("process", "ran_on", "host"),
                reverse=("host", "hosts", "process"),
            )

        if self.add_user_edges:
            self._add_categorical_entity_edges(
                data=data,
                df=process_df,
                node_type="user",
                col=self.columns.user_col,
                forward=("process", "ran_as", "user"),
                reverse=("user", "runs", "process"),
            )

        if self.add_file_edges:
            self._add_file_edges(data, process_df, file_links=file_links)

        if self.add_network_edges and net_links is not None:
            self._add_remote_edges(data, net_links=net_links)

        if self.add_same_user_edges:
            self._add_same_user_window_edges(data, process_df)

        self._attach_sessions(data)
        self._attach_report(data)
        self.data = data
        return data

    def get_session_sequences(self) -> list[torch.Tensor]:
        """Return process-index tensors sorted by time within each session.

        This is the bridge to a later GRU: run a GNN to get per-process
        embeddings, then gather those embeddings with these index tensors.
        """

        if self.data is None:
            raise RuntimeError("Call build_graph before get_session_sequences.")
        return self.session_process_indices

    def _prepare_process_df(self, df: pd.DataFrame) -> pd.DataFrame:
        c = self.columns
        if c.id_col not in df.columns:
            raise ValueError(f"Missing required id column: {c.id_col}")

        out = df[df[c.id_col].notna()].copy()
        out[c.id_col] = out[c.id_col].astype(str)
        out = out.drop_duplicates(subset=[c.id_col], keep="first").reset_index(drop=True)
        return out

    def _add_process_nodes(self, data: HeteroData, df: pd.DataFrame) -> None:
        c = self.columns
        process_ids = df[c.id_col].astype(str).tolist()
        self.node_maps["process"] = {pid: i for i, pid in enumerate(process_ids)}

        data["process"].x = self._numeric_matrix(df, self.process_feature_cols)
        data["process"].num_nodes = len(process_ids)
        data["process"].external_id = process_ids

        if c.label_col in df.columns:
            labels = pd.to_numeric(df[c.label_col], errors="coerce").fillna(0)
            data["process"].y = torch.as_tensor(labels.to_numpy(np.int64), dtype=torch.long)

        if c.bad_user_col in df.columns:
            bad_user = df[c.bad_user_col].notna().astype(np.int64)
            data["process"].bad_user = torch.as_tensor(bad_user.to_numpy(), dtype=torch.long)

        data["process"].timestamp = torch.as_tensor(
            self._timestamp_seconds(df), dtype=torch.float32
        )

        for col in (c.process_name_col, c.host_col, c.user_col):
            if col in df.columns:
                encoded, mapping = self._encode_category(df[col])
                safe_col = col.lower().replace(" ", "_")
                data["process"][f"{safe_col}_id"] = encoded
                self.category_maps[f"process.{col}"] = mapping

    def _add_parent_child_edges(self, data: HeteroData, df: pd.DataFrame) -> None:
        c = self.columns
        if c.parent_col not in df.columns:
            self._set_empty_edge(data, ("process", "spawns", "process"))
            self._set_empty_edge(data, ("process", "spawned_by", "process"))
            return

        process_map = self.node_maps["process"]
        links = df[[c.id_col, c.parent_col]].dropna().copy()
        links[c.id_col] = links[c.id_col].astype(str)
        links[c.parent_col] = links[c.parent_col].astype(str)
        links = links[links[c.parent_col].isin(process_map)]
        links = links[links[c.parent_col] != links[c.id_col]]
        links = links.drop_duplicates(subset=[c.parent_col, c.id_col])

        src = links[c.parent_col].map(process_map).to_numpy(np.int64)
        dst = links[c.id_col].map(process_map).to_numpy(np.int64)
        self._set_edge(data, ("process", "spawns", "process"), src, dst)
        self._set_edge(data, ("process", "spawned_by", "process"), dst, src)
        self._union_pairs(src, dst)

    def _add_categorical_entity_edges(
        self,
        data: HeteroData,
        df: pd.DataFrame,
        node_type: str,
        col: str,
        forward: tuple[str, str, str],
        reverse: tuple[str, str, str],
    ) -> None:
        c = self.columns
        if col not in df.columns:
            return

        links = df[[c.id_col, col]].dropna().copy()
        links[c.id_col] = links[c.id_col].astype(str)
        links[col] = links[col].map(self._clean_entity)
        links = links[links[col].notna() & (links[col] != "")]
        links = links.drop_duplicates(subset=[c.id_col, col])
        if links.empty:
            return

        entity_ids = pd.Index(pd.unique(links[col]))
        entity_map = {entity_id: i for i, entity_id in enumerate(entity_ids)}
        self.node_maps[node_type] = entity_map

        degrees = links.groupby(col)[c.id_col].nunique().reindex(entity_ids).fillna(0)
        data[node_type].x = self._degree_features(degrees.to_numpy())
        data[node_type].num_nodes = len(entity_ids)
        data[node_type].external_id = entity_ids.tolist()

        src = links[c.id_col].map(self.node_maps["process"]).to_numpy(np.int64)
        dst = links[col].map(entity_map).to_numpy(np.int64)
        self._set_edge(data, forward, src, dst)
        self._set_edge(data, reverse, dst, src)

    def _add_file_edges(
        self,
        data: HeteroData,
        process_df: pd.DataFrame,
        file_links: pd.DataFrame | None,
    ) -> None:
        c = self.columns
        if file_links is None:
            if c.file_col not in process_df.columns:
                return
            links = process_df[[c.id_col, c.file_col]].copy()
            edge_attr_cols: list[str] = []
        else:
            if c.id_col not in file_links.columns or c.file_col not in file_links.columns:
                return
            edge_attr_cols = [
                col
                for col in ("event_count", "bytes_requested", "num_raw_rows")
                if col in file_links.columns
            ]
            links = file_links[[c.id_col, c.file_col, *edge_attr_cols]].copy()

        links = self._prepare_artifact_links(
            links=links,
            id_col=c.id_col,
            artifact_col=c.file_col,
            attr_cols=edge_attr_cols,
            min_degree=self.rare_file_min_degree,
            max_degree=self.rare_file_max_degree,
        )
        if links.empty:
            return

        file_ids = pd.Index(pd.unique(links[c.file_col]))
        file_map = {file_id: i for i, file_id in enumerate(file_ids)}
        self.node_maps["file"] = file_map

        degrees = links.groupby(c.file_col)[c.id_col].nunique().reindex(file_ids).fillna(0)
        data["file"].x = self._degree_features(degrees.to_numpy())
        data["file"].num_nodes = len(file_ids)
        data["file"].external_id = file_ids.tolist()

        src = links[c.id_col].map(self.node_maps["process"]).to_numpy(np.int64)
        dst = links[c.file_col].map(file_map).to_numpy(np.int64)
        edge_attr = self._edge_attr(links, edge_attr_cols)
        self._set_edge(data, ("process", "touches", "file"), src, dst, edge_attr=edge_attr)
        self._set_edge(data, ("file", "touched_by", "process"), dst, src, edge_attr=edge_attr)
        self._union_by_artifact(links, artifact_col=c.file_col)

    def _add_remote_edges(self, data: HeteroData, net_links: pd.DataFrame) -> None:
        c = self.columns
        required = {c.id_col, c.remote_ip_col}
        if not required.issubset(net_links.columns):
            return

        attr_cols = [
            col for col in ("total_events", "total_size", "num_raw_rows") if col in net_links.columns
        ]
        keep_cols = [
            c.id_col,
            c.remote_ip_col,
            *([c.remote_port_col] if c.remote_port_col in net_links.columns else []),
            *([c.protocol_col] if c.protocol_col in net_links.columns else []),
            *attr_cols,
        ]
        links = net_links[keep_cols].copy()
        links[c.id_col] = links[c.id_col].astype(str)
        links = links[links[c.id_col].isin(self.node_maps["process"])]
        links["_remote"] = self._remote_ids(links)
        links = links[links["_remote"].notna() & (links["_remote"] != "")]

        links = self._prepare_artifact_links(
            links=links[[c.id_col, "_remote", *attr_cols]],
            id_col=c.id_col,
            artifact_col="_remote",
            attr_cols=attr_cols,
            min_degree=self.rare_remote_min_degree,
            max_degree=self.rare_remote_max_degree,
        )
        if links.empty:
            return

        remote_ids = pd.Index(pd.unique(links["_remote"]))
        remote_map = {remote_id: i for i, remote_id in enumerate(remote_ids)}
        self.node_maps["remote"] = remote_map

        degrees = links.groupby("_remote")[c.id_col].nunique().reindex(remote_ids).fillna(0)
        data["remote"].x = self._degree_features(degrees.to_numpy())
        data["remote"].num_nodes = len(remote_ids)
        data["remote"].external_id = remote_ids.tolist()

        src = links[c.id_col].map(self.node_maps["process"]).to_numpy(np.int64)
        dst = links["_remote"].map(remote_map).to_numpy(np.int64)
        edge_attr = self._edge_attr(links, attr_cols)
        self._set_edge(data, ("process", "connected_to", "remote"), src, dst, edge_attr=edge_attr)
        self._set_edge(data, ("remote", "connected_from", "process"), dst, src, edge_attr=edge_attr)
        self._union_by_artifact(links, artifact_col="_remote")

    def _add_same_user_window_edges(self, data: HeteroData, df: pd.DataFrame) -> None:
        c = self.columns
        required = {c.id_col, c.user_col, c.host_col, c.time_col}
        if not required.issubset(df.columns):
            return

        d = df[[c.id_col, c.user_col, c.host_col, c.time_col]].dropna().copy()
        if d.empty:
            return

        d["_time"] = pd.to_datetime(d[c.time_col], errors="coerce", utc=True)
        d = d[d["_time"].notna()].sort_values([c.user_col, c.host_col, "_time"])
        d["_prev_pid"] = d.groupby([c.user_col, c.host_col])[c.id_col].shift(1)
        d["_prev_time"] = d.groupby([c.user_col, c.host_col])["_time"].shift(1)
        gap = d["_time"] - d["_prev_time"]
        d = d[d["_prev_pid"].notna() & (gap <= self.same_user_window)]
        if d.empty:
            return

        process_map = self.node_maps["process"]
        src = d["_prev_pid"].astype(str).map(process_map).to_numpy(np.int64)
        dst = d[c.id_col].astype(str).map(process_map).to_numpy(np.int64)
        self._set_edge(data, ("process", "same_user_window", "process"), src, dst)
        self._set_edge(data, ("process", "same_user_window_rev", "process"), dst, src)
        self._union_pairs(src, dst)

    def _prepare_artifact_links(
        self,
        links: pd.DataFrame,
        id_col: str,
        artifact_col: str,
        attr_cols: list[str],
        min_degree: int,
        max_degree: int,
    ) -> pd.DataFrame:
        links = links.dropna(subset=[id_col, artifact_col]).copy()
        links[id_col] = links[id_col].astype(str)
        links = links[links[id_col].isin(self.node_maps["process"])]
        links[artifact_col] = links[artifact_col].map(self._clean_artifact)
        links = links[links[artifact_col].notna() & (links[artifact_col] != "")]

        group_cols = [id_col, artifact_col]
        if attr_cols:
            for col in attr_cols:
                links[col] = pd.to_numeric(links[col], errors="coerce").fillna(0)
            links = links.groupby(group_cols, as_index=False)[attr_cols].sum()
        else:
            links = links.drop_duplicates(subset=group_cols)

        degrees = links.groupby(artifact_col)[id_col].transform("nunique")
        links = links[(degrees >= min_degree) & (degrees <= max_degree)]
        return links.reset_index(drop=True)

    def _attach_sessions(self, data: HeteroData) -> None:
        if self._uf is None:
            raise RuntimeError("UnionFind was not initialized.")

        labels = self._uf.labels()
        data["process"].session_id = torch.as_tensor(labels, dtype=torch.long)

        process_ids = data["process"].external_id
        timestamps = data["process"].timestamp.numpy()
        label_to_indices: dict[int, list[int]] = {}
        for idx, label in enumerate(labels.tolist()):
            label_to_indices.setdefault(label, []).append(idx)

        self.sessions = []
        self.session_process_indices = []
        for label in sorted(label_to_indices):
            indices = label_to_indices[label]
            indices = sorted(indices, key=lambda i: timestamps[i])
            self.session_process_indices.append(torch.as_tensor(indices, dtype=torch.long))
            self.sessions.append([process_ids[i] for i in indices])

    def _attach_report(self, data: HeteroData) -> None:
        sizes = np.asarray([len(session) for session in self.sessions], dtype=np.int64)
        if sizes.size == 0:
            sizes = np.asarray([0], dtype=np.int64)

        edge_counts = {
            "__".join(edge_type): int(store.edge_index.size(1))
            for edge_type, store in data.edge_items()
        }
        self.report_ = {
            "node_types": {node_type: int(store.num_nodes) for node_type, store in data.node_items()},
            "edge_counts": edge_counts,
            "process_feature_columns": self.feature_columns_,
            "n_sessions": int(len(self.sessions)),
            "median_session_size": float(np.median(sizes)),
            "p95_session_size": float(np.quantile(sizes, 0.95)),
            "max_session_size": int(sizes.max()),
            "singletons": int((sizes == 1).sum()),
        }
        data.graph_report = self.report_

    def _numeric_matrix(self, df: pd.DataFrame, columns: Iterable[str]) -> torch.Tensor:
        present = [col for col in columns if col in df.columns]
        self.feature_columns_ = present
        if not present:
            return torch.ones((len(df), 1), dtype=torch.float32)

        matrix = df[present].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        matrix[~np.isfinite(matrix)] = 0.0

        if self.log1p_numeric:
            matrix = np.sign(matrix) * np.log1p(np.abs(matrix))

        if self.standardize_numeric and len(matrix) > 1:
            mean = matrix.mean(axis=0, keepdims=True)
            std = matrix.std(axis=0, keepdims=True)
            std[std < 1e-6] = 1.0
            matrix = (matrix - mean) / std

        return torch.as_tensor(matrix, dtype=torch.float32)

    def _timestamp_seconds(self, df: pd.DataFrame) -> np.ndarray:
        c = self.columns
        if c.time_col in df.columns:
            ts = pd.to_datetime(df[c.time_col], errors="coerce", utc=True)
            seconds = ts.astype("int64").to_numpy(dtype=np.float64) / 1_000_000_000.0
            seconds[ts.isna().to_numpy()] = 0.0
            return seconds
        seconds_col = f"{c.time_col}_seconds"
        if seconds_col in df.columns:
            return pd.to_numeric(df[seconds_col], errors="coerce").fillna(0).to_numpy(np.float64)
        return np.zeros(len(df), dtype=np.float64)

    def _encode_category(self, series: pd.Series) -> tuple[torch.Tensor, dict[str, int]]:
        values = series.map(self._clean_entity).fillna("__missing__")
        categories = pd.Index(pd.unique(values))
        mapping = {category: i for i, category in enumerate(categories)}
        encoded = values.map(mapping).to_numpy(np.int64)
        return torch.as_tensor(encoded, dtype=torch.long), mapping

    def _clean_entity(self, value: object) -> str | None:
        if pd.isna(value):
            return None
        return str(value).strip()

    def _clean_artifact(self, value: object) -> str | None:
        cleaned = self._clean_entity(value)
        if cleaned is None:
            return None
        return cleaned.lower() if self.normalize_artifacts else cleaned

    def _remote_ids(self, links: pd.DataFrame) -> pd.Series:
        c = self.columns
        remote_ip = links[c.remote_ip_col].map(self._clean_entity)
        valid_remote = remote_ip.notna() & (remote_ip != "")
        protocol = (
            links[c.protocol_col].map(self._clean_entity).fillna("UNK")
            if c.protocol_col in links.columns
            else pd.Series("UNK", index=links.index)
        )
        if c.remote_port_col in links.columns:
            port = links[c.remote_port_col].map(self._port_to_str)
        else:
            port = pd.Series("", index=links.index)
        remote = protocol.astype(str).str.upper() + "://" + remote_ip.astype(str) + ":" + port
        remote.loc[~valid_remote] = None
        return remote

    def _port_to_str(self, value: object) -> str:
        if pd.isna(value):
            return ""
        try:
            return str(int(value))
        except (TypeError, ValueError):
            return str(value).strip()

    def _degree_features(self, degrees: np.ndarray) -> torch.Tensor:
        degrees = degrees.astype(np.float32)
        return torch.as_tensor(np.log1p(degrees).reshape(-1, 1), dtype=torch.float32)

    def _edge_attr(self, links: pd.DataFrame, attr_cols: list[str]) -> torch.Tensor | None:
        if not attr_cols:
            return None
        attrs = links[attr_cols].to_numpy(dtype=np.float32)
        attrs[~np.isfinite(attrs)] = 0.0
        attrs = np.log1p(np.maximum(attrs, 0.0))
        return torch.as_tensor(attrs, dtype=torch.float32)

    def _set_empty_edge(self, data: HeteroData, edge_type: tuple[str, str, str]) -> None:
        data[edge_type].edge_index = torch.empty((2, 0), dtype=torch.long)

    def _set_edge(
        self,
        data: HeteroData,
        edge_type: tuple[str, str, str],
        src: np.ndarray,
        dst: np.ndarray,
        edge_attr: torch.Tensor | None = None,
    ) -> None:
        if len(src) == 0:
            self._set_empty_edge(data, edge_type)
            return
        edge_index = np.vstack([src, dst]).astype(np.int64)
        data[edge_type].edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_attr is not None:
            data[edge_type].edge_attr = edge_attr

    def _union_pairs(self, src: np.ndarray, dst: np.ndarray) -> None:
        if self._uf is None:
            return
        for a, b in zip(src.tolist(), dst.tolist()):
            self._uf.union(int(a), int(b))

    def _union_by_artifact(self, links: pd.DataFrame, artifact_col: str) -> None:
        if self._uf is None:
            return
        process_map = self.node_maps["process"]
        for _, group in links.groupby(artifact_col):
            process_indices = group[self.columns.id_col].map(process_map).dropna().astype(int).tolist()
            if len(process_indices) < 2:
                continue
            first = process_indices[0]
            for other in process_indices[1:]:
                self._uf.union(first, other)


class HeteroGraphEncoder(nn.Module):
    """Small hetero GraphSAGE/GAT encoder for the graph returned above."""

    def __init__(
        self,
        metadata: tuple[list[str], list[tuple[str, str, str]]],
        hidden_channels: int = 64,
        out_channels: int = 64,
        num_layers: int = 2,
        conv: str = "sage",
        heads: int = 2,
    ):
        super().__init__()
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            rel_convs = {}
            for edge_type in metadata[1]:
                if conv == "sage":
                    rel_convs[edge_type] = SAGEConv((-1, -1), hidden_channels)
                elif conv == "gat":
                    rel_convs[edge_type] = GATConv(
                        (-1, -1),
                        hidden_channels,
                        heads=heads,
                        concat=False,
                        add_self_loops=False,
                    )
                else:
                    raise ValueError("conv must be either 'sage' or 'gat'")
            self.convs.append(HeteroConv(rel_convs, aggr="sum"))
        self.output = nn.ModuleDict(
            {node_type: nn.LazyLinear(out_channels) for node_type in metadata[0]}
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        edge_index_dict: dict[tuple[str, str, str], torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        for conv in self.convs:
            out = conv(x_dict, edge_index_dict)
            x_dict = {
                node_type: F.relu(out.get(node_type, x))
                for node_type, x in x_dict.items()
            }
        return {
            node_type: self.output[node_type](x)
            for node_type, x in x_dict.items()
        }


class SessionGRUAggregator(nn.Module):
    """Pool ordered process embeddings into one embedding per session."""

    def __init__(self, input_channels: int, hidden_channels: int = 128, out_channels: int = 128):
        super().__init__()
        self.gru = nn.GRU(input_channels, hidden_channels, batch_first=True)
        self.output = nn.Linear(hidden_channels, out_channels)

    def forward(
        self,
        process_embeddings: torch.Tensor,
        session_sequences: list[torch.Tensor],
    ) -> torch.Tensor:
        if not session_sequences:
            return process_embeddings.new_empty((0, self.output.out_features))

        sequences = [process_embeddings[idx] for idx in session_sequences if len(idx) > 0]
        lengths = torch.as_tensor([len(seq) for seq in sequences], dtype=torch.long, device="cpu")
        padded = pad_sequence(sequences, batch_first=True)
        packed = pack_padded_sequence(
            padded,
            lengths=lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self.gru(packed)
        return self.output(hidden[-1])
