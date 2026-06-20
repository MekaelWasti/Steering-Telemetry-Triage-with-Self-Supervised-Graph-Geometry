"""Simple clustering and cluster-level review metrics for session embeddings."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import MiniBatchKMeans


GENERIC_TERM_VALUES = {
    "0",
    "1",
    "2",
    "c",
    "d",
    "x64",
    "x86",
    "program",
    "programs",
    "files",
    "file",
    "microsoft",
    "windows",
    "system32",
    "syswow64",
    "appdata",
    "local",
    "roaming",
    "user",
    "users",
    "none",
    "true",
    "false",
    "-c",
    "-s",
    "--",
}


@dataclass
class SessionClusterConfig:
    """Configuration for the first prototype clusterer."""

    n_clusters: int | None = None
    min_clusters: int = 8
    max_clusters: int = 64
    random_state: int = 42
    batch_size: int = 2048
    n_init: int = 10
    score_col: str = "rank_ensemble_score"
    label_cols: tuple[str, ...] = ("red_team", "bad_user")
    context_cols: tuple[str, ...] = ("primary_host", "top_processes")


CLUSTER_RANKING_STRATEGIES: dict[str, tuple[str, ...]] = {
    "max_then_mean": ("score_max", "score_mean"),
    "p95_then_mean": ("score_p95", "score_mean"),
    "mean_then_p95": ("score_mean", "score_p95"),
    "top_decile_count": ("score_top_decile_sessions", "score_p95"),
    "top_decile_rate": ("score_top_decile_rate", "score_p95"),
    "size_adjusted_p95": ("score_p95_size_adjusted", "score_p95"),
}


def choose_n_clusters(
    n_rows: int,
    min_clusters: int = 8,
    max_clusters: int = 64,
) -> int:
    """Choose a conservative cluster count for analyst review."""

    if n_rows <= 1:
        return 1
    suggested = int(round(np.sqrt(n_rows)))
    return max(1, min(max_clusters, max(min_clusters, suggested), n_rows))


def cluster_session_embeddings(
    embeddings: np.ndarray,
    config: SessionClusterConfig | None = None,
) -> tuple[np.ndarray, MiniBatchKMeans]:
    """Cluster session embeddings with MiniBatchKMeans."""

    config = config or SessionClusterConfig()
    X = np.asarray(embeddings, dtype=np.float32)
    if X.ndim != 2 or len(X) == 0:
        raise ValueError("embeddings must be a non-empty 2D array")

    n_clusters = config.n_clusters or choose_n_clusters(
        len(X),
        min_clusters=config.min_clusters,
        max_clusters=config.max_clusters,
    )
    n_clusters = min(max(1, n_clusters), len(X))
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        random_state=config.random_state,
        batch_size=min(config.batch_size, max(len(X), 1)),
        n_init=config.n_init,
    )
    labels = model.fit_predict(X)
    return labels.astype(np.int64), model


def attach_cluster_labels(
    session_scores: pd.DataFrame,
    cluster_labels: Sequence[object],
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Return session scores with a cluster id column."""

    if len(session_scores) != len(cluster_labels):
        raise ValueError("session_scores and cluster_labels must have the same length")
    out = session_scores.copy()
    out[cluster_col] = list(cluster_labels)
    return out


def summarize_clusters(
    clustered_sessions: pd.DataFrame,
    config: SessionClusterConfig | None = None,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Summarize cluster size, anomaly score, labels, and context concentration."""

    config = config or SessionClusterConfig()
    if cluster_col not in clustered_sessions.columns:
        raise ValueError(f"clustered_sessions must contain {cluster_col!r}")
    if config.score_col not in clustered_sessions.columns:
        raise ValueError(f"clustered_sessions must contain score column {config.score_col!r}")

    score_threshold = float(clustered_sessions[config.score_col].quantile(0.90))
    total_top_decile = int((clustered_sessions[config.score_col] >= score_threshold).sum())
    rows = []
    for cluster_id, group in clustered_sessions.groupby(cluster_col, sort=True):
        high_score_count = int((group[config.score_col] >= score_threshold).sum())
        high_score_rate = high_score_count / len(group) if len(group) else np.nan
        score_p95 = float(group[config.score_col].quantile(0.95))
        row: dict[str, object] = {
            cluster_col: cluster_id,
            "n_sessions": int(len(group)),
            "score_max": float(group[config.score_col].max()),
            "score_mean": float(group[config.score_col].mean()),
            "score_p95": score_p95,
            "score_top_decile_sessions": high_score_count,
            "score_top_decile_rate": float(high_score_rate),
            "score_top_decile_share": (
                high_score_count / total_top_decile if total_top_decile else np.nan
            ),
            "score_p95_size_adjusted": float(score_p95 * np.log1p(len(group))),
        }

        for label_col in config.label_cols:
            if label_col not in group.columns:
                continue
            labels = _binary_labels(group[label_col])
            row[f"{label_col}_sessions"] = int(labels.sum())
            row[f"{label_col}_rate"] = float(labels.mean()) if len(labels) else np.nan

        for context_col in config.context_cols:
            if context_col not in group.columns:
                continue
            values = group[context_col].dropna().astype(str)
            values = values[values.str.len() > 0]
            prefix = context_col[4:] if context_col.startswith("top_") else context_col
            row[f"distinct_{prefix}"] = int(values.nunique())
            if values.empty:
                row[f"top_{prefix}"] = ""
                row[f"top_{prefix}_share"] = np.nan
            else:
                counts = values.value_counts()
                row[f"top_{prefix}"] = counts.index[0]
                row[f"top_{prefix}_share"] = float(counts.iloc[0] / len(group))

        rows.append(row)

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    return summary.sort_values(["score_max", "score_mean"], ascending=False).reset_index(drop=True)


def cluster_keyphrase_report(
    session_text: sparse.spmatrix,
    cluster_labels: Sequence[object],
    feature_names: Sequence[str],
    top_n_terms: int = 10,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Extract simple TF-IDF keyphrases for each cluster."""

    matrix = session_text.tocsr()
    if matrix.shape[0] != len(cluster_labels):
        raise ValueError("session_text rows must align to cluster_labels")
    if matrix.shape[1] != len(feature_names):
        raise ValueError("feature_names length must match session_text columns")

    labels = pd.Index(cluster_labels)
    rows = []
    for cluster_id in sorted(pd.unique(labels)):
        row_mask = labels == cluster_id
        cluster_matrix = matrix[np.asarray(row_mask)]
        if cluster_matrix.shape[0] == 0:
            continue
        mean_scores = np.asarray(cluster_matrix.mean(axis=0)).ravel()
        top_terms: list[str] = []
        grouped_terms: dict[str, list[str]] = {
            "process": [],
            "args": [],
            "file_path": [],
            "ancestor": [],
            "other": [],
        }
        if mean_scores.size and np.nanmax(mean_scores) > 0:
            top_indices = np.argsort(mean_scores)[::-1]
            for idx in top_indices:
                if mean_scores[idx] <= 0:
                    break
                term = _clean_feature_name(feature_names[idx])
                if _is_useful_keyphrase(term):
                    top_terms.append(term)
                    group = _keyphrase_group(term)
                    grouped_terms[group].append(_compact_keyphrase(term))
                if len(top_terms) >= top_n_terms:
                    break
        grouped_terms = {
            group: _dedupe_preserve_order(values)
            for group, values in grouped_terms.items()
        }
        rows.append(
            {
                cluster_col: cluster_id,
                "cluster_label": _make_cluster_label(grouped_terms, top_terms),
                "process_keyphrases": ", ".join(grouped_terms["process"]),
                "arg_keyphrases": ", ".join(grouped_terms["args"]),
                "file_path_keyphrases": ", ".join(grouped_terms["file_path"]),
                "ancestor_keyphrases": ", ".join(grouped_terms["ancestor"]),
                "other_keyphrases": ", ".join(grouped_terms["other"]),
                "keyphrases": ", ".join(top_terms),
            }
        )
    return pd.DataFrame(rows)


def add_cluster_keyphrases(
    cluster_summary: pd.DataFrame,
    keyphrase_report: pd.DataFrame,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Attach keyphrase strings to a cluster summary table."""

    if cluster_summary.empty or keyphrase_report.empty:
        out = cluster_summary.copy()
        if "keyphrases" not in out.columns:
            out["keyphrases"] = ""
        if "cluster_label" not in out.columns:
            out["cluster_label"] = ""
        return out
    return cluster_summary.merge(keyphrase_report, on=cluster_col, how="left")


def cluster_review_report(
    cluster_summary: pd.DataFrame,
    label_cols: Sequence[str] = ("red_team", "bad_user"),
    top_cluster_fractions: Sequence[float] = (0.05, 0.10, 0.20),
    sort_cols: Sequence[str] = ("score_max", "score_mean"),
    ranking_strategy: str = "max_then_mean",
) -> pd.DataFrame:
    """Simulate analyst review of top-ranked clusters."""

    if cluster_summary.empty:
        return pd.DataFrame()
    if "n_sessions" not in cluster_summary.columns:
        raise ValueError("cluster_summary must contain n_sessions")

    available_sort_cols = [col for col in sort_cols if col in cluster_summary.columns]
    if not available_sort_cols:
        raise ValueError(f"cluster_summary has none of the requested sort columns: {list(sort_cols)}")

    ordered = cluster_summary.sort_values(available_sort_cols, ascending=False)
    total_sessions = int(ordered["n_sessions"].sum())
    rows = []
    for label_col in label_cols:
        label_count_col = f"{label_col}_sessions"
        if label_count_col not in ordered.columns:
            continue
        total_positive = int(ordered[label_count_col].sum())
        baseline = total_positive / total_sessions if total_sessions else np.nan
        for fraction in top_cluster_fractions:
            top_n_clusters = max(1, int(np.ceil(len(ordered) * fraction)))
            top = ordered.head(top_n_clusters)
            reviewed_sessions = int(top["n_sessions"].sum())
            hits = int(top[label_count_col].sum())
            positive_rate = hits / reviewed_sessions if reviewed_sessions else np.nan
            rows.append(
                {
                    "label": label_col,
                    "ranking_strategy": ranking_strategy,
                    "top_cluster_fraction": float(fraction),
                    "top_n_clusters": int(top_n_clusters),
                    "reviewed_sessions": reviewed_sessions,
                    "hits": hits,
                    "total_positive": total_positive,
                    "session_review_fraction": reviewed_sessions / total_sessions
                    if total_sessions
                    else np.nan,
                    "cluster_positive_rate": positive_rate,
                    "recall_at_cluster_review": hits / total_positive if total_positive else np.nan,
                    "lift_vs_baseline": positive_rate / baseline if baseline else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compare_cluster_review_strategies(
    cluster_summary: pd.DataFrame,
    strategies: dict[str, Sequence[str]] | None = None,
    label_cols: Sequence[str] = ("red_team", "bad_user"),
    top_cluster_fractions: Sequence[float] = (0.05, 0.10, 0.20),
) -> pd.DataFrame:
    """Compare several label-free ways to rank clusters for analyst review.

    The strategies use only score/size summary columns.  Labels are consumed
    after ranking to evaluate which strategy would have recovered positives.
    """

    strategies = strategies or CLUSTER_RANKING_STRATEGIES
    frames = []
    for name, sort_cols in strategies.items():
        if not any(col in cluster_summary.columns for col in sort_cols):
            continue
        frames.append(
            cluster_review_report(
                cluster_summary,
                label_cols=label_cols,
                top_cluster_fractions=top_cluster_fractions,
                sort_cols=sort_cols,
                ranking_strategy=name,
            )
        )
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _clean_feature_name(name: object) -> str:
    text = str(name)
    if text.startswith("text:"):
        text = text[5:]
    return text


def _is_useful_keyphrase(term: str) -> bool:
    parts = [part.strip() for part in term.split() if part.strip()]
    if not parts:
        return False

    values = []
    for part in parts:
        value = part.split("=", 1)[1] if "=" in part else part
        value = value.strip().lower().strip("'\"")
        shortened = _shorten_value(value).lower().strip("'\"")
        if shortened and shortened not in GENERIC_TERM_VALUES and len(shortened) > 1:
            values.append(shortened)
        elif value and value not in GENERIC_TERM_VALUES and len(value) > 1:
            values.append(value)

    if not values:
        return False
    return True


def _keyphrase_group(term: str) -> str:
    prefixes = [_term_prefix(part) for part in term.split()]
    if any(prefix == "proc" for prefix in prefixes):
        return "process"
    if any(prefix == "arg" for prefix in prefixes):
        return "args"
    if any(prefix in {"file", "path", "rare_file"} for prefix in prefixes):
        return "file_path"
    if any(prefix in {"ancestor", "ancestor_path"} for prefix in prefixes):
        return "ancestor"
    return "other"


def _term_prefix(part: str) -> str:
    return part.split("=", 1)[0].strip().lower() if "=" in part else ""


def _compact_keyphrase(term: str) -> str:
    values = []
    for part in term.split():
        value = part.split("=", 1)[1] if "=" in part else part
        value = _shorten_value(value.strip())
        if value and value.lower() not in GENERIC_TERM_VALUES:
            values.append(value)
    return " ".join(_dedupe_preserve_order(values))


def _make_cluster_label(grouped_terms: dict[str, list[str]], top_terms: list[str]) -> str:
    process = _first(grouped_terms["process"]) or _first(grouped_terms["ancestor"])
    arg = _first(grouped_terms["args"])
    file_path = _first(grouped_terms["file_path"])

    if process and arg:
        return f"{process} + {arg}"
    if process and file_path:
        if process.lower() in file_path.lower() or file_path.lower() in process.lower():
            return process
        return f"{process} + {file_path}"
    if process:
        return process
    if arg:
        return f"args: {arg}"
    if file_path:
        return f"file/path: {file_path}"
    if top_terms:
        return _compact_keyphrase(top_terms[0]) or "unlabeled cluster"
    return "unlabeled cluster"


def _shorten_value(value: str) -> str:
    cleaned = value.replace("\\", "/")
    if "/" in cleaned:
        tail = cleaned.rstrip("/").split("/")[-1]
        return tail or value
    return value


def _first(values: Sequence[str]) -> str:
    return values[0] if values else ""


def _dedupe_preserve_order(values: Sequence[str]) -> list[str]:
    seen = set()
    out = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            out.append(value)
    return out


def _binary_labels(labels: Sequence[object]) -> np.ndarray:
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)

    clean = series.astype("string")
    present = clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")
    return present.fillna(False).to_numpy(dtype=np.int64)
