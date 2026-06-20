"""Reusable unsupervised detector for process telemetry.

The detector wraps the current strongest notebook path:

    semantic process text -> TF-IDF -> TruncatedSVD -> dense embeddings
    -> kNN distance / IsolationForest anomaly scores

Labels are intentionally not used by the detector.  The evaluation helpers at
the bottom can consume labels after scoring to report neighborhood purity and
top-k enrichment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from semantic_feature_baseline import SemanticFeatureConfig, SemanticProcessFeatureBuilder


@dataclass
class UnsupervisedTelemetryConfig:
    """Configuration for label-free process embedding and scoring."""

    semantic: SemanticFeatureConfig = field(
        default_factory=lambda: SemanticFeatureConfig(
            include_numeric=False,
            include_process_name=True,
            include_args=True,
            include_process_path=True,
            include_file=True,
            include_ancestor_path=True,
            max_text_features=20_000,
            min_df=2,
            ngram_range=(1, 2),
            rare_file_max_degree=20,
        )
    )
    id_col: str = "pid_hash"
    svd_components: int = 64
    random_state: int = 42
    knn_k: int = 15
    isolation_estimators: int = 200
    isolation_contamination: str | float = "auto"
    isolation_n_jobs: int = 1
    fit_isolation_forest: bool = True


class UnsupervisedTelemetryDetector:
    """Fit and score process embeddings without using labels."""

    def __init__(self, config: UnsupervisedTelemetryConfig | None = None):
        self.config = config or UnsupervisedTelemetryConfig()
        self.semantic_builder_: SemanticProcessFeatureBuilder | None = None
        self.svd_: TruncatedSVD | None = None
        self.scaler_: StandardScaler | None = None
        self.neighbor_model_: NearestNeighbors | None = None
        self.isolation_model_: IsolationForest | None = None
        self.train_embeddings_: np.ndarray | None = None
        self.train_process_ids_: list[str] = []
        self.report_: dict[str, object] = {}

    def fit(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> "UnsupervisedTelemetryDetector":
        """Fit semantic features, SVD, scaler, and unsupervised score models."""

        ordered_ids = self._ordered_process_ids(df, process_ids)
        if not ordered_ids:
            raise ValueError("No process ids available for fitting")

        self.semantic_builder_ = SemanticProcessFeatureBuilder(self.config.semantic)
        text_features = self.semantic_builder_.fit_transform(df, ordered_ids)

        n_components = self._effective_svd_components(text_features)
        self.svd_ = TruncatedSVD(
            n_components=n_components,
            random_state=self.config.random_state,
        )
        dense = self.svd_.fit_transform(text_features).astype(np.float32, copy=False)

        self.scaler_ = StandardScaler()
        embeddings = self.scaler_.fit_transform(dense).astype(np.float32, copy=False)

        self.train_embeddings_ = embeddings
        self.train_process_ids_ = ordered_ids

        n_neighbors = min(self.config.knn_k, len(embeddings))
        self.neighbor_model_ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.neighbor_model_.fit(embeddings)

        self.isolation_model_ = None
        if self.config.fit_isolation_forest and len(embeddings) >= 2:
            self.isolation_model_ = IsolationForest(
                n_estimators=self.config.isolation_estimators,
                contamination=self.config.isolation_contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.isolation_n_jobs,
            )
            self.isolation_model_.fit(embeddings)

        explained = getattr(self.svd_, "explained_variance_ratio_", np.array([], dtype=float))
        self.report_ = {
            "n_train_processes": len(ordered_ids),
            "semantic_feature_shape": tuple(text_features.shape),
            "embedding_shape": tuple(embeddings.shape),
            "svd_components": n_components,
            "svd_explained_variance": float(np.sum(explained)),
            "knn_k": n_neighbors,
            "isolation_forest_fit": self.isolation_model_ is not None,
        }
        return self

    def fit_transform(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Fit the detector and return training embeddings."""

        self.fit(df, process_ids)
        if self.train_embeddings_ is None:
            raise RuntimeError("Detector fit did not produce embeddings")
        return self.train_embeddings_

    def transform(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> np.ndarray:
        """Transform new process rows into the fitted embedding space."""

        self._check_fit()
        ordered_ids = self._ordered_process_ids(df, process_ids)
        text_features = self.semantic_builder_.transform(df, ordered_ids)
        dense = self.svd_.transform(text_features).astype(np.float32, copy=False)
        return self.scaler_.transform(dense).astype(np.float32, copy=False)

    def score_processes(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Return process ids, unsupervised anomaly scores, and rank ensemble."""

        ordered_ids = self._ordered_process_ids(df, process_ids)
        embeddings = self.transform(df, ordered_ids)
        scores = self.score_embeddings(embeddings)
        scores.insert(0, self.config.id_col, ordered_ids)
        return scores

    def score_embeddings(self, embeddings: np.ndarray) -> pd.DataFrame:
        """Score precomputed embeddings against the fitted training distribution."""

        self._check_fit()
        embeddings = np.asarray(embeddings, dtype=np.float32)
        distances, _ = self.neighbor_model_.kneighbors(embeddings)

        score_table = pd.DataFrame(
            {
                "knn_train_distance_score": distances.mean(axis=1),
            }
        )
        rank_cols = ["knn_train_distance_rank"]
        score_table["knn_train_distance_rank"] = _percent_rank(
            score_table["knn_train_distance_score"]
        )

        if self.isolation_model_ is not None:
            score_table["isolation_forest_score"] = -self.isolation_model_.decision_function(embeddings)
            score_table["isolation_forest_rank"] = _percent_rank(
                score_table["isolation_forest_score"]
            )
            rank_cols.append("isolation_forest_rank")

        score_table["rank_ensemble_score"] = score_table[rank_cols].mean(axis=1)
        return score_table

    def _check_fit(self) -> None:
        if (
            self.semantic_builder_ is None
            or self.svd_ is None
            or self.scaler_ is None
            or self.neighbor_model_ is None
            or self.train_embeddings_ is None
        ):
            raise RuntimeError("Call fit before transform or score")

    def _ordered_process_ids(
        self,
        df: pd.DataFrame,
        process_ids: Iterable[str] | None,
    ) -> list[str]:
        if self.config.id_col not in df.columns:
            raise ValueError(f"df must contain {self.config.id_col!r}")

        if process_ids is not None:
            return [str(pid) for pid in process_ids]

        work = df[df[self.config.id_col].notna()].drop_duplicates(self.config.id_col, keep="first")
        return work[self.config.id_col].astype(str).tolist()

    def _effective_svd_components(self, matrix: sparse.spmatrix) -> int:
        if matrix.shape[0] < 2 or matrix.shape[1] < 2:
            return 1
        return max(1, min(self.config.svd_components, matrix.shape[0] - 1, matrix.shape[1] - 1))


def knn_label_report(
    embeddings: np.ndarray,
    labels: Sequence[object],
    embedding_name: str,
    label_name: str,
    k: int = 15,
) -> dict[str, float | int | str]:
    """Evaluate neighborhood purity without using labels for fitting."""

    X = np.asarray(embeddings, dtype=np.float32)
    y = _binary_labels(labels)
    if len(X) != len(y):
        raise ValueError("embeddings and labels must have the same length")
    if len(X) < 2:
        raise ValueError("At least two rows are required for a kNN label report")

    n_neighbors = min(k + 1, len(X))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
    nn.fit(X)
    _, neighbors = nn.kneighbors(X)

    neighbor_rates = np.empty(len(X), dtype=np.float32)
    for row_idx, row_neighbors in enumerate(neighbors):
        kept = [idx for idx in row_neighbors if idx != row_idx][:k]
        if kept:
            neighbor_rates[row_idx] = float(y[kept].mean())
        else:
            neighbor_rates[row_idx] = np.nan

    pos = y == 1
    neg = ~pos
    baseline = float(y.mean())
    pos_rate = _safe_nanmean(neighbor_rates[pos])
    neg_rate = _safe_nanmean(neighbor_rates[neg])

    return {
        "embedding": embedding_name,
        "label": label_name,
        "k": int(k),
        "rows": int(len(y)),
        "positive_rows": int(pos.sum()),
        "negative_rows": int(neg.sum()),
        "baseline_positive_rate": baseline,
        "all_point_neighbor_positive_rate": _safe_nanmean(neighbor_rates),
        "positive_point_neighbor_positive_rate": pos_rate,
        "negative_point_neighbor_positive_rate": neg_rate,
        "separation_gap_pos_minus_neg": pos_rate - neg_rate,
        "positive_neighbor_lift_vs_baseline": pos_rate / baseline if baseline > 0 else np.nan,
        "negative_contamination_lift_vs_baseline": neg_rate / baseline if baseline > 0 else np.nan,
    }


def top_fraction_enrichment(
    score_table: pd.DataFrame,
    score_col: str,
    labels: Sequence[object],
    label_name: str,
    top_fractions: Sequence[float] = (0.005, 0.01, 0.02, 0.05, 0.10),
) -> pd.DataFrame:
    """Report label enrichment in the highest-scoring review budgets."""

    y = _binary_labels(labels)
    if len(score_table) != len(y):
        raise ValueError("score_table and labels must have the same length")
    if score_col not in score_table.columns:
        raise ValueError(f"score_table does not contain score column {score_col!r}")

    work = score_table.copy()
    work["_label"] = y
    work = work.sort_values(score_col, ascending=False)

    rows = []
    total_positive = int(y.sum())
    baseline = float(y.mean())
    for fraction in top_fractions:
        top_n = max(1, int(np.ceil(len(work) * fraction)))
        top = work.head(top_n)
        hits = int(top["_label"].sum())
        top_rate = hits / top_n
        rows.append(
            {
                "score": score_col,
                "label": label_name,
                "top_fraction": float(fraction),
                "top_n": int(top_n),
                "hits": hits,
                "total_positive": total_positive,
                "baseline_positive_rate": baseline,
                "top_positive_rate": float(top_rate),
                "recall_at_top": hits / total_positive if total_positive else np.nan,
                "lift_vs_baseline": top_rate / baseline if baseline > 0 else np.nan,
            }
        )
    return pd.DataFrame(rows)


def collect_top_fraction_enrichment(
    score_table: pd.DataFrame,
    embedding_name: str,
    label_cols: Sequence[str] = ("red_team", "bad_user"),
    score_cols: Sequence[str] = (
        "knn_train_distance_score",
        "isolation_forest_score",
        "rank_ensemble_score",
    ),
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Collect top-k enrichment for multiple labels and score columns."""

    frames = []
    for label_col in label_cols:
        if label_col not in score_table.columns:
            continue
        for score_col in score_cols:
            if score_col not in score_table.columns:
                continue
            frame = top_fraction_enrichment(
                score_table,
                score_col,
                score_table[label_col],
                label_col,
                top_fractions=top_fractions,
            )
            frame.insert(0, "embedding", embedding_name)
            frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def best_enrichment_by_budget(
    enrichment: pd.DataFrame,
    metric: str = "recall_at_top",
) -> pd.DataFrame:
    """Return the best embedding/score row for each label and top fraction."""

    if enrichment.empty:
        return enrichment.copy()
    required = {"label", "top_fraction", metric}
    missing = required - set(enrichment.columns)
    if missing:
        raise ValueError(f"enrichment is missing required columns: {sorted(missing)}")

    sorted_rows = enrichment.sort_values(
        ["label", "top_fraction", metric, "lift_vs_baseline"],
        ascending=[True, True, False, False],
    )
    return sorted_rows.groupby(["label", "top_fraction"], as_index=False).head(1).reset_index(drop=True)


def top_ranked_context_report(
    score_table: pd.DataFrame,
    score_col: str,
    context_cols: Sequence[str] = ("primary_host", "top_processes"),
    label_col: str | None = None,
    top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
) -> pd.DataFrame:
    """Summarize host/process concentration in the highest-scoring rows."""

    if score_col not in score_table.columns:
        raise ValueError(f"score_table does not contain score column {score_col!r}")

    available_context = [col for col in context_cols if col in score_table.columns]
    work = score_table.sort_values(score_col, ascending=False).copy()

    rows = []
    for fraction in top_fractions:
        top_n = max(1, int(np.ceil(len(work) * fraction)))
        top = work.head(top_n)
        row: dict[str, object] = {
            "score": score_col,
            "top_fraction": float(fraction),
            "top_n": int(top_n),
        }
        if label_col is not None and label_col in top.columns:
            labels = _binary_labels(top[label_col])
            row["hits"] = int(labels.sum())
            row["top_positive_rate"] = float(labels.mean()) if len(labels) else np.nan

        for col in available_context:
            values = top[col].dropna().astype(str)
            values = values[values.str.len() > 0]
            prefix = col.replace(" ", "_")
            if prefix.startswith("top_"):
                prefix = prefix[4:]
            row[f"distinct_{prefix}"] = int(values.nunique())
            if values.empty:
                row[f"top_{prefix}"] = ""
                row[f"top_{prefix}_rows"] = 0
                row[f"top_{prefix}_share"] = np.nan
            else:
                counts = values.value_counts()
                row[f"top_{prefix}"] = counts.index[0]
                row[f"top_{prefix}_rows"] = int(counts.iloc[0])
                row[f"top_{prefix}_share"] = float(counts.iloc[0] / top_n)
        rows.append(row)

    return pd.DataFrame(rows)


def mean_pool_by_group(
    embeddings: np.ndarray,
    group_ids: Sequence[object],
) -> tuple[np.ndarray, pd.Index]:
    """Mean-pool row embeddings by a group id, such as a session id."""

    X = np.asarray(embeddings, dtype=np.float32)
    groups = pd.Index(group_ids)
    if len(X) != len(groups):
        raise ValueError("embeddings and group_ids must have the same length")

    frame = pd.DataFrame({"group": groups})
    grouped = frame.groupby("group", sort=True).indices
    pooled = []
    ordered_groups = []
    for group, row_indices in grouped.items():
        ordered_groups.append(group)
        pooled.append(X[np.fromiter(row_indices, dtype=np.int64)].mean(axis=0))
    return np.vstack(pooled).astype(np.float32, copy=False), pd.Index(ordered_groups)


def _binary_labels(labels: Sequence[object]) -> np.ndarray:
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)

    clean = series.astype("string")
    present = clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")
    return present.to_numpy(dtype=np.int64)


def _percent_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def _safe_nanmean(values: np.ndarray) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.nanmean(values))
