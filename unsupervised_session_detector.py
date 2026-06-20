"""Session-level unsupervised detector built on process embeddings.

This module is the prototype bridge from process embeddings to analyst-facing
session rankings:

    process dataframe + session ids
      -> semantic process embeddings
      -> mean-pooled session embeddings
      -> kNN / IsolationForest session anomaly scores
      -> session quality report and ranked review table

Labels are optional and are only used in metadata/evaluation columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors

from sessionization_quality import (
    SessionQualityThresholds,
    session_quality_issues,
    session_quality_report,
)
from unsupervised_telemetry_detector import (
    UnsupervisedTelemetryConfig,
    UnsupervisedTelemetryDetector,
    mean_pool_by_group,
)


@dataclass
class EmbeddingScorerConfig:
    """Configuration for scoring an arbitrary embedding matrix."""

    knn_k: int = 15
    isolation_estimators: int = 200
    isolation_contamination: str | float = "auto"
    isolation_n_jobs: int = 1
    random_state: int = 42
    fit_isolation_forest: bool = True


class EmbeddingAnomalyScorer:
    """Fit kNN and optional IsolationForest on embeddings, then score queries."""

    def __init__(self, config: EmbeddingScorerConfig | None = None):
        self.config = config or EmbeddingScorerConfig()
        self.neighbor_model_: NearestNeighbors | None = None
        self.isolation_model_: IsolationForest | None = None
        self.train_embeddings_: np.ndarray | None = None
        self.report_: dict[str, object] = {}

    def fit(self, embeddings: np.ndarray) -> "EmbeddingAnomalyScorer":
        X = np.asarray(embeddings, dtype=np.float32)
        if X.ndim != 2 or len(X) == 0:
            raise ValueError("embeddings must be a non-empty 2D array")

        self.train_embeddings_ = X
        n_neighbors = min(self.config.knn_k, len(X))
        self.neighbor_model_ = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        self.neighbor_model_.fit(X)

        self.isolation_model_ = None
        if self.config.fit_isolation_forest and len(X) >= 2:
            self.isolation_model_ = IsolationForest(
                n_estimators=self.config.isolation_estimators,
                contamination=self.config.isolation_contamination,
                random_state=self.config.random_state,
                n_jobs=self.config.isolation_n_jobs,
            )
            self.isolation_model_.fit(X)

        self.report_ = {
            "n_train_embeddings": int(len(X)),
            "embedding_dim": int(X.shape[1]),
            "knn_k": int(n_neighbors),
            "isolation_forest_fit": self.isolation_model_ is not None,
        }
        return self

    def score(self, embeddings: np.ndarray) -> pd.DataFrame:
        if self.neighbor_model_ is None:
            raise RuntimeError("Call fit before score")

        X = np.asarray(embeddings, dtype=np.float32)
        distances, _ = self.neighbor_model_.kneighbors(X)
        scores = pd.DataFrame({"knn_train_distance_score": distances.mean(axis=1)})
        rank_cols = ["knn_train_distance_rank"]
        scores["knn_train_distance_rank"] = _percent_rank(scores["knn_train_distance_score"])

        if self.isolation_model_ is not None:
            scores["isolation_forest_score"] = -self.isolation_model_.decision_function(X)
            scores["isolation_forest_rank"] = _percent_rank(scores["isolation_forest_score"])
            rank_cols.append("isolation_forest_rank")

        scores["rank_ensemble_score"] = scores[rank_cols].mean(axis=1)
        return scores


@dataclass
class UnsupervisedSessionConfig:
    """Configuration for the session detector prototype."""

    process: UnsupervisedTelemetryConfig = field(default_factory=UnsupervisedTelemetryConfig)
    session_scorer: EmbeddingScorerConfig = field(default_factory=EmbeddingScorerConfig)
    quality_thresholds: SessionQualityThresholds = field(default_factory=SessionQualityThresholds)
    label_cols: tuple[str, ...] = ("red_team", "bad_user")
    host_col: str = "hostname"
    user_col: str = "user_name"
    process_name_col: str = "process_name"
    time_cols: tuple[str, ...] = ("process_started", "first_seen", "last_seen")


class UnsupervisedSessionDetector:
    """Fit process embeddings, pool them by session, and score sessions."""

    def __init__(self, config: UnsupervisedSessionConfig | None = None):
        self.config = config or UnsupervisedSessionConfig()
        self.process_detector_ = UnsupervisedTelemetryDetector(self.config.process)
        self.session_scorer_ = EmbeddingAnomalyScorer(self.config.session_scorer)
        self.train_session_embeddings_: np.ndarray | None = None
        self.train_session_ids_: pd.Index | None = None
        self.last_score_session_embeddings_: np.ndarray | None = None
        self.last_score_session_ids_: pd.Index | None = None
        self.train_quality_report_: dict[str, object] = {}
        self.train_quality_issues_: list[str] = []
        self.last_score_quality_report_: dict[str, object] = {}
        self.last_score_quality_issues_: list[str] = []
        self.report_: dict[str, object] = {}

    def fit(
        self,
        df: pd.DataFrame,
        session_ids: Sequence[object],
        process_ids: Iterable[str] | None = None,
    ) -> "UnsupervisedSessionDetector":
        """Fit process and session scorers without using labels for training."""

        ordered_ids = self.process_detector_._ordered_process_ids(df, process_ids)
        process_embeddings = self.process_detector_.fit_transform(df, ordered_ids)
        aligned_df = self._aligned_process_frame(df, ordered_ids)
        aligned_sessions = self._aligned_session_ids(df, session_ids, ordered_ids, process_ids)

        self.train_quality_report_ = session_quality_report(
            aligned_df,
            aligned_sessions.to_numpy(),
            label_cols=self.config.label_cols,
        )
        self.train_quality_issues_ = session_quality_issues(
            self.train_quality_report_,
            self.config.quality_thresholds,
            label_cols=self.config.label_cols,
        )

        session_embeddings, session_index = mean_pool_by_group(process_embeddings, aligned_sessions)
        self.session_scorer_.fit(session_embeddings)
        self.train_session_embeddings_ = session_embeddings
        self.train_session_ids_ = session_index

        self.report_ = {
            "process_detector": self.process_detector_.report_,
            "session_scorer": self.session_scorer_.report_,
            "train_quality_report": self.train_quality_report_,
            "train_quality_issues": self.train_quality_issues_,
        }
        return self

    def score_sessions(
        self,
        df: pd.DataFrame,
        session_ids: Sequence[object],
        process_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Score sessions in a dataframe with the fitted process/session models."""

        if self.train_session_embeddings_ is None:
            raise RuntimeError("Call fit before score_sessions")

        ordered_ids = self.process_detector_._ordered_process_ids(df, process_ids)
        process_embeddings = self.process_detector_.transform(df, ordered_ids)
        aligned_df = self._aligned_process_frame(df, ordered_ids)
        aligned_sessions = self._aligned_session_ids(df, session_ids, ordered_ids, process_ids)

        self.last_score_quality_report_ = session_quality_report(
            aligned_df,
            aligned_sessions.to_numpy(),
            label_cols=self.config.label_cols,
        )
        self.last_score_quality_issues_ = session_quality_issues(
            self.last_score_quality_report_,
            self.config.quality_thresholds,
            label_cols=self.config.label_cols,
        )

        session_embeddings, session_index = mean_pool_by_group(process_embeddings, aligned_sessions)
        self.last_score_session_embeddings_ = session_embeddings
        self.last_score_session_ids_ = session_index
        scores = self.session_scorer_.score(session_embeddings)
        metadata = self._session_metadata(aligned_df, aligned_sessions, session_index)
        return pd.concat([metadata.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)

    def fit_score(
        self,
        train_df: pd.DataFrame,
        train_session_ids: Sequence[object],
        score_df: pd.DataFrame,
        score_session_ids: Sequence[object],
        train_process_ids: Iterable[str] | None = None,
        score_process_ids: Iterable[str] | None = None,
    ) -> pd.DataFrame:
        """Fit on one split and score another split."""

        self.fit(train_df, train_session_ids, process_ids=train_process_ids)
        return self.score_sessions(score_df, score_session_ids, process_ids=score_process_ids)

    def _aligned_process_frame(
        self,
        df: pd.DataFrame,
        process_ids: Sequence[str],
    ) -> pd.DataFrame:
        id_col = self.config.process.id_col
        work = df[df[id_col].notna()].drop_duplicates(id_col, keep="first").copy()
        work[id_col] = work[id_col].astype(str)
        aligned = work.set_index(id_col).reindex(pd.Index(process_ids, name=id_col)).reset_index()
        return aligned

    def _aligned_session_ids(
        self,
        df: pd.DataFrame,
        session_ids: Sequence[object],
        process_ids: Sequence[str],
        supplied_process_ids: Iterable[str] | None,
    ) -> pd.Series:
        id_col = self.config.process.id_col
        if supplied_process_ids is not None:
            if len(session_ids) != len(process_ids):
                raise ValueError("session_ids must align to supplied process_ids")
            return pd.Series(list(session_ids), index=pd.Index(process_ids, name=id_col))

        if len(session_ids) != len(df):
            raise ValueError("session_ids must align to df rows when process_ids is not supplied")

        work = df[[id_col]].copy()
        work["_session_id"] = list(session_ids)
        work = work[work[id_col].notna()].drop_duplicates(id_col, keep="first")
        work[id_col] = work[id_col].astype(str)
        aligned = work.set_index(id_col).reindex(pd.Index(process_ids, name=id_col))["_session_id"]
        if aligned.isna().any():
            missing = int(aligned.isna().sum())
            raise ValueError(f"{missing} process ids are missing session ids")
        return aligned

    def _session_metadata(
        self,
        process_df: pd.DataFrame,
        session_ids: pd.Series,
        session_index: pd.Index,
    ) -> pd.DataFrame:
        meta = process_df.copy()
        meta["_session_id"] = session_ids.to_numpy()
        rows = []
        for session_id in session_index:
            group = meta[meta["_session_id"] == session_id]
            row: dict[str, object] = {
                "session_id": session_id,
                "n_processes": int(len(group)),
            }
            self._add_label_metadata(row, group)
            self._add_context_metadata(row, group)
            rows.append(row)
        return pd.DataFrame(rows)

    def _add_label_metadata(self, row: dict[str, object], group: pd.DataFrame) -> None:
        for label_col in self.config.label_cols:
            if label_col in group.columns:
                row[label_col] = int(_binary_labels(group[label_col]).max())

    def _add_context_metadata(self, row: dict[str, object], group: pd.DataFrame) -> None:
        if self.config.host_col in group.columns:
            row["primary_host"] = _top_values(group[self.config.host_col], 1)
        if self.config.user_col in group.columns:
            row["top_users"] = _top_values(group[self.config.user_col], 3)
        if self.config.process_name_col in group.columns:
            row["top_processes"] = _top_values(group[self.config.process_name_col], 5)

        for time_col in self.config.time_cols:
            if time_col not in group.columns:
                continue
            ts = pd.to_datetime(group[time_col], errors="coerce", utc=True)
            if ts.notna().any():
                row[f"{time_col}_min"] = ts.min()
                row[f"{time_col}_max"] = ts.max()


def _percent_rank(values: pd.Series) -> pd.Series:
    return values.rank(method="average", pct=True)


def _binary_labels(labels: Sequence[object]) -> np.ndarray:
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)

    clean = series.astype("string")
    present = clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")
    return present.fillna(False).to_numpy(dtype=np.int64)


def _top_values(series: pd.Series, n: int) -> str:
    clean = series.dropna().astype(str)
    if clean.empty:
        return ""
    return ", ".join(clean.value_counts().head(n).index.tolist())
