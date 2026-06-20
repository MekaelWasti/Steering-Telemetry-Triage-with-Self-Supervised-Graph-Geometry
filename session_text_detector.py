"""Direct session-text SVD detector.

This is the reusable version of the notebook's stronger session-text idea:

    process text TF-IDF
      -> sparse mean-pool by session
      -> SVD session embeddings
      -> kNN / IsolationForest session anomaly scores

It differs from ``UnsupervisedSessionDetector``:

- ``UnsupervisedSessionDetector``: process TF-IDF -> process SVD -> mean-pool dense process embeddings.
- This module: process TF-IDF -> mean-pool sparse session text -> session SVD.

Both are label-free during fitting/scoring.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler

from semantic_feature_baseline import SemanticFeatureConfig, SemanticProcessFeatureBuilder
from sessionization_quality import (
    SessionQualityThresholds,
    session_quality_issues,
    session_quality_report,
)
from unsupervised_session_detector import EmbeddingAnomalyScorer, EmbeddingScorerConfig


@dataclass
class SessionTextConfig:
    """Configuration for direct session-text embeddings and scoring."""

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
    session_scorer: EmbeddingScorerConfig = field(default_factory=EmbeddingScorerConfig)
    quality_thresholds: SessionQualityThresholds = field(default_factory=SessionQualityThresholds)
    id_col: str = "pid_hash"
    svd_components: int = 64
    random_state: int = 42
    pool_mode: str = "mean"
    label_cols: tuple[str, ...] = ("red_team", "bad_user")
    host_col: str = "hostname"
    user_col: str = "user_name"
    process_name_col: str = "process_name"
    time_cols: tuple[str, ...] = ("process_started", "first_seen", "last_seen")


class SessionTextDetector:
    """Fit and score direct session-text embeddings without labels."""

    def __init__(self, config: SessionTextConfig | None = None):
        self.config = config or SessionTextConfig()
        self.semantic_builder_ = SemanticProcessFeatureBuilder(self.config.semantic)
        self.svd_: TruncatedSVD | None = None
        self.scaler_: StandardScaler | None = None
        self.session_scorer_ = EmbeddingAnomalyScorer(self.config.session_scorer)
        self.feature_names_: list[str] = []
        self.train_session_text_: sparse.csr_matrix | None = None
        self.train_session_embeddings_: np.ndarray | None = None
        self.train_session_ids_: pd.Index | None = None
        self.last_score_session_text_: sparse.csr_matrix | None = None
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
    ) -> "SessionTextDetector":
        ordered_ids = self._ordered_process_ids(df, process_ids)
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

        process_text = self.semantic_builder_.fit_transform(df, ordered_ids)
        session_text, session_index = sparse_pool_by_group(
            process_text,
            aligned_sessions,
            mode=self.config.pool_mode,
        )
        session_embeddings = self._fit_session_embeddings(session_text)
        self.session_scorer_.fit(session_embeddings)

        self.feature_names_ = list(self.semantic_builder_.feature_names_)
        self.train_session_text_ = session_text
        self.train_session_embeddings_ = session_embeddings
        self.train_session_ids_ = session_index
        self.report_ = {
            "semantic_feature_shape": tuple(process_text.shape),
            "session_text_shape": tuple(session_text.shape),
            "session_embedding_shape": tuple(session_embeddings.shape),
            "svd_components": int(self.svd_.n_components),
            "svd_explained_variance": float(np.sum(self.svd_.explained_variance_ratio_)),
            "pool_mode": self.config.pool_mode,
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
        if self.svd_ is None or self.scaler_ is None:
            raise RuntimeError("Call fit before score_sessions")

        ordered_ids = self._ordered_process_ids(df, process_ids)
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

        process_text = self.semantic_builder_.transform(df, ordered_ids)
        session_text, session_index = sparse_pool_by_group(
            process_text,
            aligned_sessions,
            mode=self.config.pool_mode,
        )
        session_embeddings = self._transform_session_embeddings(session_text)
        self.last_score_session_text_ = session_text
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
        self.fit(train_df, train_session_ids, process_ids=train_process_ids)
        return self.score_sessions(score_df, score_session_ids, process_ids=score_process_ids)

    def _fit_session_embeddings(self, session_text: sparse.csr_matrix) -> np.ndarray:
        n_components = effective_svd_components(session_text, self.config.svd_components)
        self.svd_ = TruncatedSVD(n_components=n_components, random_state=self.config.random_state)
        dense = self.svd_.fit_transform(session_text).astype(np.float32, copy=False)
        self.scaler_ = StandardScaler()
        return self.scaler_.fit_transform(dense).astype(np.float32, copy=False)

    def _transform_session_embeddings(self, session_text: sparse.csr_matrix) -> np.ndarray:
        dense = self.svd_.transform(session_text).astype(np.float32, copy=False)
        return self.scaler_.transform(dense).astype(np.float32, copy=False)

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

    def _aligned_process_frame(
        self,
        df: pd.DataFrame,
        process_ids: Sequence[str],
    ) -> pd.DataFrame:
        work = df[df[self.config.id_col].notna()].drop_duplicates(self.config.id_col, keep="first").copy()
        work[self.config.id_col] = work[self.config.id_col].astype(str)
        return work.set_index(self.config.id_col).reindex(pd.Index(process_ids, name=self.config.id_col)).reset_index()

    def _aligned_session_ids(
        self,
        df: pd.DataFrame,
        session_ids: Sequence[object],
        process_ids: Sequence[str],
        supplied_process_ids: Iterable[str] | None,
    ) -> pd.Series:
        if supplied_process_ids is not None:
            if len(session_ids) != len(process_ids):
                raise ValueError("session_ids must align to supplied process_ids")
            return pd.Series(list(session_ids), index=pd.Index(process_ids, name=self.config.id_col))

        if len(session_ids) != len(df):
            raise ValueError("session_ids must align to df rows when process_ids is not supplied")

        work = df[[self.config.id_col]].copy()
        work["_session_id"] = list(session_ids)
        work = work[work[self.config.id_col].notna()].drop_duplicates(self.config.id_col, keep="first")
        work[self.config.id_col] = work[self.config.id_col].astype(str)
        aligned = work.set_index(self.config.id_col).reindex(pd.Index(process_ids, name=self.config.id_col))["_session_id"]
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
            row: dict[str, object] = {"session_id": session_id, "n_processes": int(len(group))}
            for label_col in self.config.label_cols:
                if label_col in group.columns:
                    row[label_col] = int(binary_labels(group[label_col]).max())
            if self.config.host_col in group.columns:
                row["primary_host"] = top_values(group[self.config.host_col], 1)
            if self.config.user_col in group.columns:
                row["top_users"] = top_values(group[self.config.user_col], 3)
            if self.config.process_name_col in group.columns:
                row["top_processes"] = top_values(group[self.config.process_name_col], 5)
            for time_col in self.config.time_cols:
                if time_col in group.columns:
                    ts = pd.to_datetime(group[time_col], errors="coerce", utc=True)
                    if ts.notna().any():
                        row[f"{time_col}_min"] = ts.min()
                        row[f"{time_col}_max"] = ts.max()
            rows.append(row)
        return pd.DataFrame(rows)


def sparse_pool_by_group(
    matrix: sparse.spmatrix,
    group_ids: Sequence[object],
    mode: str = "mean",
) -> tuple[sparse.csr_matrix, pd.Index]:
    """Sparse mean/sum pool rows by group id."""

    if len(group_ids) != matrix.shape[0]:
        raise ValueError("group_ids length must match matrix rows")
    if mode not in {"mean", "sum"}:
        raise ValueError("mode must be 'mean' or 'sum'")

    groups = pd.Index(group_ids)
    codes, uniques = pd.factorize(groups, sort=True)
    row_idx = np.arange(len(groups), dtype=np.int64)
    if mode == "mean":
        counts = np.bincount(codes)
        weights = 1.0 / counts[codes]
    else:
        weights = np.ones(len(groups), dtype=np.float32)

    membership = sparse.csr_matrix(
        (weights.astype(np.float32), (codes, row_idx)),
        shape=(len(uniques), len(groups)),
    )
    pooled = membership @ matrix.tocsr()
    return pooled.tocsr(), pd.Index(uniques)


def effective_svd_components(matrix: sparse.spmatrix, requested: int) -> int:
    if matrix.shape[0] < 2 or matrix.shape[1] < 2:
        return 1
    return max(1, min(requested, matrix.shape[0] - 1, matrix.shape[1] - 1))


def binary_labels(labels: Sequence[object]) -> np.ndarray:
    series = pd.Series(labels)
    if pd.api.types.is_numeric_dtype(series):
        return (pd.to_numeric(series, errors="coerce").fillna(0).to_numpy() > 0).astype(np.int64)

    clean = series.astype("string")
    present = clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")
    return present.fillna(False).to_numpy(dtype=np.int64)


def top_values(series: pd.Series, n: int) -> str:
    clean = series.dropna().astype(str)
    if clean.empty:
        return ""
    return ", ".join(clean.value_counts().head(n).index.tolist())
