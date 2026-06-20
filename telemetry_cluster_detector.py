"""Clustering-first unsupervised telemetry detector (library entry point).

This is the reusable, importable form of the prototype.  It wraps the
validated pipeline into a single ``fit`` / ``detect`` API:

    raw process-ish dataframe
      -> schema adapter (telemetry_schema)
      -> bounded sessions with file-resource edges (hetero_graph_builder)
      -> direct session-text SVD embeddings (session_text_detector)
      -> MiniBatchKMeans cluster triage (session_clustering)
      -> ranked cluster review queue + keyphrase labels  [PRIMARY OUTPUT]
      -> per-session anomaly scores                       [SECONDARY OUTPUT]

Why clustering-first: a grid across seeds 42/7/123 (20k/10k) and seed 42
(50k/25k) showed the IsolationForest ranked session queue collapses at scale
(~5-7x lift at 20k -> ~1x at 50k), while MiniBatchKMeans cluster triage scales
the other way (~76% red-team recall at ~17% review at 50k).  The file-resource
("looser") sessionizer passes the session quality guardrail at every seed and
scale, so it is the default here.  See PROTOTYPE_PLAN.md / ESSENTIAL_STUDY_GUIDE.md.

Labels (``red_team`` / ``bad_user``) are NEVER used for fitting, scoring, or
ranking.  They are consumed only by ``evaluate_*`` helpers, after detection, to
produce evidence for a later writeup.

Example:
    from telemetry_cluster_detector import TelemetryClusterDetector
    det = TelemetryClusterDetector().fit(train_df)
    result = det.detect(test_df)
    result.cluster_triage.head(15)      # ranked clusters for an analyst
    result.save("/tmp/run")             # durable, writeup-ready artifacts
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import pandas as pd

from hetero_graph_builder import HeterogeneousGeometricGraphBuilder
from semantic_feature_baseline import SemanticFeatureConfig
from session_clustering import (
    CLUSTER_RANKING_STRATEGIES,
    SessionClusterConfig,
    add_cluster_keyphrases,
    attach_cluster_labels,
    cluster_keyphrase_report,
    cluster_review_report,
    cluster_session_embeddings,
    compare_cluster_review_strategies,
    summarize_clusters,
)
from session_text_detector import SessionTextConfig, SessionTextDetector
from sessionization_quality import SessionQualityThresholds
from telemetry_schema import adapt_telemetry_dataframe
from unsupervised_session_detector import EmbeddingScorerConfig
from unsupervised_telemetry_detector import collect_top_fraction_enrichment


@dataclass
class ClusterDetectorConfig:
    """Scale-validated defaults for the clustering-first detector."""

    # --- sessionization (validated default: file-resource edges) ---
    add_file_edges: bool = True
    rare_file_min_degree: int = 2
    rare_file_max_degree: int = 10
    parent_max_children: int | None = 25

    # --- semantic embeddings ---
    max_text_features: int = 12_000
    svd_components: int = 48
    min_df: int = 2
    ngram_range: tuple[int, int] = (1, 2)
    semantic_rare_file_max_degree: int = 20

    # --- session anomaly scoring (secondary output) ---
    knn_k: int = 15
    isolation_estimators: int = 50
    score_col: str = "rank_ensemble_score"

    # --- clustering (primary output) ---
    n_clusters: int | None = None
    min_clusters: int = 8
    max_clusters: int = 64
    cluster_terms: int = 8
    # mean-of-ensemble ranking; beat size_adjusted_p95 at 50k (recall@17%: 0.87 vs 0.76).
    # See experiment_cluster_ranking.py.
    default_ranking_strategy: str = "mean_then_p95"

    # --- quality gate ---
    max_session_fraction: float = 0.05
    min_positive_sessions: int = 2

    # --- misc ---
    label_cols: tuple[str, ...] = ("red_team", "bad_user")
    id_col: str = "pid_hash"
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.default_ranking_strategy not in CLUSTER_RANKING_STRATEGIES:
            raise ValueError(
                f"default_ranking_strategy must be one of {sorted(CLUSTER_RANKING_STRATEGIES)}"
            )


@dataclass
class ClusterDetectionResult:
    """Output of :meth:`TelemetryClusterDetector.detect`.

    ``cluster_triage`` is the primary analyst artifact: clusters ranked by the
    configured label-free strategy, with keyphrase labels and context.
    """

    cluster_triage: pd.DataFrame
    cluster_summary: pd.DataFrame
    session_scores: pd.DataFrame
    review_report: pd.DataFrame
    strategy_comparison: pd.DataFrame
    quality_report: dict[str, object]
    quality_issues: list[str]
    graph_report: dict[str, object]
    n_clusters: int
    ranking_strategy: str
    # row-aligned with session_scores: the SVD session embedding matrix and ids
    # (exposed for downstream UMAP / datamapplot visualization)
    session_embeddings: "object" = None  # np.ndarray (n_sessions, svd_components)
    session_ids: "object" = None  # pd.Index of session ids

    @property
    def quality_ok(self) -> bool:
        return not self.quality_issues

    def evaluate_cluster_triage(
        self,
        top_cluster_fractions: Sequence[float] = (0.05, 0.10, 0.20),
    ) -> pd.DataFrame:
        """Recall / lift of the configured triage strategy (needs labels)."""

        sort_cols = CLUSTER_RANKING_STRATEGIES[self.ranking_strategy]
        return cluster_review_report(
            self.cluster_summary,
            top_cluster_fractions=top_cluster_fractions,
            sort_cols=sort_cols,
            ranking_strategy=self.ranking_strategy,
        )

    def evaluate_session_ranking(
        self,
        score_col: str = "rank_ensemble_score",
        top_fractions: Sequence[float] = (0.01, 0.05, 0.10),
    ) -> pd.DataFrame:
        """Top-k enrichment of the secondary ranked session queue (needs labels)."""

        if score_col not in self.session_scores.columns:
            return pd.DataFrame()
        return collect_top_fraction_enrichment(
            self.session_scores,
            embedding_name="session_ranked_queue",
            top_fractions=top_fractions,
        )

    def save(self, output_dir: str | Path) -> list[str]:
        """Write writeup-ready artifacts (metadata + CSV tables)."""

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        saved: list[str] = []

        meta = {
            "n_clusters": self.n_clusters,
            "ranking_strategy": self.ranking_strategy,
            "quality_ok": self.quality_ok,
            "quality_issues": self.quality_issues,
            "quality_report": self.quality_report,
            "graph_report": self.graph_report,
        }
        meta_path = out / "metadata.json"
        with meta_path.open("w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, default=str)
        saved.append(str(meta_path))

        tables = {
            "cluster_triage": self.cluster_triage,
            "cluster_summary": self.cluster_summary,
            "session_scores": self.session_scores,
            "cluster_review_report": self.review_report,
            "cluster_strategy_review_report": self.strategy_comparison,
        }
        for name, table in tables.items():
            if table is None or table.empty:
                continue
            path = out / f"{name}.csv"
            table.to_csv(path, index=False)
            saved.append(str(path))
        return saved


class TelemetryClusterDetector:
    """Clustering-first unsupervised detector with a stable fit/detect API."""

    def __init__(self, config: ClusterDetectorConfig | None = None):
        self.config = config or ClusterDetectorConfig()
        self._session_detector = SessionTextDetector(self._session_text_config())
        self.train_graph_report_: dict[str, object] = {}
        self.fitted_ = False

    # ------------------------------------------------------------------ API
    def fit(self, raw_train_df: pd.DataFrame) -> "TelemetryClusterDetector":
        """Fit embeddings + anomaly scorer on (ideally mostly benign) telemetry."""

        df, process_ids, session_ids, report = self._adapt_and_sessionize(raw_train_df)
        self.train_graph_report_ = report
        self._session_detector.fit(df, session_ids, process_ids=process_ids)
        self.fitted_ = True
        return self

    def detect(self, raw_target_df: pd.DataFrame) -> ClusterDetectionResult:
        """Score + cluster target telemetry into an analyst review queue."""

        if not self.fitted_:
            raise RuntimeError("Call fit before detect")

        df, process_ids, session_ids, report = self._adapt_and_sessionize(raw_target_df)
        session_scores = self._session_detector.score_sessions(
            df, session_ids, process_ids=process_ids
        )

        embeddings = self._session_detector.last_score_session_embeddings_
        if embeddings is None or len(embeddings) == 0:
            raise RuntimeError("No session embeddings were produced for the target data")

        cluster_config = self._cluster_config()
        cluster_labels, cluster_model = cluster_session_embeddings(embeddings, cluster_config)
        clustered = attach_cluster_labels(session_scores, cluster_labels)
        summary = summarize_clusters(clustered, cluster_config)
        summary = self._attach_keyphrases(summary, cluster_labels)

        review_report = cluster_review_report(
            summary, sort_cols=self._ranking_sort_cols(), ranking_strategy=self.config.default_ranking_strategy
        )
        strategy_comparison = compare_cluster_review_strategies(summary)
        triage = self._build_triage(summary)

        return ClusterDetectionResult(
            cluster_triage=triage,
            cluster_summary=summary,
            session_scores=clustered,
            review_report=review_report,
            strategy_comparison=strategy_comparison,
            quality_report=self._session_detector.last_score_quality_report_,
            quality_issues=list(self._session_detector.last_score_quality_issues_),
            graph_report=report,
            n_clusters=int(cluster_model.n_clusters),
            ranking_strategy=self.config.default_ranking_strategy,
            session_embeddings=embeddings,
            session_ids=self._session_detector.last_score_session_ids_,
        )

    def fit_detect(
        self, raw_train_df: pd.DataFrame, raw_target_df: pd.DataFrame
    ) -> ClusterDetectionResult:
        return self.fit(raw_train_df).detect(raw_target_df)

    # ------------------------------------------------------------- internals
    def _adapt_and_sessionize(
        self, raw_df: pd.DataFrame
    ) -> tuple[pd.DataFrame, list[str], list[object], dict[str, object]]:
        adapted, _, _ = adapt_telemetry_dataframe(raw_df)
        builder = HeterogeneousGeometricGraphBuilder(
            rare_file_min_degree=self.config.rare_file_min_degree,
            rare_file_max_degree=self.config.rare_file_max_degree,
            parent_max_children=self.config.parent_max_children,
            add_same_user_edges=False,
            add_network_edges=False,
            add_file_edges=self.config.add_file_edges,
        )
        data = builder.build_graph(adapted)
        process_ids = list(data["process"].external_id)
        session_ids = data["process"].session_id.cpu().numpy().tolist()
        return adapted, process_ids, session_ids, builder.report_

    def _attach_keyphrases(
        self, summary: pd.DataFrame, cluster_labels: Sequence[object]
    ) -> pd.DataFrame:
        text = self._session_detector.last_score_session_text_
        names = self._session_detector.feature_names_
        if summary.empty or text is None or not names:
            return add_cluster_keyphrases(summary, pd.DataFrame())
        keyphrases = cluster_keyphrase_report(
            text, cluster_labels, names, top_n_terms=self.config.cluster_terms
        )
        return add_cluster_keyphrases(summary, keyphrases)

    def _build_triage(self, summary: pd.DataFrame) -> pd.DataFrame:
        if summary.empty:
            return summary
        sort_cols = [c for c in self._ranking_sort_cols() if c in summary.columns]
        ordered = summary.sort_values(sort_cols, ascending=False) if sort_cols else summary
        priority_cols = [
            "cluster_id",
            "cluster_label",
            "n_sessions",
            "score_p95",
            "score_p95_size_adjusted",
            "score_max",
            "score_mean",
            "top_primary_host",
            "top_processes",
            "process_keyphrases",
            "arg_keyphrases",
            "file_path_keyphrases",
            "ancestor_keyphrases",
        ]
        # label columns last, evaluation-only
        for label_col in self.config.label_cols:
            priority_cols += [f"{label_col}_sessions", f"{label_col}_rate"]
        cols = [c for c in priority_cols if c in ordered.columns]
        remaining = [c for c in ordered.columns if c not in cols]
        return ordered[cols + remaining].reset_index(drop=True)

    def _ranking_sort_cols(self) -> tuple[str, ...]:
        return CLUSTER_RANKING_STRATEGIES[self.config.default_ranking_strategy]

    def _semantic_config(self) -> SemanticFeatureConfig:
        return SemanticFeatureConfig(
            include_numeric=False,
            include_process_name=True,
            include_args=True,
            include_process_path=True,
            include_file=True,
            include_ancestor_path=True,
            max_text_features=self.config.max_text_features,
            min_df=self.config.min_df,
            ngram_range=self.config.ngram_range,
            rare_file_max_degree=self.config.semantic_rare_file_max_degree,
        )

    def _session_text_config(self) -> SessionTextConfig:
        return SessionTextConfig(
            semantic=self._semantic_config(),
            session_scorer=EmbeddingScorerConfig(
                knn_k=self.config.knn_k,
                isolation_estimators=self.config.isolation_estimators,
                random_state=self.config.random_state,
            ),
            quality_thresholds=SessionQualityThresholds(
                max_session_fraction=self.config.max_session_fraction,
                min_positive_sessions=self.config.min_positive_sessions,
            ),
            svd_components=self.config.svd_components,
            random_state=self.config.random_state,
            label_cols=self.config.label_cols,
            id_col=self.config.id_col,
        )

    def _cluster_config(self) -> SessionClusterConfig:
        return SessionClusterConfig(
            n_clusters=self.config.n_clusters,
            min_clusters=self.config.min_clusters,
            max_clusters=self.config.max_clusters,
            random_state=self.config.random_state,
            score_col=self.config.score_col,
            label_cols=self.config.label_cols,
        )
