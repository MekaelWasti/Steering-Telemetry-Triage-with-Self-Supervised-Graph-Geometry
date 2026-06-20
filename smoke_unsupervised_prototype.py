"""Smoke-test the modular unsupervised telemetry prototype on ACME data.

This is intentionally small and fast.  It proves the reusable modules can:

1. Build bounded sessions.
2. Fit semantic process embeddings without labels.
3. Pool process embeddings into sessions.
4. Score held-out sessions.
5. Evaluate labels only after scoring.

Example:
    python3 smoke_unsupervised_prototype.py --train-n 5000 --test-benign-n 3000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from hetero_graph_builder import HeterogeneousGeometricGraphBuilder
from semantic_feature_baseline import SemanticFeatureConfig
from session_clustering import (
    SessionClusterConfig,
    add_cluster_keyphrases,
    attach_cluster_labels,
    compare_cluster_review_strategies,
    cluster_keyphrase_report,
    cluster_review_report,
    cluster_session_embeddings,
    summarize_clusters,
)
from session_text_detector import SessionTextConfig, SessionTextDetector
from sessionization_quality import SessionQualityThresholds
from telemetry_schema import adapt_telemetry_dataframe
from unsupervised_session_detector import (
    EmbeddingScorerConfig,
    UnsupervisedSessionConfig,
    UnsupervisedSessionDetector,
)
from unsupervised_telemetry_detector import (
    UnsupervisedTelemetryConfig,
    best_enrichment_by_budget,
    collect_top_fraction_enrichment,
    knn_label_report,
    top_ranked_context_report,
)


def main() -> None:
    args = parse_args()
    train, train_schema_report = read_process_df(args.train_path)
    test, test_schema_report = read_process_df(args.test_path)

    print("Train schema report:")
    print(compact_schema_report(train_schema_report) if args.compact else train_schema_report)
    print("Test schema report:")
    print(compact_schema_report(test_schema_report) if args.compact else test_schema_report)

    train_fit = train.sample(n=min(args.train_n, len(train)), random_state=args.random_state)
    train_fit = train_fit.drop_duplicates("pid_hash", keep="first").reset_index(drop=True)
    test_eval = stratified_test_sample(test, args.test_benign_n, args.random_state)

    train_process_ids, train_session_ids, train_graph_report = build_sessions(
        train_fit,
        rare_file_min_degree=args.rare_file_min_degree,
        rare_file_max_degree=args.rare_file_max_degree,
        parent_max_children=args.parent_max_children,
        add_file_edges=not args.disable_file_edges,
    )
    test_process_ids, test_session_ids, test_graph_report = build_sessions(
        test_eval,
        rare_file_min_degree=args.rare_file_min_degree,
        rare_file_max_degree=args.rare_file_max_degree,
        parent_max_children=args.parent_max_children,
        add_file_edges=not args.disable_file_edges,
    )

    print("Train graph report:")
    print(session_report_slice(train_graph_report))
    print("Test graph report:")
    print(session_report_slice(test_graph_report))

    detector = UnsupervisedSessionDetector(
        UnsupervisedSessionConfig(
            process=UnsupervisedTelemetryConfig(
                semantic=SemanticFeatureConfig(
                    include_numeric=False,
                    include_process_name=True,
                    include_args=True,
                    include_process_path=True,
                    include_file=True,
                    include_ancestor_path=True,
                    max_text_features=args.max_text_features,
                    min_df=2,
                    ngram_range=(1, 2),
                    rare_file_max_degree=20,
                ),
                svd_components=args.svd_components,
                knn_k=args.knn_k,
                isolation_estimators=args.isolation_estimators,
                random_state=args.random_state,
            ),
            session_scorer=EmbeddingScorerConfig(
                knn_k=args.knn_k,
                isolation_estimators=args.isolation_estimators,
                random_state=args.random_state,
            ),
            quality_thresholds=SessionQualityThresholds(
                max_session_fraction=args.max_session_fraction,
                min_positive_sessions=2,
            ),
        )
    )

    session_scores = detector.fit_score(
        train_fit,
        train_session_ids,
        test_eval,
        test_session_ids,
        train_process_ids=train_process_ids,
        score_process_ids=test_process_ids,
    )

    print("Detector report:")
    print(compact_detector_report(detector.report_) if args.compact else detector.report_)
    print("Score quality report:")
    print(
        session_report_slice(detector.last_score_quality_report_)
        if args.compact
        else detector.last_score_quality_report_
    )
    print("Score quality issues:")
    print(detector.last_score_quality_issues_)

    process_embeddings = detector.process_detector_.transform(test_eval, process_ids=test_process_ids)
    aligned_test = test_eval.set_index("pid_hash").reindex(test_process_ids).reset_index()
    process_reports = [
        knn_label_report(process_embeddings, aligned_test["red_team"], "prototype_process_svd", "red_team"),
        knn_label_report(process_embeddings, aligned_test["bad_user"], "prototype_process_svd", "bad_user"),
    ]
    process_report_df = pd.DataFrame(process_reports)
    print("Process kNN label reports:")
    print(compact_knn_report(process_report_df).to_string(index=False))

    pooled_enrichment = print_session_evaluation(
        "Mean-pooled process SVD sessions",
        session_scores,
        compact=args.compact,
    )

    direct_session_detector = SessionTextDetector(
        SessionTextConfig(
            semantic=SemanticFeatureConfig(
                include_numeric=False,
                include_process_name=True,
                include_args=True,
                include_process_path=True,
                include_file=True,
                include_ancestor_path=True,
                max_text_features=args.max_text_features,
                min_df=2,
                ngram_range=(1, 2),
                rare_file_max_degree=20,
            ),
            session_scorer=EmbeddingScorerConfig(
                knn_k=args.knn_k,
                isolation_estimators=args.isolation_estimators,
                random_state=args.random_state,
            ),
            quality_thresholds=SessionQualityThresholds(
                max_session_fraction=args.max_session_fraction,
                min_positive_sessions=2,
            ),
            svd_components=args.svd_components,
            random_state=args.random_state,
        )
    )
    direct_session_scores = direct_session_detector.fit_score(
        train_fit,
        train_session_ids,
        test_eval,
        test_session_ids,
        train_process_ids=train_process_ids,
        score_process_ids=test_process_ids,
    )
    print("Direct session-text detector report:")
    print(
        compact_detector_report(direct_session_detector.report_)
        if args.compact
        else direct_session_detector.report_
    )
    print("Direct session-text score quality report:")
    print(
        session_report_slice(direct_session_detector.last_score_quality_report_)
        if args.compact
        else direct_session_detector.last_score_quality_report_
    )
    print("Direct session-text score quality issues:")
    print(direct_session_detector.last_score_quality_issues_)
    direct_enrichment = print_session_evaluation(
        "Direct session-text SVD sessions",
        direct_session_scores,
        compact=args.compact,
    )

    comparison = pd.concat([pooled_enrichment, direct_enrichment], ignore_index=True)
    print("Best session embedding by label and review budget:")
    best_rows = best_enrichment_by_budget(comparison)[
            [
                "label",
                "top_fraction",
                "embedding",
                "score",
                "top_n",
                "hits",
                "total_positive",
                "top_positive_rate",
                "recall_at_top",
                "lift_vs_baseline",
            ]
        ]
    print(best_rows.to_string(index=False))

    clustered_scores = pd.DataFrame()
    cluster_summary = pd.DataFrame()
    review_report = pd.DataFrame()
    strategy_review_report = pd.DataFrame()
    if direct_session_detector.last_score_session_embeddings_ is not None:
        cluster_config = SessionClusterConfig(
            n_clusters=args.n_clusters if args.n_clusters > 0 else None,
            min_clusters=args.min_clusters,
            max_clusters=args.max_clusters,
            random_state=args.random_state,
            score_col="rank_ensemble_score",
        )
        cluster_labels, cluster_model = cluster_session_embeddings(
            direct_session_detector.last_score_session_embeddings_,
            cluster_config,
        )
        clustered_scores = attach_cluster_labels(direct_session_scores, cluster_labels)
        cluster_summary = summarize_clusters(clustered_scores, cluster_config)
        if (
            direct_session_detector.last_score_session_text_ is not None
            and direct_session_detector.feature_names_
        ):
            keyphrases = cluster_keyphrase_report(
                direct_session_detector.last_score_session_text_,
                cluster_labels,
                direct_session_detector.feature_names_,
                top_n_terms=args.cluster_terms,
            )
            cluster_summary = add_cluster_keyphrases(cluster_summary, keyphrases)
        review_report = cluster_review_report(cluster_summary)
        strategy_review_report = compare_cluster_review_strategies(cluster_summary)

        print("Direct session-text cluster model:")
        print({"n_clusters": int(cluster_model.n_clusters), "score_col": cluster_config.score_col})
        print("Top direct session-text clusters:")
        display_cols = cluster_display_columns(compact=args.compact)
        display_cols = [col for col in display_cols if col in cluster_summary.columns]
        print(cluster_summary[display_cols].head(args.top_clusters).to_string(index=False))
        print("Cluster review report:")
        print(review_report.to_string(index=False))
        print("Cluster ranking strategy comparison:")
        print(compact_strategy_report(strategy_review_report).to_string(index=False))

    if args.output_dir is not None:
        saved = save_outputs(
            args.output_dir,
            metadata={
                "args": vars(args),
                "train_schema_report": train_schema_report,
                "test_schema_report": test_schema_report,
                "train_graph_report": train_graph_report,
                "test_graph_report": test_graph_report,
                "mean_pooled_detector_report": detector.report_,
                "mean_pooled_score_quality_report": detector.last_score_quality_report_,
                "mean_pooled_score_quality_issues": detector.last_score_quality_issues_,
                "direct_session_text_detector_report": direct_session_detector.report_,
                "direct_session_text_score_quality_report": direct_session_detector.last_score_quality_report_,
                "direct_session_text_score_quality_issues": direct_session_detector.last_score_quality_issues_,
            },
            tables={
                "process_knn_report": process_report_df,
                "mean_pooled_session_scores": session_scores,
                "direct_session_text_scores": direct_session_scores,
                "session_enrichment_comparison": comparison,
                "best_session_enrichment": best_rows,
                "clustered_direct_session_scores": clustered_scores,
                "cluster_summary": cluster_summary,
                "cluster_review_report": review_report,
                "cluster_strategy_review_report": strategy_review_report,
            },
        )
        print("Saved smoke output artifacts:")
        for path in saved:
            print(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-path",
        type=Path,
        default=Path("Data/data/ACME4/gold/train-process_uber_summary.parquet"),
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=Path("Data/data/ACME4/gold/test-process_uber_summary.parquet"),
    )
    parser.add_argument("--train-n", type=int, default=5000)
    parser.add_argument("--test-benign-n", type=int, default=3000)
    parser.add_argument("--max-text-features", type=int, default=5000)
    parser.add_argument("--svd-components", type=int, default=32)
    parser.add_argument("--knn-k", type=int, default=15)
    parser.add_argument("--isolation-estimators", type=int, default=50)
    parser.add_argument("--max-session-fraction", type=float, default=0.05)
    parser.add_argument("--rare-file-min-degree", type=int, default=2)
    parser.add_argument("--rare-file-max-degree", type=int, default=2)
    parser.add_argument("--parent-max-children", type=int, default=25)
    parser.add_argument("--disable-file-edges", action="store_true")
    parser.add_argument("--n-clusters", type=int, default=0)
    parser.add_argument("--min-clusters", type=int, default=8)
    parser.add_argument("--max-clusters", type=int, default=64)
    parser.add_argument("--top-clusters", type=int, default=15)
    parser.add_argument("--cluster-terms", type=int, default=8)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def read_process_df(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_parquet(path)
    adapted, _, report = adapt_telemetry_dataframe(df)
    return adapted, report


def stratified_test_sample(df: pd.DataFrame, benign_n: int, random_state: int) -> pd.DataFrame:
    red_label = numeric_label_series(df, "red_team")
    bad_label = present_label_series(df, "bad_user")

    red = df[red_label == 1]
    bad = df[bad_label == 1]
    benign_mask = red_label.eq(0) & bad_label.eq(0)
    if benign_mask.any():
        benign = df[benign_mask].sample(
            n=min(benign_n, int(benign_mask.sum())),
            random_state=random_state,
        )
    else:
        benign = df.sample(n=min(benign_n, len(df)), random_state=random_state)

    sample = pd.concat([red, bad, benign], ignore_index=True)
    return sample.drop_duplicates("pid_hash", keep="first").reset_index(drop=True)


def numeric_label_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def present_label_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        clean = df[col].astype("string")
        return (clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")).astype(int)
    return pd.Series(0, index=df.index, dtype="int64")


def build_sessions(
    df: pd.DataFrame,
    rare_file_min_degree: int = 2,
    rare_file_max_degree: int = 2,
    parent_max_children: int | None = 25,
    add_file_edges: bool = True,
) -> tuple[list[str], list[object], dict[str, object]]:
    builder = HeterogeneousGeometricGraphBuilder(
        rare_file_min_degree=rare_file_min_degree,
        rare_file_max_degree=rare_file_max_degree,
        parent_max_children=parent_max_children,
        add_same_user_edges=False,
        add_network_edges=False,
        add_file_edges=add_file_edges,
    )
    data = builder.build_graph(df)
    return (
        list(data["process"].external_id),
        data["process"].session_id.cpu().numpy().tolist(),
        builder.report_,
    )


def session_report_slice(report: dict[str, object]) -> dict[str, object]:
    keys = [
        "n_sessions",
        "singletons",
        "median_session_size",
        "p95_session_size",
        "max_session_size",
        "max_session_fraction",
        "sessionization_drops",
    ]
    return {key: report.get(key) for key in keys}


def print_session_evaluation(
    name: str,
    session_scores: pd.DataFrame,
    compact: bool = False,
) -> pd.DataFrame:
    print(f"{name} top-k enrichment:")
    enrichment = collect_top_fraction_enrichment(
        session_scores,
        embedding_name=name,
        top_fractions=(0.01, 0.05, 0.10),
    )
    if not enrichment.empty:
        if compact:
            display = best_enrichment_by_budget(enrichment)[
                [
                    "label",
                    "top_fraction",
                    "score",
                    "top_n",
                    "hits",
                    "total_positive",
                    "top_positive_rate",
                    "recall_at_top",
                    "lift_vs_baseline",
                ]
            ]
            print(display.to_string(index=False))
        else:
            print(
                enrichment.sort_values(
                    ["label", "top_fraction", "lift_vs_baseline"],
                    ascending=[True, True, False],
                )
            )

    print(f"{name} top-ranked context concentration:")
    context = top_ranked_context_report(
        session_scores,
        "rank_ensemble_score",
        context_cols=("primary_host", "top_processes"),
        label_col="red_team" if "red_team" in session_scores.columns else None,
        top_fractions=(0.01, 0.05, 0.10),
    )
    if compact:
        cols = [
            "top_fraction",
            "top_n",
            "hits",
            "top_positive_rate",
            "top_primary_host",
            "top_primary_host_share",
            "top_processes",
            "top_processes_share",
        ]
        cols = [col for col in cols if col in context.columns]
        print(context[cols].to_string(index=False))
    else:
        print(context)
    return enrichment


def compact_schema_report(report: dict[str, object]) -> dict[str, object]:
    mapped = report.get("mapped_columns", {})
    mapped_keys = sorted(mapped) if isinstance(mapped, dict) else []
    return {
        "raw_shape": report.get("raw_shape"),
        "adapted_shape": report.get("adapted_shape"),
        "mapped_canonical_fields": mapped_keys,
        "generated_columns": report.get("generated_columns", {}),
        "missing_recommended": report.get("missing_recommended", []),
        "notes": report.get("notes", []),
    }


def compact_detector_report(report: dict[str, object]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key in (
        "semantic_feature_shape",
        "session_text_shape",
        "session_embedding_shape",
        "svd_components",
        "svd_explained_variance",
        "pool_mode",
        "session_scorer",
        "train_quality_issues",
    ):
        if key in report:
            out[key] = report[key]

    process_report = report.get("process_detector")
    if isinstance(process_report, dict):
        out["process_detector"] = {
            key: process_report.get(key)
            for key in (
                "n_train_processes",
                "semantic_feature_shape",
                "embedding_shape",
                "svd_components",
                "svd_explained_variance",
            )
            if key in process_report
        }

    if "train_quality_report" in report:
        out["train_quality_report"] = session_report_slice(report["train_quality_report"])
    return out


def compact_knn_report(report: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "embedding",
        "label",
        "rows",
        "positive_rows",
        "baseline_positive_rate",
        "positive_point_neighbor_positive_rate",
        "negative_point_neighbor_positive_rate",
        "separation_gap_pos_minus_neg",
    ]
    return report[[col for col in cols if col in report.columns]]


def cluster_display_columns(compact: bool = False) -> list[str]:
    cols = [
        "cluster_id",
        "cluster_label",
        "n_sessions",
        "score_max",
        "score_p95",
        "score_mean",
        "score_top_decile_sessions",
        "score_top_decile_rate",
        "red_team_sessions",
        "red_team_rate",
        "bad_user_sessions",
        "bad_user_rate",
        "top_primary_host",
        "top_processes",
        "process_keyphrases",
        "arg_keyphrases",
        "file_path_keyphrases",
        "ancestor_keyphrases",
    ]
    if not compact:
        cols.append("keyphrases")
    return cols


def compact_strategy_report(report: pd.DataFrame) -> pd.DataFrame:
    if report.empty:
        return report
    cols = [
        "label",
        "ranking_strategy",
        "top_cluster_fraction",
        "top_n_clusters",
        "reviewed_sessions",
        "hits",
        "total_positive",
        "session_review_fraction",
        "cluster_positive_rate",
        "recall_at_cluster_review",
        "lift_vs_baseline",
    ]
    cols = [col for col in cols if col in report.columns]
    return (
        report[cols]
        .sort_values(
            ["label", "top_cluster_fraction", "recall_at_cluster_review", "lift_vs_baseline"],
            ascending=[True, True, False, False],
        )
        .groupby(["label", "top_cluster_fraction"], as_index=False)
        .head(3)
        .reset_index(drop=True)
    )


def save_outputs(
    output_dir: Path,
    metadata: dict[str, object],
    tables: dict[str, pd.DataFrame],
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[str] = []

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    saved.append(str(metadata_path))

    for name, table in tables.items():
        if table is None or table.empty:
            continue
        path = output_dir / f"{name}.csv"
        table.to_csv(path, index=False)
        saved.append(str(path))

    return saved


if __name__ == "__main__":
    main()
