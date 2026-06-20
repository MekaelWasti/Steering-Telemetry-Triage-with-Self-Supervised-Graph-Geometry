"""Run a small, fixed grid of unsupervised smoke-test configurations.

This keeps the prototype moving without turning the notebook into the control
plane.  Each grid row runs ``smoke_unsupervised_prototype.py`` with one
sessionization configuration, saves full artifacts, and emits a compact summary
CSV for comparison.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class GridConfig:
    name: str
    rare_file_min_degree: int = 2
    rare_file_max_degree: int = 2
    parent_max_children: int = 25
    disable_file_edges: bool = False


DEFAULT_GRID = (
    GridConfig(
        name="parent_only",
        disable_file_edges=True,
        parent_max_children=25,
    ),
    GridConfig(
        name="bounded_file_degree_2",
        rare_file_min_degree=2,
        rare_file_max_degree=2,
        parent_max_children=25,
    ),
    GridConfig(
        name="looser_file_degree_10",
        rare_file_min_degree=2,
        rare_file_max_degree=10,
        parent_max_children=25,
    ),
)


def main() -> None:
    args = parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for config in DEFAULT_GRID:
        run_dir = args.output_root / config.name
        run_dir.mkdir(parents=True, exist_ok=True)
        if not args.summarize_only:
            command = build_command(args, config, run_dir)
            print(f"Running {config.name}")
            print(" ".join(command))
            completed = run_command(command, args.cwd, run_dir, args.show_child_output)
            if completed.returncode != 0:
                print_log_tail(run_dir / "run.log")
                raise SystemExit(f"{config.name} failed with exit code {completed.returncode}")
        run_rows.append(summarize_run(config, run_dir))

    summary = pd.DataFrame(run_rows)
    summary_path = args.output_root / "grid_summary.csv"
    summary.to_csv(summary_path, index=False)
    print("Grid summary:")
    print(summary.to_string(index=False))
    print(f"Saved {summary_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--output-root", type=Path, default=Path("/private/tmp/acme_smoke_grid"))
    parser.add_argument("--train-n", type=int, default=1000)
    parser.add_argument("--test-benign-n", type=int, default=500)
    parser.add_argument("--max-text-features", type=int, default=1500)
    parser.add_argument("--svd-components", type=int, default=16)
    parser.add_argument("--isolation-estimators", type=int, default=10)
    parser.add_argument("--max-clusters", type=int, default=16)
    parser.add_argument("--top-clusters", type=int, default=5)
    parser.add_argument("--cluster-terms", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--show-child-output", action="store_true")
    return parser.parse_args()


def build_command(args: argparse.Namespace, config: GridConfig, run_dir: Path) -> list[str]:
    command = [
        sys.executable,
        "smoke_unsupervised_prototype.py",
        "--train-n",
        str(args.train_n),
        "--test-benign-n",
        str(args.test_benign_n),
        "--max-text-features",
        str(args.max_text_features),
        "--svd-components",
        str(args.svd_components),
        "--isolation-estimators",
        str(args.isolation_estimators),
        "--max-clusters",
        str(args.max_clusters),
        "--top-clusters",
        str(args.top_clusters),
        "--cluster-terms",
        str(args.cluster_terms),
        "--random-state",
        str(args.random_state),
        "--rare-file-min-degree",
        str(config.rare_file_min_degree),
        "--rare-file-max-degree",
        str(config.rare_file_max_degree),
        "--parent-max-children",
        str(config.parent_max_children),
        "--compact",
        "--output-dir",
        str(run_dir),
    ]
    if config.disable_file_edges:
        command.append("--disable-file-edges")
    return command


def run_command(
    command: list[str],
    cwd: Path,
    run_dir: Path,
    show_child_output: bool = False,
) -> subprocess.CompletedProcess[str]:
    if show_child_output:
        return subprocess.run(command, cwd=cwd, text=True)

    completed = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    log_path = run_dir / "run.log"
    log_path.write_text(
        "COMMAND:\n"
        + " ".join(command)
        + "\n\nSTDOUT:\n"
        + completed.stdout
        + "\n\nSTDERR:\n"
        + completed.stderr,
        encoding="utf-8",
    )
    return completed


def summarize_run(config: GridConfig, run_dir: Path) -> dict[str, object]:
    metadata = read_json(run_dir / "metadata.json")
    cluster_strategies = read_csv(run_dir / "cluster_strategy_review_report.csv")
    best_enrichment = read_csv(run_dir / "best_session_enrichment.csv")
    cluster_summary = read_csv(run_dir / "cluster_summary.csv")

    test_graph = metadata.get("test_graph_report", {})
    direct_issues = metadata.get("direct_session_text_score_quality_issues", [])
    row: dict[str, object] = {
        "config": config.name,
        "disable_file_edges": config.disable_file_edges,
        "rare_file_max_degree": config.rare_file_max_degree,
        "parent_max_children": config.parent_max_children,
        "test_n_sessions": test_graph.get("n_sessions"),
        "test_max_session_size": test_graph.get("max_session_size"),
        "test_max_session_fraction": test_graph.get("max_session_fraction"),
        "test_parent_dropped_edges": (
            test_graph.get("sessionization_drops", {}) or {}
        ).get("parent_child_high_fanout_edge_count"),
        "quality_pass": not bool(direct_issues),
        "quality_issues": "; ".join(str(issue) for issue in direct_issues),
        "n_clusters": int(len(cluster_summary)) if not cluster_summary.empty else 0,
    }

    add_best_session_rows(row, best_enrichment)
    add_best_cluster_strategy_rows(row, cluster_strategies)
    return row


def add_best_session_rows(row: dict[str, object], best_enrichment: pd.DataFrame) -> None:
    if best_enrichment.empty:
        return
    for label in ("red_team", "bad_user"):
        subset = best_enrichment[
            (best_enrichment["label"] == label)
            & (best_enrichment["top_fraction"].round(2) == 0.10)
        ]
        if subset.empty:
            continue
        best = subset.sort_values("recall_at_top", ascending=False).iloc[0]
        row[f"{label}_best_session_score"] = best.get("score")
        row[f"{label}_best_session_recall_at_10pct"] = best.get("recall_at_top")
        row[f"{label}_best_session_lift_at_10pct"] = best.get("lift_vs_baseline")


def add_best_cluster_strategy_rows(row: dict[str, object], strategies: pd.DataFrame) -> None:
    if strategies.empty:
        return
    for label in ("red_team", "bad_user"):
        subset = strategies[
            (strategies["label"] == label)
            & (strategies["top_cluster_fraction"].round(2) == 0.20)
        ]
        if subset.empty:
            continue
        best = subset.sort_values(
            ["recall_at_cluster_review", "lift_vs_baseline"],
            ascending=False,
        ).iloc[0]
        row[f"{label}_best_cluster_strategy"] = best.get("ranking_strategy")
        row[f"{label}_best_cluster_recall"] = best.get("recall_at_cluster_review")
        row[f"{label}_best_cluster_review_fraction"] = best.get("session_review_fraction")
        row[f"{label}_best_cluster_lift"] = best.get("lift_vs_baseline")


def read_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def print_log_tail(path: Path, max_chars: int = 4000) -> None:
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8", errors="replace")
    print(text[-max_chars:])


if __name__ == "__main__":
    main()
