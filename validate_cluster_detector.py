"""Full-scale validation of TelemetryClusterDetector through its public API.

Confirms the clean library reproduces the headline scale result
(~76% red-team recall at ~17% review at 50k with file-resource sessionization).
Mirrors the train/test sampling the smoke grid used so numbers are comparable.

    python3 validate_cluster_detector.py --train-n 50000 --test-benign-n 25000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from telemetry_cluster_detector import ClusterDetectorConfig, TelemetryClusterDetector


def numeric_label(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    return pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)


def present_label(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(0, index=df.index, dtype="int64")
    clean = df[col].astype("string")
    return (clean.notna() & clean.str.strip().ne("") & clean.str.lower().ne("nan")).astype(int)


def stratified_test_sample(df: pd.DataFrame, benign_n: int, seed: int) -> pd.DataFrame:
    red = numeric_label(df, "red_team")
    bad = present_label(df, "bad_user")
    benign_mask = red.eq(0) & bad.eq(0)
    benign = df[benign_mask].sample(n=min(benign_n, int(benign_mask.sum())), random_state=seed)
    sample = pd.concat([df[red == 1], df[bad == 1], benign], ignore_index=True)
    return sample.drop_duplicates("pid_hash", keep="first").reset_index(drop=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-path", type=Path, default=Path("Data/data/ACME4/gold/train-process_uber_summary.parquet"))
    p.add_argument("--test-path", type=Path, default=Path("Data/data/ACME4/gold/test-process_uber_summary.parquet"))
    p.add_argument("--train-n", type=int, default=50000)
    p.add_argument("--test-benign-n", type=int, default=25000)
    p.add_argument("--output-dir", type=Path, default=Path("/private/tmp/acme_cluster_lib_50k"))
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    train_raw = pd.read_parquet(args.train_path)
    test_raw = pd.read_parquet(args.test_path)
    train = train_raw.sample(n=min(args.train_n, len(train_raw)), random_state=args.random_state)
    train = train.drop_duplicates("pid_hash", keep="first").reset_index(drop=True)
    test = stratified_test_sample(test_raw, args.test_benign_n, args.random_state)
    print(f"train rows={len(train)}  test rows={len(test)}", flush=True)

    config = ClusterDetectorConfig(random_state=args.random_state, max_clusters=64)
    detector = TelemetryClusterDetector(config)
    print("fitting...", flush=True)
    detector.fit(train)
    print("detecting...", flush=True)
    result = detector.detect(test)

    print(f"\nquality_ok={result.quality_ok}  n_clusters={result.n_clusters}  strategy={result.ranking_strategy}")
    print(f"max_session_fraction={result.quality_report.get('max_session_fraction')}")
    print(f"n_sessions={result.quality_report.get('n_sessions')}  max_session_size={result.quality_report.get('max_session_size')}")

    print("\n=== CLUSTER TRIAGE EVALUATION (labels post-hoc) ===")
    print(result.evaluate_cluster_triage().to_string(index=False))

    print("\n=== SECONDARY: ranked session queue enrichment ===")
    enr = result.evaluate_session_ranking()
    if not enr.empty:
        cols = [c for c in ["label", "top_fraction", "score", "recall_at_top", "lift_vs_baseline"] if c in enr.columns]
        print(enr[cols].to_string(index=False))

    saved = result.save(args.output_dir)
    # also persist embeddings for the notebook to reuse without recompute
    emb_path = Path(args.output_dir) / "session_embeddings.parquet"
    emb_df = pd.DataFrame(result.session_embeddings)
    emb_df.columns = [f"e{i}" for i in range(emb_df.shape[1])]
    emb_df["cluster_id"] = result.session_scores["cluster_id"].to_numpy()
    for label_col in ("red_team", "bad_user"):
        if label_col in result.session_scores.columns:
            emb_df[label_col] = result.session_scores[label_col].to_numpy()
    emb_df.to_parquet(emb_path)
    print(f"\nsaved {len(saved)} artifacts + {emb_path}")
    print("=== VALIDATION DONE ===")


if __name__ == "__main__":
    main()
