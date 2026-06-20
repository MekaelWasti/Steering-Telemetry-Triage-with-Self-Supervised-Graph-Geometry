"""Essential scoring + metric primitives for cluster-triage experiments.

This is the tight, correct surface to experiment against.  It deliberately
exposes only the few things that matter, and it operates on a **session review
budget** (the real analyst workload), not on a cluster count.

------------------------------------------------------------------------------
THE THREE PER-SESSION SCORING SIGNALS  (already produced by the detector)
------------------------------------------------------------------------------
Columns on `result.session_scores`:

  knn_train_distance_score   distance from the benign TRAIN manifold.
                             Best at the very top (tiny budgets); answers
                             "how unlike normal is this session?".
  isolation_forest_score     how easily the session is isolated in embedding
                             space.  Broad anomaly signal.
  rank_ensemble_score        rank-average of the two above.  Most stable
                             general-purpose default.

To experiment with a new scorer: add a column to session_scores and pass its
name as `score_col` below.  Nothing else needs to change.

------------------------------------------------------------------------------
THE THREE LABEL-FREE CLUSTER-RANKING AGGREGATIONS
------------------------------------------------------------------------------
Given a per-session score, a cluster gets ONE number so clusters can be ranked:

  "mean"          mean session score in the cluster.  CURRENT WINNER at 50k.
  "p95"           95th percentile score (a few hot sessions lift the cluster).
  "size_adj_p95"  p95 * log1p(n_sessions).  Rewards big clusters; was the old
                  default, but over-weights large benign-heavy clusters.

------------------------------------------------------------------------------
THE ONE CORE METRIC
------------------------------------------------------------------------------
  recall_at_budget(...)   positive-recall after reviewing the top clusters up to
                          a fraction of all SESSIONS (default 17%).  Plus lift.

Everything else (curves, comparison table, plot) is built from that.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

DEFAULT_BUDGET = 0.17
SCORE_SIGNALS = ("rank_ensemble_score", "isolation_forest_score", "knn_train_distance_score")
AGGREGATIONS = ("mean", "p95", "size_adj_p95")


def _binary(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return (pd.to_numeric(s, errors="coerce").fillna(0) > 0).astype(int)
    c = s.astype("string")
    return (c.notna() & c.str.strip().ne("") & c.str.lower().ne("nan")).astype(int)


def cluster_scoreboard(
    session_scores: pd.DataFrame,
    score_col: str,
    label_col: str,
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """One row per cluster: size, positives, and the three aggregate scores."""

    df = session_scores[[cluster_col, score_col]].copy()
    df["_pos"] = _binary(session_scores[label_col]).to_numpy()
    g = df.groupby(cluster_col)
    out = pd.DataFrame({
        "n_sessions": g.size(),
        "positives": g["_pos"].sum(),
        "mean": g[score_col].mean(),
        "p95": g[score_col].quantile(0.95),
    })
    out["size_adj_p95"] = out["p95"] * np.log1p(out["n_sessions"])
    out["positive_rate"] = out["positives"] / out["n_sessions"]
    return out.reset_index()


def recall_budget_curve(
    scoreboard: pd.DataFrame,
    rank_by: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (session_review_fraction, recall) walking clusters top-down."""

    o = scoreboard.sort_values(rank_by, ascending=False)
    total_sessions = float(o["n_sessions"].sum())
    total_pos = float(o["positives"].sum())
    review_frac = (o["n_sessions"].cumsum() / total_sessions).to_numpy()
    recall = (o["positives"].cumsum() / total_pos).to_numpy() if total_pos else np.zeros(len(o))
    # prepend origin for clean plotting
    return np.r_[0.0, review_frac], np.r_[0.0, recall]


def recall_at_budget(
    session_scores: pd.DataFrame,
    score_col: str = "rank_ensemble_score",
    agg: str = "mean",
    label_col: str = "red_team",
    budget: float = DEFAULT_BUDGET,
    cluster_col: str = "cluster_id",
) -> dict[str, float]:
    """THE core metric: recall + lift after reviewing top clusters up to `budget` of sessions."""

    board = cluster_scoreboard(session_scores, score_col, label_col, cluster_col)
    rf, rc = recall_budget_curve(board, agg)
    i = int(np.searchsorted(rf, budget, side="left"))
    i = min(max(i, 1), len(rf) - 1)
    total = len(session_scores)
    base_rate = float(_binary(session_scores[label_col]).mean())
    reviewed_frac = float(rf[i])
    recall = float(rc[i])
    # lift = (positives recovered / sessions reviewed) vs base rate
    reviewed_rate = (recall * base_rate) / reviewed_frac if reviewed_frac else np.nan
    return {
        "score_col": score_col,
        "agg": agg,
        "label": label_col,
        "budget": budget,
        "recall": round(recall, 4),
        "session_review_fraction": round(reviewed_frac, 4),
        "clusters_reviewed": i,
        "lift_vs_baseline": round(reviewed_rate / base_rate, 3) if base_rate else np.nan,
    }


def compare_rankings(
    session_scores: pd.DataFrame,
    label_col: str = "red_team",
    budget: float = DEFAULT_BUDGET,
    score_cols: Sequence[str] = SCORE_SIGNALS,
    aggs: Sequence[str] = AGGREGATIONS,
) -> pd.DataFrame:
    """Grid of recall@budget over (scoring signal x aggregation)."""

    rows = [
        recall_at_budget(session_scores, score_col=sc, agg=ag, label_col=label_col, budget=budget)
        for sc in score_cols
        if sc in session_scores.columns
        for ag in aggs
    ]
    return pd.DataFrame(rows).sort_values("recall", ascending=False).reset_index(drop=True)


def plot_recall_curves(
    session_scores: pd.DataFrame,
    ax,
    label_col: str = "red_team",
    score_cols: Sequence[str] = SCORE_SIGNALS,
    aggs: Sequence[str] = AGGREGATIONS,
    budget: float = DEFAULT_BUDGET,
    include_oracle: bool = True,
):
    """Plot recall vs session-review budget for each (signal, agg). Returns ax."""

    for sc in score_cols:
        if sc not in session_scores.columns:
            continue
        board = cluster_scoreboard(session_scores, sc, label_col)
        for ag in aggs:
            rf, rc = recall_budget_curve(board, ag)
            lw = 2.6 if (sc == "rank_ensemble_score" and ag == "mean") else 1.3
            ax.plot(rf, rc, lw=lw, alpha=0.85, label=f"{sc.replace('_score','')} / {ag}")
    if include_oracle:
        board = cluster_scoreboard(session_scores, "rank_ensemble_score", label_col)
        rf, rc = recall_budget_curve(board, "positives")  # rank clusters by true positives
        ax.plot(rf, rc, "k--", lw=2.2, label="ORACLE (by positives)")
    ax.axvline(budget, color="grey", ls=":", lw=1)
    ax.set_xlim(0, 0.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("session review budget")
    ax.set_ylabel(f"{label_col} recall")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    return ax
