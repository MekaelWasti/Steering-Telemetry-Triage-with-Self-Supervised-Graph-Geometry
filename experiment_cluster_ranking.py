"""Focused experiment: improve LABEL-FREE cluster ranking.

One metric: red-team recall at a ~17% session-review budget (current operating point).
Compares ranking clusters by different per-session anomaly signals + aggregations,
against the current default (size_adjusted_p95 on rank_ensemble) and an oracle.
Saves one plot: recall vs review budget.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BUDGET = 0.17
SCORES = ["rank_ensemble_score", "isolation_forest_score", "knn_train_distance_score"]
AGGS = {
    "mean": lambda g, col: g[col].mean(),
    "p95": lambda g, col: g[col].quantile(0.95),
    "size_adj_p95": lambda g, col: g[col].quantile(0.95) * np.log1p(len(g)),
}


def cluster_table(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cid, g in s.groupby("cluster_id"):
        row = {"cluster_id": cid, "n": len(g), "red": int(g["red_team"].sum())}
        for col in SCORES:
            for agg_name, fn in AGGS.items():
                row[f"{col}__{agg_name}"] = float(fn(g, col))
        rows.append(row)
    return pd.DataFrame(rows)


def recall_curve(ct: pd.DataFrame, rank_col: str, total_sessions: int, total_red: int):
    ordered = ct.sort_values(rank_col, ascending=False)
    cum_sessions = ordered["n"].cumsum().to_numpy()
    cum_red = ordered["red"].cumsum().to_numpy()
    review_frac = cum_sessions / total_sessions
    recall = cum_red / total_red
    return review_frac, recall


def recall_at_budget(review_frac, recall, budget=BUDGET) -> float:
    idx = np.searchsorted(review_frac, budget, side="left")
    idx = min(idx, len(recall) - 1)
    return float(recall[idx])


def main() -> None:
    s = pd.read_csv("/private/tmp/acme_cluster_lib_50k/session_scores.csv")
    total_sessions = len(s)
    total_red = int(s["red_team"].sum())
    ct = cluster_table(s)

    results = []
    curves = {}
    for col in SCORES:
        for agg in AGGS:
            rank_col = f"{col}__{agg}"
            rf, rc = recall_curve(ct, rank_col, total_sessions, total_red)
            curves[f"{col.replace('_score','')} / {agg}"] = (rf, rc)
            results.append({"signal": col.replace("_score", ""), "agg": agg,
                            "recall_at_17pct": recall_at_budget(rf, rc)})

    # oracle: rank clusters by actual red count
    rf_o, rc_o = recall_curve(ct, "red", total_sessions, total_red)
    curves["ORACLE (by red count)"] = (rf_o, rc_o)

    res = pd.DataFrame(results).sort_values("recall_at_17pct", ascending=False).reset_index(drop=True)
    lookup = {(r.signal, r.agg): r.recall_at_17pct for r in res.itertuples()}
    baseline = lookup[("rank_ensemble", "size_adj_p95")]
    oracle = recall_at_budget(rf_o, rc_o)
    print(f"total sessions={total_sessions:,}  red-team={total_red}  budget={BUDGET:.0%}")
    print(f"\nBASELINE (current default = rank_ensemble / size_adj_p95): recall@17% = {baseline:.3f}")
    print(f"ORACLE (rank by red count):                                 recall@17% = {oracle:.3f}")
    print("\nAll label-free strategies, recall@17% (sorted):")
    print(res.to_string(index=False))

    # plot
    fig, ax = plt.subplots(figsize=(11, 7))
    for name, (rf, rc) in curves.items():
        style = dict(lw=1.4, alpha=0.8)
        if name.startswith("ORACLE"):
            style = dict(lw=2.5, color="black", ls="--")
        elif "rank_ensemble / size_adj_p95" in name:
            style = dict(lw=2.5, color="crimson")
        ax.plot(np.r_[0, rf], np.r_[0, rc], label=name, **style)
    ax.axvline(BUDGET, color="grey", ls=":", lw=1)
    ax.set_xlim(0, 0.5); ax.set_ylim(0, 1)
    ax.set_xlabel("session review budget (fraction of sessions reviewed)")
    ax.set_ylabel("red-team recall")
    ax.set_title("Label-free cluster ranking: red-team recall vs review budget (50k)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.savefig("/tmp/_cluster_ranking.png", dpi=120, bbox_inches="tight")
    print("\nsaved /tmp/_cluster_ranking.png")


if __name__ == "__main__":
    main()
