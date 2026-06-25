from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
state = np.load(
    ROOT / "tmp/jupyter-notebook/discovery_state.npz",
    allow_pickle=True,
)

real = state["real"]
shuffled = state["shuffled"]
node_stats = state["node_stats"]
labels = state["labels"].astype(str)
y = (labels == "malicious").astype(int)


def rank01(values):
    return pd.Series(values).rank(method="average", pct=True).to_numpy(float)


def knn_indices(X, k):
    _, idx = cKDTree(np.asarray(X, dtype=np.float32)).query(
        X,
        k=min(k + 1, len(X)),
        workers=-1,
    )
    return idx[:, 1:]


def first_hit_rank(scores):
    order = np.argsort(scores)[::-1]
    return int(np.flatnonzero(y[order])[0] + 1)


def consensus_regions(views, k):
    neighbors = [knn_indices(view, k) for view in views]
    regions = []
    stability = np.zeros(len(labels), dtype=float)
    for center in range(len(labels)):
        sets = [set(seed_neighbors[center].tolist()) | {center} for seed_neighbors in neighbors]
        counts = {}
        for members in sets:
            for member in members:
                counts[member] = counts.get(member, 0) + 1
        region = np.asarray(
            sorted(member for member, count in counts.items() if count >= 2),
            dtype=int,
        )
        regions.append(region)
        pairwise = []
        for left in range(len(sets)):
            for right in range(left + 1, len(sets)):
                pairwise.append(
                    len(sets[left] & sets[right]) / max(1, len(sets[left] | sets[right]))
                )
        stability[center] = np.mean(pairwise)
    return regions, stability


def select_regions(regions, scores, n_regions=8, max_overlap=0.35):
    selected = []
    for center in np.argsort(scores)[::-1]:
        members = set(regions[center].tolist())
        if not members:
            continue
        if any(
            len(members & previous) / max(1, len(members | previous)) > max_overlap
            for _, previous in selected
        ):
            continue
        selected.append((int(center), members))
        if len(selected) == n_regions:
            break
    return selected


base_dist, _ = cKDTree(node_stats).query(node_stats, k=16, workers=-1)
base = rank01(base_dist[:, 1:].mean(axis=1))
raw_order = np.argsort(base)[::-1]
print(f"Raw first malicious rank: {first_hit_rank(base)}")
print(f"Raw malicious in top 25/50/100: {y[raw_order[:25]].sum()}/{y[raw_order[:50]].sum()}/{y[raw_order[:100]].sum()}")

rows = []
for view_name, views in (("real", real), ("shuffled", shuffled)):
    for k in (15, 25, 40, 60):
        regions, stability = consensus_regions(views, k)
        for top_n in (3, 5, 10):
            concentration = np.asarray([
                np.sort(base[members])[-min(top_n, len(members)):].mean()
                for members in regions
            ])
            formulas = {
                "concentration": concentration,
                "concentration_x_stability": concentration * (0.5 + 0.5 * stability),
                "base_plus_concentration": 0.5 * base + 0.5 * concentration,
                "base_support_stability": (
                    0.45 * base + 0.40 * concentration + 0.15 * stability
                ),
            }
            for formula, scores in formulas.items():
                selected = select_regions(regions, scores)
                region_rows = []
                total_reviewed = 0
                first_anchor_after = None
                for rank, (center, member_set) in enumerate(selected, 1):
                    members = np.asarray(sorted(member_set), dtype=int)
                    review_order = members[np.argsort(base[members])[::-1]]
                    malicious_positions = np.flatnonzero(y[review_order])
                    if first_anchor_after is None and len(malicious_positions):
                        first_anchor_after = total_reviewed + int(malicious_positions[0]) + 1
                    total_reviewed += len(review_order)
                    region_rows.append(int(y[members].sum()))
                union = set().union(*(members for _, members in selected))
                rows.append({
                    "view": view_name,
                    "k": k,
                    "top_n": top_n,
                    "formula": formula,
                    "first_anchor_after": first_anchor_after or np.inf,
                    "top_region_malicious": region_rows[0],
                    "top3_regions_malicious": sum(region_rows[:3]),
                    "top5_regions_malicious": sum(region_rows[:5]),
                    "top5_union_size": len(set().union(*(members for _, members in selected[:5]))),
                    "top5_unique_malicious": int(y[list(set().union(*(members for _, members in selected[:5])))].sum()),
                    "selected_centers": [center for center, _ in selected],
                })

results = pd.DataFrame(rows)
summary_cols = [
    "view", "k", "top_n", "formula", "first_anchor_after",
    "top_region_malicious", "top3_regions_malicious",
    "top5_unique_malicious", "top5_union_size",
]
print("\nBest real discovery variants by first anchor, then top-region coverage")
print(
    results[results["view"] == "real"][summary_cols]
    .sort_values(
        ["first_anchor_after", "top_region_malicious", "top5_unique_malicious"],
        ascending=[True, False, False],
    )
    .head(24)
    .to_string(index=False)
)
print("\nMatched real/shuffled comparison for prespecified k=25, top5, base-support-stability")
print(
    results[
        (results["k"] == 25)
        & (results["top_n"] == 5)
        & (results["formula"] == "base_support_stability")
    ][summary_cols].to_string(index=False)
)

# Detailed handoff for the simplest robust method from the diagnostic:
# a stable 40-neighbor region scored by its three strongest anomaly signals.
print("\nMatched real/shuffled comparison for k=40, top3 concentration")
print(
    results[
        (results["k"] == 40)
        & (results["top_n"] == 3)
        & (results["formula"] == "concentration")
    ][summary_cols].to_string(index=False)
)

k = 40
regions, stability = consensus_regions(real, k)
concentration = np.asarray([
    np.sort(base[members])[-min(3, len(members)):].mean()
    for members in regions
])
region_score = concentration
selected = select_regions(regions, region_score)

detail = []
anchor = None
reviewed_before_anchor = []
for region_rank, (center, member_set) in enumerate(selected, 1):
    members = np.asarray(sorted(member_set), dtype=int)
    review_order = members[np.argsort(base[members])[::-1]]
    malicious_positions = np.flatnonzero(y[review_order])
    detail.append({
        "region_rank": region_rank,
        "center": center,
        "region_size": len(members),
        "malicious": int(y[members].sum()),
        "stability": stability[center],
        "score": region_score[center],
        "first_malicious_position": (
            int(malicious_positions[0]) + 1 if len(malicious_positions) else None
        ),
    })
    if anchor is None:
        if len(malicious_positions):
            reviewed_before_anchor.extend(review_order[:malicious_positions[0]].tolist())
            anchor = int(review_order[malicious_positions[0]])
            discovery_region = members
            break
        reviewed_before_anchor.extend(review_order.tolist())

print("\nStable-region detail through first anchor")
print(pd.DataFrame(detail).to_string(index=False))
print(f"Anchor session: {anchor}; reviewed before confirmation: {len(reviewed_before_anchor)}")

if anchor is not None:
    reviewed_sessions = set(reviewed_before_anchor) | {anchor}
    for name, views in (("real", real), ("shuffled", shuffled)):
        seed_scores = []
        for view in views:
            Z = normalize(np.nan_to_num(view))
            seed_scores.append(Z @ Z[anchor])
        similarity = np.mean(seed_scores, axis=0)
        order = np.argsort(similarity)[::-1]
        order = np.asarray([idx for idx in order if idx not in reviewed_sessions], dtype=int)
        hits = int(y[order[:100]].sum())
        recall = hits / (y.sum() - 1)
        print(
            f"{name} consensus steering from discovered anchor: "
            f"hits@100={hits}, recall@100={recall:.3f}"
        )

    Z = normalize(np.nan_to_num(node_stats))
    similarity = Z @ Z[anchor]
    order = np.argsort(similarity)[::-1]
    order = np.asarray([idx for idx in order if idx not in reviewed_sessions], dtype=int)
    hits = int(y[order[:100]].sum())
    print(
        f"node-stat steering from discovered anchor: "
        f"hits@100={hits}, recall@100={hits / (y.sum() - 1):.3f}"
    )


def discover_with_regions(views, k=40, evidence_points=3):
    method_regions, method_stability = consensus_regions(views, k)
    method_scores = np.asarray([
        np.sort(base[members])[-min(evidence_points, len(members)):].mean()
        for members in method_regions
    ])
    selected_regions = select_regions(method_regions, method_scores)
    reviewed = []
    for _, member_set in selected_regions:
        members = np.asarray(sorted(member_set), dtype=int)
        for idx in members[np.argsort(base[members])[::-1]]:
            reviewed.append(int(idx))
            if y[idx]:
                return int(idx), reviewed
    return None, reviewed


def steering_hits(views, method_anchor, already_reviewed, budget=100):
    scores = []
    for view in views:
        Z = normalize(np.nan_to_num(view))
        scores.append(Z @ Z[method_anchor])
    similarity = np.mean(scores, axis=0)
    blocked = set(already_reviewed)
    order = [idx for idx in np.argsort(similarity)[::-1] if idx not in blocked]
    return int(y[np.asarray(order[:budget], dtype=int)].sum())


raw_anchor = int(raw_order[np.flatnonzero(y[raw_order])[0]])
raw_reviewed = raw_order[:first_hit_rank(base)].tolist()
raw_hits = steering_hits(node_stats[None, :, :], raw_anchor, raw_reviewed)
real_anchor, real_reviewed = discover_with_regions(real)
real_hits = steering_hits(real, real_anchor, real_reviewed)
shuffled_anchor, shuffled_reviewed = discover_with_regions(shuffled)
shuffled_hits = steering_hits(shuffled, shuffled_anchor, shuffled_reviewed)

print("\nEnd-to-end method-specific workflows")
print(pd.DataFrame([
    {
        "method": "raw_only",
        "reviews_to_anchor": len(raw_reviewed),
        "anchor": raw_anchor,
        "additional_hits_at_100": raw_hits,
        "malicious_found_total": 1 + raw_hits,
        "total_reviews": len(raw_reviewed) + 100,
    },
    {
        "method": "shuffled_graph",
        "reviews_to_anchor": len(shuffled_reviewed),
        "anchor": shuffled_anchor,
        "additional_hits_at_100": shuffled_hits,
        "malicious_found_total": 1 + shuffled_hits,
        "total_reviews": len(shuffled_reviewed) + 100,
    },
    {
        "method": "real_graph",
        "reviews_to_anchor": len(real_reviewed),
        "anchor": real_anchor,
        "additional_hits_at_100": real_hits,
        "malicious_found_total": 1 + real_hits,
        "total_reviews": len(real_reviewed) + 100,
    },
]).to_string(index=False))
