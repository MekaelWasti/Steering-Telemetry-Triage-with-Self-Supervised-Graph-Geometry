---
title: "Steering Telemetry Triage with Self-Supervised Graph Geometry"
subtitle: "Turning one analyst-confirmed incident into retrieval of related malicious activity"
---

## TL;DR

- We turn raw process telemetry (ACME4) into a **heterogeneous graph**, learn **session
  embeddings with no labels**, and test whether malicious sessions become *locally coherent*
  and *reviewable*.
- The headline finding: the learned geometry is **weak as a standalone anomaly detector**, but
  **strongly amplifies a single confirmed incident** — one analyst-confirmed malicious session,
  used as a query "anchor," retrieves the rest of the related malicious activity far better than
  raw features or shuffled-graph controls.
- This is framed as a **few-shot triage** result, not a zero-shot / cross-dataset claim.

> _All numbers below were verified by a clean top-to-bottom notebook run on the 100k-process
> slice (`ROWS = 100_000`, 2026-06-24). Notebook outputs, in-notebook prose, and this document agree._

---

## Problem

- Security analysts drown in process telemetry; most of it is benign and unlabeled.
- Supervised detection needs labels that don't exist at scale and don't transfer across campaigns.
- **Question:** can we learn structure from telemetry *without labels*, such that confirming **one**
  malicious session lets us pull back the related ones — a realistic "I found one, show me the rest" workflow?

## Data

- **Source:** ACME4 gold process telemetry (`train-process_uber_summary.parquet`).
- **Slice:** late chronological window, sorted by `process_started`, deduplicated by `pid_hash`,
  tail-sliced for reproducibility. Intended size: **100,000 processes**.
- **Labels:** red-team flag, **used only for evaluation — never for training or ranking.**

## Method

**1. Graph construction (heterogeneous, typed edges)**

- **Nodes:** processes and *rare* files (files with degree 2–15, to drop noisy hub files).
- **Edges (5 relations):**
  - `parent_child` — process spawn lineage
  - `touches` — process → file
  - `same_user_time_window` — processes by the same known user within 5 min
  - `same_host_time_window` — fallback for rows with **no user**, same host within 5 min
  - (file linkage via rare-file sharing)
- **Sessions** (the evaluation unit): grouped by identity, **split after 5 min of inactivity or
  30 min total duration** — this bounds session size and prevents transitive multi-day blobs.
- A session is labeled **malicious** if any process in it carries the red-team flag.

**2. Self-supervised learning**

- **HeteroSAGE** (HeteroConv + SAGEConv) trained on **typed link prediction**: score real edges
  above randomly sampled fake edges. **No labels touch training.**
- Three fixed seeds (42/43/44) → all results report mean ± std across seeds.
- **Final encoder drops the `same_host_time_window` relation** — ablation showed this dense,
  noisy fallback relation *diluted* malicious-session coherence.

**3. Evaluation protocol & controls** — designed to be hard to fool:

- **Real-vs-shuffled control:** same graph topology, edges shuffled → isolates whether *real
  structure* matters vs. just having edges.
- **Feature ablation:** which part of the handcrafted baseline carries signal (node telemetry vs.
  session size vs. edge counts).
- **Label-shuffle control:** confirms recall isn't achievable by chance.
- **Leave-one-anchor-out retrieval:** every malicious session is the query anchor exactly once;
  the anchor is removed from evaluation.
- **Paired Wilcoxon tests** across all anchors vs. each control.

## Experiments

| Method | What it tests |
|---|---|
| `graph_stats` | Handcrafted session aggregates — the minimum bar |
| `node_stats_baseline` | Raw process telemetry, no graph |
| `sage_real_full` | Learned embeddings, all real edges |
| `sage_real_no_host` | **Final model** — real edges minus the noisy same-host relation |
| `sage_shuffled_no_host` | Matched topology control (shuffled edges) |
| One-anchor cosine retrieval | The few-shot steering experiment |
| Discovery → steering | Label-free way to *find* the first anchor (exploratory) |

## Results

**1. Structure matters (grouping)**

- Removing the same-host relation improved real-edge separation to **0.195 ± 0.019**, vs.
  **0.052 ± 0.023** for the matched shuffled-edge control; full real graph **0.176 ± 0.011**.
- → The grouping benefit comes from **real typed topology**, not from edge count or the dense
  same-host relation.

**2. As a standalone anomaly detector, the graph is *not* the winner (honest negative)**

- Raw node statistics remain the better **global outlier** detector — recovering **73.7%** of
  malicious sessions in the top 10% by distance.
- The learned embedding's strength is **local neighborhood coherence**, not global ranking.

**3. Few-shot steering (the headline)**

With one confirmed malicious session as the anchor, ranking all others by cosine similarity:

| Representation | Average Precision | Recall@100 |
|---|---|---|
| **Real SAGE (no-host)** | **0.1396 ± 0.0917** | **0.4659 ± 0.2624** |
| Raw node statistics | 0.0660 | 0.2719 |
| Shuffled-edge SAGE | 0.0308 | 0.1121 |
| Random ranking | 0.0058 | 0.0319 |

- Paired one-sided Wilcoxon over all malicious anchors: **p < 0.0005** against both controls,
  for AP and recall@100.

**4. End-to-end, label-free discovery → steering (exploratory)**

Closing the "but where does the first anchor come from?" gap, using labels only to *simulate* the
analyst's confirmation:

| Stage | Real graph | Raw anomaly baseline | Shuffled-graph control |
|---|---|---|---|
| Reviews to first confirmed anchor (lower is better) | **26** | 86 | 67 |
| Related malicious recovered in next 100 reviews | **10 / 18** | 3 / 18 | 1 / 18 |

- The graph-derived "stable regions" find a confirmable incident ~3× faster than sorting by raw
  anomaly score, and the steered queue then recovers the majority of remaining malicious sessions.
- Treated as **exploratory**: the region hyperparameters were tuned on this slice and must be
  frozen and re-tested on a holdout before this is a confirmatory claim.

**Defensible conclusion:** *self-supervised relational learning produces a malicious-session
geometry that is weak for unsupervised outlier ranking but strongly amplifies a single
analyst-confirmed example into retrieval of related malicious activity.*

## Interactive artifact (for the Quarto embed)

The end-to-end workflow — **telemetry background → priority discovery region → confirmed anchor →
steered retrieval** — is captured in one interactive DataMap:

`artifacts/presentation/discovery_to_steering_datamap.html`

Embed it in Quarto with an iframe:

```{=html}
<iframe src="artifacts/presentation/discovery_to_steering_datamap.html"
        width="100%" height="840" style="border:none;"></iframe>
```

The map subtitle reports the verified end-to-end result: _26 reviews to first confirmation • 10/18
related malicious sessions recovered in the next 100_.

Supporting exports in the same folder: `steering_progression.mp4` / `.gif` (the animated
"one incident steers the search" progression) — use either as a static-fallback poster for the embed.

## Validity: what is tested vs. not yet

_This is a proof-of-idea / progress result. The framing it supports is "here is an idea, and it
appears to work on this slice" — not "this generalizes" and not "this is a detector."_

**Solidly valid (the load-bearing checks pass):**

- **No label leakage into training.** The encoder trains only on typed link prediction; the
  `red_team` label is touched solely in evaluation. The "learned without labels" claim is real.
- **Fair, correctly-wired controls.** The shuffled-edge SAGE uses *matched topology* (same graph,
  scrambled edges), node-stats is a clean "no graph" baseline, and the random reference matches its
  analytic expectation (recall@100 ≈ 100/3130 ≈ 0.032) — a sign the harness isn't silently broken.
- **No self-match inflation.** The anchor is held out of its own retrieval (`keep[anchor] = False`).
- **Not cherry-picked.** Every malicious session is the anchor once (leave-one-anchor-out), over 3 seeds.

**In-sample — keeps the result *suggestive*, not confirmatory:**

1. **No held-out test set.** Evaluation is transductive — the model embeds every session on one
   slice and is scored on those same sessions. Valid for "are related malicious sessions coherent
   in this space," but it does **not** test generalization to unseen data.
2. **The "drop same-host" choice used the eval metric.** That encoder variant was selected because
   it scored best on malicious coherence *on this data* — in-sample model selection, so the headline
   no-host number is mildly optimistic. Freeze it as a decision, then confirm on a holdout.
3. **Optimistic p-values.** The paired Wilcoxon assumes independent pairs; the 19 anchors are likely
   one campaign and therefore correlated. p ≈ 0.0001–0.0005 is real evidence of *an* effect, but not
   19 independent experiments' worth.

**All three caveats collapse into one fix** (already in Next Steps): freeze the architecture →
chronological holdout → second dataset. Stating them up front *strengthens* the pitch — it shows
the result's exact standing and disarms the obvious reviewer questions.

## Limitations

- **Only 19 malicious sessions**, from **one ACME time slice**, **likely one red-team campaign** —
  so high behavioral similarity inflates retrieval and the statistics rest on a small, correlated sample.
- High variance: recall@100 std (±0.26) is more than half the mean — some anchors retrieve very little.
- The **discovery stage** beats the baselines *on this slice*, but is **exploratory** — its
  hyperparameters were tuned on this same slice, and the result is a single deterministic
  realization, so it must be frozen and re-tested on a holdout before being trusted.
- This is a **few-shot triage** result, **not** an unsupervised anomaly-detection or zero-shot claim.

## Next steps

- **Freeze the architecture** and repeat the grouping + one-anchor protocol on a **chronological
  holdout** and a **second telemetry dataset**.
- Require the discovery stage to **beat the raw-anomaly baseline** on held-out data before treating
  it as confirmatory.
- Add agentic / proposed edges only if they beat the same real-vs-shuffled controls **and** improve
  held-out one-anchor retrieval.

## Reproducibility note

- Training is label-free; labels are used only for evaluation coloring and metrics.
- Seeds 42/43/44; `EPOCHS=8`, `HIDDEN=OUT=48`. All metrics reported as mean ± std across seeds.
- Verified by a single clean top-to-bottom run on 2026-06-24 at `ROWS = 100_000` (run time ~62 s):
  100,000 processes → 3,131 sessions → 19 malicious; every metric in this document matches the
  regenerated notebook outputs and the interactive map.
