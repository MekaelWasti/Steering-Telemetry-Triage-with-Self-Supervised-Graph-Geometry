# Essential Study Guide: Unsupervised Cyber Telemetry Embeddings

Last updated: 2026-06-18

## Headline (Scale-Validated, 2026-06-18)

A grid across seeds 42/7/123 (20k/10k) and seed 42 (50k/25k) settled the open questions. Where older sections below disagree, this wins:

- **Clustering is the core product, not the ranked queue.** IsolationForest session ranking looked great at 20k (5–7.6x lift) but collapsed to ~1x at 50k. MiniBatchKMeans cluster triage scales the other way: at 50k, looser-file-edge clustering recovers ~76% of red-team sessions at ~17% review (4.6x lift); parent_only clustering hits 38% recall at just 3.9% review (9.7x lift).
- **Looser file-resource edges are the default sessionizer.** They pass the quality guardrail at every seed and scale (max_session_fraction ≤ 0.04). The old "reject looser / sessionization is the bottleneck" stance was a 1k small-sample artifact and is retired.
- **Goal:** a reusable, clustering-first detector library whose artifacts can become a research writeup later.

## How This Guide Will Be Maintained

This is the side study guide for the project. I will update it whenever the prototype direction, code modules, metrics, or interpretation changes.

The goal is not to record every experiment. The goal is to keep the critical ideas clear enough that you can study the system while we build it.

## One-Sentence Goal

Build a mostly label-free cyber telemetry system that turns process activity into embeddings, ranks suspicious sessions for analyst review, and groups similar suspicious behavior into interpretable clusters.

## Current Working Thesis

The strongest unsupervised signal so far is semantic process/session text, not graph-only embeddings.

Current best path:

```text
process-ish dataframe
  -> schema adapter
  -> semantic text tokens
  -> TF-IDF
  -> session pooling
  -> Truncated SVD embeddings
  -> kNN / IsolationForest anomaly scores
  -> ranked sessions
  -> MiniBatchKMeans clusters
  -> cluster labels/keyphrases for analyst review
```

Important interpretation:

- Graph structure is still useful, especially for sessionization and context.
- Supervised GraphSAGE proved the graph can learn labels, but it is not the unsupervised winner yet.
- Direct session-text SVD is the current default session embedding.
- Mean-pooled process SVD remains a useful baseline and bridge to future graph/sequence embeddings.

## Current Reusable Modules

| File | Role |
| --- | --- |
| `telemetry_schema.py` | Maps differently named telemetry columns into canonical process fields. This is the future LLM-assisted schema extension point. |
| `semantic_feature_baseline.py` | Builds semantic TF-IDF features from process names, args, paths, files, users, hosts, and ancestor-like context. |
| `unsupervised_telemetry_detector.py` | Process-level semantic SVD embeddings, anomaly scores, kNN checks, and top-k enrichment helpers. |
| `unsupervised_session_detector.py` | Mean-pools process embeddings into session embeddings and scores sessions. |
| `session_text_detector.py` | Current default: pools sparse process text by session before SVD, then scores session embeddings. |
| `session_clustering.py` | Clusters session embeddings and summarizes cluster-level review performance plus keyphrases. |
| `hetero_graph_builder.py` | Builds PyG heterogeneous graphs and bounded graph-derived sessions. |
| `graph_sessions.py` | Smaller pluggable graph sessionizer with parent/resource guardrails. |
| `sessionization_quality.py` | Reports session quality, giant components, and label spread. |
| `smoke_unsupervised_prototype.py` | End-to-end smoke runner for ACME train/test samples. |

## Core Ideas To Study

### 1. Embeddings

An embedding is a numeric vector for a process or session.

A useful embedding places behaviorally similar items near each other. For this project, "useful" means suspicious behavior forms retrievable neighborhoods or high-scoring anomaly pockets without using labels during training.

### 2. Semantic Text Features

We turn telemetry strings into tokens such as:

```text
proc=powershell.exe
arg=encodedcommand
path=windows/system32
file=ntds.dit
ancestor_path=winword.exe>cmd.exe>powershell.exe
```

Why this works:

- Process names and command lines contain attacker tradecraft.
- Paths and files expose unusual targets.
- Ancestor-like context captures a lightweight process-tree signal.
- TF-IDF highlights rare, meaningful tokens.
- SVD compresses sparse text into dense vectors without labels.

### 3. Direct Session-Text SVD

There are two session embedding choices:

```text
Mean-pooled process SVD:
process text -> process TF-IDF/SVD -> average process vectors per session

Direct session-text SVD:
process text -> pool sparse TF-IDF by session -> SVD over sessions
```

Current default: direct session-text SVD.

Reason: it lets SVD see the whole session text distribution before compression, and it performed slightly better at broader review budgets in the smoke tests.

### 4. kNN Label Checks

kNN is used only for evaluation, not training.

Question: when a point is suspicious, are its nearest neighbors also suspicious?

Most important fields:

| Metric | Meaning |
| --- | --- |
| `positive_point_neighbor_positive_rate` | How pure suspicious neighborhoods are. Higher is better. |
| `negative_point_neighbor_positive_rate` | How contaminated benign neighborhoods are. Lower is better. |
| `separation_gap_pos_minus_neg` | Difference between those two. Higher is better. |
| `lift_vs_baseline` | How much better the neighborhood is than random prevalence. Higher is better. |

This is more reliable than only looking at UMAP plots.

### 5. Top-K / Review-Budget Metrics

The analyst question is:

```text
If we review only the top 1%, 5%, or 10% of sessions, how many suspicious sessions do we recover?
```

Important fields:

| Metric | Meaning |
| --- | --- |
| `top_positive_rate` | Fraction of reviewed items that are positive. |
| `recall_at_top` | Fraction of all positives recovered in the reviewed set. |
| `lift_vs_baseline` | How much better than random review this ranking is. |

This turns embedding quality into analyst usefulness.

### 6. Sessionization

Sessionization decides which processes belong to one activity.

This is a major bottleneck. A good embedding cannot fix bad session boundaries.

Current guardrails:

- cap high-fanout parent processes
- cap rare-file/resource edges
- avoid same-user temporal cliques until needed
- track `max_session_size`
- track `max_session_fraction`
- track label spread only for evaluation

If a sessionizer creates a giant connected component, session-level results are not trustworthy.

### 7. Clustering

Clustering is downstream of session embeddings:

```text
direct session-text SVD embeddings
  -> MiniBatchKMeans
  -> cluster summary
  -> cluster review metrics
  -> grouped keyphrases / short labels
```

The first clusterer is intentionally simple. Its job is not final topic modeling. Its job is to answer:

- Which clusters should an analyst inspect first?
- How many sessions would that review cover?
- How many suspicious sessions would it recover?
- Are top clusters dominated by one host or process family?
- Do the keyphrases make the cluster understandable?

Current keyphrase groups:

- process names
- command-line args
- file/path terms
- ancestor/context terms
- other terms

Treat cluster labels as draft analyst hints, not polished topic names.

## Current Evidence Snapshot

### Process-Level Holdout

The cleanest process-level result remains semantic text SVD on held-out processes.

Approximate result:

| Label | Positive-neighbor rate | Negative-neighbor contamination | Separation gap |
| --- | ---: | ---: | ---: |
| `bad_user` | 0.886 | 0.016 | 0.870 |
| `red_team` | 0.905 | 0.024 | 0.881 |

Interpretation:

- Suspicious processes land near suspicious processes.
- Benign processes mostly do not land near suspicious processes.
- This is real embedding signal, not only visual separation.

### Session-Level Smoke Tests

Direct session-text SVD is now the default session embedding candidate.

Why:

- It tied the best top-1 percent kNN-style session result in the larger smoke run.
- It slightly beat mean-pooled process SVD at 5 percent and 10 percent review budgets.
- It matches the intuition that sessions should be compressed as sessions, not only as averages of process vectors.

Important caution:

- Session-level conclusions are only valid when session quality passes.
- Watch `max_session_fraction`; giant components can make metrics misleading.

### First Cluster Baseline

MiniBatchKMeans over direct session-text SVD embeddings is mechanically working.

Latest small smoke result:

- 32 clusters
- reviewing the top 20 percent of clusters covered about 15.6 percent of sessions
- recovered about 45.5 percent of red-team sessions
- recovered about 49.8 percent of bad-user sessions
- top clusters had useful host/process concentration
- draft keyphrases were useful but noisy, so grouping/cleaning is now being improved

Interpretation:

- Clustering has analyst-review leverage.
- It is not yet final topic modeling.
- The next useful improvement is interpretability and validation, not a new model family.

## What Not To Over-Interpret

- Do not interpret random or untrained GraphSAGE UMAPs as meaningful.
- Do not claim supervised GraphSAGE proves unsupervised detection works.
- Do not use label columns as training features in unsupervised experiments.
- Do not trust session results if one giant session merges thousands of processes.
- Do not treat UMAP as evaluation; use kNN and review-budget metrics.

## Next Concrete Work

Current next step:

```text
scale the selected bounded session config
  -> run 10k-50k process samples
  -> keep bounded rare-file degree 2 as the default
  -> keep parent-only as the conservative baseline
  -> reject looser rare-file degree 10 unless the giant-component issue is fixed
```

Near-term improvements:

- run the artifact-backed grid at 10k-50k scale
- inspect `grid_summary.csv` and `cluster_summary.csv` in the notebook
- improve cluster labels from grouped keyphrases after the larger run
- later: topic modeling over the best clusters

## The Current Top Three Prototype Tasks

These are the three things to keep doing, in order:

1. **Embedding and ranking:** keep direct session-text SVD as the default unsupervised session embedding, with mean-pooled process SVD as the baseline.
2. **Sessionization guardrails:** compare only bounded session configs that pass `max_session_fraction`; do not chase metrics from giant components.
3. **Cluster triage:** rank clusters with score concentration metrics such as `size_adjusted_p95`, then inspect cluster labels/keyphrases.

## Prototype Smoke Artifacts

The smoke runner can now write durable outputs:

```bash
python3 smoke_unsupervised_prototype.py ... --compact --output-dir /path/to/output
```

Saved files:

- `metadata.json`
- `process_knn_report.csv`
- `mean_pooled_session_scores.csv`
- `direct_session_text_scores.csv`
- `session_enrichment_comparison.csv`
- `best_session_enrichment.csv`
- `clustered_direct_session_scores.csv`
- `cluster_summary.csv`
- `cluster_review_report.csv`
- `cluster_strategy_review_report.csv`

Why this matters:

- The notebook can load tables instead of rerunning everything.
- We can compare runs across settings.
- We can inspect clusters and sessions without growing the notebook.
- The prototype is becoming a reusable pipeline, not only an experiment transcript.

The grid runner compares the three current sessionization configs:

```bash
python3 run_unsupervised_smoke_grid.py --output-root /path/to/grid
```

Configs:

- `parent_only`: no file/resource session edges.
- `bounded_file_degree_2`: current default; rare file/resource edges are capped tightly.
- `looser_file_degree_10`: stress test; useful for seeing failure modes, not a default.

The grid writes per-config artifact folders and one combined `grid_summary.csv`.

## Latest Compact Smoke Test

Command:

```bash
python3 smoke_unsupervised_prototype.py --train-n 1000 --test-benign-n 500 --max-text-features 1500 --svd-components 16 --isolation-estimators 10 --max-clusters 16 --top-clusters 5 --cluster-terms 5 --compact
```

Result:

- Python compile checks passed.
- ACME train/test mapped cleanly through `telemetry_schema.py`.
- Train sessions: 992 from 1,000 sampled processes.
- Test sessions: 1,209 from 1,688 sampled processes.
- Test `max_session_fraction`: about 0.044, under the 0.05 guardrail.
- Session quality issues: none.
- Direct session-text SVD beat mean-pooled process SVD on this small run's best session-ranking rows.
- Direct session-text top 10 percent by IsolationForest found 119 / 726 red-team sessions and 119 / 663 bad-user sessions.
- Compact output mode now keeps the smoke test readable.
- Output artifacts were written and verified at `/private/tmp/acme_smoke_outputs`.

Cluster label check:

- Labels are cleaner than before.
- Example labels: `msedge.exe + --annotation`, `microsoftedgeupdate.exe`, `cleanmgr.exe + temp`, `file/path: windowspowershell`, `file/path: taskhostw.exe`.
- Generic path crumbs such as `rare_file=c:\program` are now filtered out as standalone labels.

Important caveat:

- On this tiny 16-cluster run, ranking clusters by max/mean anomaly score was not reliably aligned with label density.
- That does not invalidate the embeddings; it means cluster ranking needs separate validation.
- The prototype now compares alternate unlabeled cluster ranking strategies:
  - `max_then_mean`
  - `p95_then_mean`
  - `mean_then_p95`
  - `top_decile_count`
  - `top_decile_rate`
  - `size_adjusted_p95`
- On the tiny run, `top_decile_count` and `size_adjusted_p95` recovered far more labeled positives than `max_then_mean`, but they also reviewed more sessions.
- Next cluster work should validate those ranking strategies on larger samples and track `session_review_fraction`, not only cluster count.

## Latest Grid Result

Command:

```bash
python3 run_unsupervised_smoke_grid.py --output-root /private/tmp/acme_smoke_grid --train-n 1000 --test-benign-n 500 --max-text-features 1500 --svd-components 16 --isolation-estimators 10 --max-clusters 16 --top-clusters 5 --cluster-terms 5
```

Result:

| Config | Quality pass | Max session fraction | Max session size | Red-team session recall at 10 percent | Bad-user session recall at 10 percent |
| --- | ---: | ---: | ---: | ---: | ---: |
| `parent_only` | yes | 0.044 | 74 | 0.158 | 0.172 |
| `bounded_file_degree_2` | yes | 0.044 | 74 | 0.164 | 0.179 |
| `looser_file_degree_10` | no | 0.245 | 414 | 0.177 | 0.186 |

Interpretation:

- `bounded_file_degree_2` is the current default because it passes the guardrail and slightly improves over parent-only.
- `parent_only` remains the conservative baseline.
- `looser_file_degree_10` is rejected for now even though its recall is higher, because the giant component makes the session-level result untrustworthy.
- Best cluster-ranking strategy in this small grid was `size_adjusted_p95`, but its review fraction is substantial, so this needs larger validation.

## Minimal Mental Model

If you remember only one thing:

```text
We are not trying to build the fanciest GNN yet.
We are trying to prove that unsupervised embeddings can reduce analyst search space.
Right now, semantic session text + SVD + ranking/clustering is the strongest path.
```

## Glossary

| Term | Plain meaning |
| --- | --- |
| TF-IDF | Gives higher weight to terms that are rare but informative. |
| SVD | Compresses many sparse text features into fewer dense numeric dimensions. |
| Embedding | Numeric representation of a process or session. |
| kNN | Nearest-neighbor check: who lives close to whom in embedding space. |
| IsolationForest | Unsupervised anomaly scorer based on how easy a point is to isolate. |
| Session | A group of related processes treated as one activity. |
| Giant component | A too-large session caused by over-linking processes in a graph. |
| Review budget | The fraction of sessions/clusters an analyst is willing to inspect. |
| Lift | How much better a ranking is than random review. |
| Cluster keyphrase | A short text hint explaining what a cluster seems to contain. |
