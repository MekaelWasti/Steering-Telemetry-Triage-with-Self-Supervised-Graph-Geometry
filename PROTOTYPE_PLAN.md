# Prototype Plan: Modular Unsupervised Cyber Telemetry Detector

Last updated: 2026-06-18

## Current Decision

Stop expanding the notebook as the main artifact. The notebook proved enough: semantic text SVD is the current unsupervised winner.

The prototype should now become a reusable **library** (clustering-first), with artifacts/metrics structured so they can back a research writeup later.

### Scale-Validated Direction (supersedes earlier "ranked-first / reject looser" notes below)

A grid across seeds 42/7/123 at 20k/10k and seed 42 at 50k/25k changed two standing decisions:

- **Product is CLUSTERING-FIRST.** The IsolationForest ranked session queue showed 5–7.6x lift at 20k but collapsed to ~1.0–1.8x (no signal) at 50k — a small-scale mirage. MiniBatchKMeans cluster triage went the other way: at 50k, `looser_file_degree_10` clustering recovers ~76% of red-team sessions at ~17% review (4.6x lift); `parent_only` clustering is ultra-precise (38% recall @ 3.9% review, 9.7x). Cluster triage is the core product; the ranked queue is now a secondary view. (Caveat: isolation-estimators held at 50 across scales; the queue collapse may be partly under-tuning, not demonstrated.)
- **`looser_file_degree_10` is the default sessionizer; "reject looser" is OVERTURNED.** It passes the session quality guardrail at every seed and scale (max_session_fraction ≤ 0.04 < 0.05). The earlier rejection was a small-sample (1k) component-blowup artifact that does not occur at scale. "Sessionization is the bottleneck" no longer holds.

Sections below this point predate the scale validation and are kept for history; where they conflict, this block wins.

`ESSENTIAL_STUDY_GUIDE.md` is maintained alongside this plan as the plain-English study companion: current thesis, critical concepts, result interpretation, and next steps.

Status update:

- Created `telemetry_schema.py` for canonical schema inference/adaptation and future LLM-assisted mappings.
- Created `unsupervised_telemetry_detector.py` for process-level semantic SVD embeddings and scores.
- Created `unsupervised_session_detector.py` for mean-pooled session embeddings and session anomaly scores.
- Created `session_text_detector.py` for direct session-text SVD embeddings.
- Created `session_clustering.py` for MiniBatchKMeans clusters and cluster-level review metrics.
- Added `parent_max_children`, `max_session_fraction`, and sessionization drop reporting to `hetero_graph_builder.py`.
- Created `sessionization_quality.py` for reusable giant-component and label-spread checks.
- Created `smoke_unsupervised_prototype.py` for an end-to-end modular smoke test.
- Added `--compact` and `--output-dir` to the smoke runner so experiments create readable logs and reusable CSV/JSON artifacts.
- Created `run_unsupervised_smoke_grid.py` to compare the top three sessionization configurations and write `grid_summary.csv`.

## Top Three Priorities

### 1. Modular Process Embedding And Scoring

Build a reusable unsupervised detector around the winning process-level path:

```text
process dataframe
  -> semantic text features
  -> TF-IDF
  -> TruncatedSVD
  -> normalized dense embeddings
  -> kNN distance / IsolationForest anomaly scores
  -> ranked review queue
```

Acceptance criteria:

- Fits without using labels.
- Transforms train and test data with the same feature space.
- Produces one embedding per process.
- Produces anomaly scores and ranks.
- Reports kNN label metrics only as evaluation.

Current status: implemented for process embeddings in `UnsupervisedTelemetryDetector`.

### 2. Sessionization Reliability Gate

Before session-level detection is treated as valid, every sessionization run must report:

- number of processes
- number of sessions
- singleton count
- median, p95, and max session size
- max session fraction
- red-team and bad-user label spread for evaluation only

Acceptance criteria:

- No giant session silently passes.
- High-fanout parent edges are capped or dropped.
- Rare artifact edges are degree-capped.
- Session detector results are marked experimental until the max component is controlled.

Current status: implemented as `session_quality_report` and integrated into the session detector flow.

### 3. Analyst-Style Evaluation

Evaluate embeddings by simulated analyst effort:

- top-k enrichment
- recall-at-review-budget
- negative contamination
- host/process concentration of top-ranked items
- kNN neighborhood purity

Acceptance criteria:

- Metrics can compare raw numeric, semantic text SVD, graph embeddings, and session embeddings.
- Labels are used only after fitting/scoring.
- Results answer: "Would an analyst inspect less garbage?"

Current status: top-k enrichment, kNN neighborhood checks, and top-ranked host/process concentration reports are reusable.

## What Is Parked For Later

- More GNN architectures such as GAT, GATv2, or HGT.
- LLM-assisted edge proposal.
- Topic modeling clusters.
- Transformer or GRU sequence models.
- UI/map polish.

These are valuable, but only after the core unsupervised detector and sessionization gate are stable.

## Current Evidence Snapshot

Process-level holdout says semantic text SVD is strong:

- `bad_user` separation gap: about 0.87
- `red_team` separation gap: about 0.88
- negative-neighbor contamination: about 1.5 percent to 2.4 percent

Session-level in-sample direct text SVD is promising:

- top 1 percent red-team sessions: 63 / 68
- lift: about 19x

Session-level holdout is not yet trustworthy:

- current sessionizer creates a 5,623-process test component
- stricter sessionization improves label spread but still has a 3,009-process component

## Immediate Implementation Path

1. Use `UnsupervisedTelemetryDetector` as the process-level baseline module.
2. Use `session_quality_report` on every sessionization experiment.
3. Promote the best session scoring logic out of the notebook.
4. Keep the notebook as an experiment viewer, not the source of truth.

## Prototype Architecture

```text
ACME or other process-ish dataframe
  -> schema adapter into canonical process fields
  -> semantic SVD process embeddings
  -> bounded graph/session builder
  -> session quality gate
  -> pooled session embeddings
  -> kNN / IsolationForest / rank-ensemble scores
  -> analyst review queue metrics
```

Swappable pieces:

- Schema adapter: heuristic synonym mapper now, explicit/LLM-assisted mapper later.
- Embedding source: semantic SVD now, graph/sequence embeddings later.
- Session embedding source: mean-pooled process SVD and direct session-text SVD are both supported.
- Sessionizer: PyG graph builder now, pluggable `GraphSessionizer` later.
- Scorer: kNN distance, IsolationForest, rank ensemble.
- Evaluation: kNN purity, top-k enrichment, recall-at-budget, contamination, host/process concentration.
- Cluster layer: MiniBatchKMeans over direct session-text embeddings, then cluster review metrics.
- Cluster labels: first-pass TF-IDF keyphrases from each cluster's pooled session text.
- Run artifacts: metadata, session scores, enrichment reports, cluster summaries, and cluster strategy comparisons.

## Latest Smoke Test

Command:

```bash
python3 smoke_unsupervised_prototype.py --train-n 3000 --test-benign-n 1500 --max-text-features 3000 --svd-components 24 --isolation-estimators 25
```

Outcome:

- Passed compile checks.
- ACME train/test data passed through the generic schema adapter.
- Train graph: 2,995 sessions, max session size 2.
- Test graph: 2,181 sessions, max session size 74, max session fraction 0.0275.
- Test quality gate had no issues under `max_session_fraction=0.05`.
- Session scores showed above-baseline enrichment for `red_team` and `bad_user`.
- The smoke runner now compares mean-pooled process embeddings against direct session-text embeddings.
- The smoke runner now clusters direct session-text embeddings and reports cluster review performance.

Larger comparison:

```bash
python3 smoke_unsupervised_prototype.py --train-n 10000 --test-benign-n 5000 --max-text-features 8000 --svd-components 32 --isolation-estimators 50
```

Outcome:

- Test session quality passed with `max_session_fraction=0.0121`.
- Direct session-text SVD tied the best top-1 percent kNN result.
- Direct session-text SVD slightly beat mean-pooled process SVD at 5 percent and 10 percent review budgets.
- Current default for session anomaly ranking: direct session-text SVD.

Next validation:

1. Run `run_unsupervised_smoke_grid.py` at 10k to 50k scale.
2. Keep `bounded_file_degree_2` as the default if it keeps passing the session guardrail.
3. Compare cluster-ranking strategies using `grid_summary.csv` and per-run `cluster_strategy_review_report.csv`.

## First Cluster Baseline

Command:

```bash
python3 smoke_unsupervised_prototype.py --train-n 3000 --test-benign-n 1500 --max-text-features 3000 --svd-components 24 --isolation-estimators 25 --max-clusters 32 --top-clusters 10
```

Outcome:

- Direct session-text embeddings clustered into 32 clusters.
- Top 20 percent of clusters reviewed about 15.6 percent of sessions.
- Those clusters recovered about 45.5 percent of red-team sessions.
- Those clusters recovered about 49.8 percent of bad-user sessions.
- Top clusters were host/process-family concentrated, which is useful but should be tracked.
- Cluster summaries now include draft keyphrases; they are informative but still noisy.

Next prototype polish:

- Compact smoke output is available with `--compact`.
- Durable smoke artifacts are available with `--output-dir`.
- Cluster keyphrases are grouped into process, command-line, path/file, and ancestor groups.
- Short cluster labels are generated from the grouped keyphrases.
- Generic path crumbs are filtered from standalone labels.
- Cluster strategy comparison now includes `max_then_mean`, `p95_then_mean`, `mean_then_p95`, `top_decile_count`, `top_decile_rate`, and `size_adjusted_p95`.
- Next: validate cluster ranking on larger samples and track session review fraction alongside recall.

## Latest Artifact-Backed Smoke

Command:

```bash
python3 smoke_unsupervised_prototype.py --train-n 1000 --test-benign-n 500 --max-text-features 1500 --svd-components 16 --isolation-estimators 10 --max-clusters 16 --top-clusters 5 --cluster-terms 5 --compact --output-dir /private/tmp/acme_smoke_outputs
```

Verified outputs:

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

Current read:

- Direct session-text SVD remains the default session embedding candidate.
- Sorting clusters by `max_then_mean` is brittle on the small smoke run.
- `top_decile_count` and `size_adjusted_p95` found more positives, but they reviewed more sessions.
- Larger validation should compare cluster recall against `session_review_fraction`, not only top cluster fraction.

## Latest Grid Smoke

Command:

```bash
python3 run_unsupervised_smoke_grid.py --output-root /private/tmp/acme_smoke_grid --train-n 1000 --test-benign-n 500 --max-text-features 1500 --svd-components 16 --isolation-estimators 10 --max-clusters 16 --top-clusters 5 --cluster-terms 5
```

Verified:

- Per-config artifacts were written under `/private/tmp/acme_smoke_grid`.
- Combined summary was written to `/private/tmp/acme_smoke_grid/grid_summary.csv`.
- `run_unsupervised_smoke_grid.py --summarize-only` regenerated the summary without rerunning models.

Decision from this small grid:

- `bounded_file_degree_2` is the current default sessionizer.
- `parent_only` is the conservative baseline.
- `looser_file_degree_10` is rejected for now because `max_session_fraction=0.245` exceeds the 0.05 guardrail.
- `size_adjusted_p95` is the current leading cluster-ranking strategy to validate at larger scale.

## AI-Assisted Extension Point

The LLM should first be used to propose configuration, not to replace the detector:

```text
raw dataframe columns + sample values
  -> LLM proposes canonical schema mapping
  -> optional LLM proposes extra feature groups / edge types
  -> deterministic adapter validates columns exist
  -> same unsupervised detector runs
```

This keeps the system auditable. The detector still uses TF-IDF/SVD, graph sessions, kNN distance, and IsolationForest; the AI only helps choose how an unfamiliar dataframe maps into those inputs.
