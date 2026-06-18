# Refactor to heterogeneous graph

> User context
>
> Attachments:
>
> - how would I refactor this class to be a heteregenous geometric graph using torc…: `~/.codex/attachments/6b776d78-4cc9-413f-a620-86d71fc1ca76/pasted-text.txt`
> - class GeometricGraphBuilder: """ Create a PyTorch Geometric graph from the pass…: `~/.codex/attachments/b6eab42f-a9f6-4286-a8cd-93e7ab261996/pasted-text.txt`

<details><summary>16 previous messages</summary>

> I’ll open the attached snippets first and let them define the shape of the change. Then I’ll check the notebook/repo context so the refactor fits what you already have.
>
> The pasted request is essentially a refactor from a NetworkX-only process graph into a PyTorch Geometric `HeteroData` builder, with process/artifact/user/host-style node types and relation-specific edge types. I’m checking where that class lives so I can either patch the concrete implementation or give you drop-in notebook code if it only exists there.
>
> There’s already a separate `graph_sessions.py` that is a lean sessionizer, but the notebook has the exact `GeometricGraphBuilder` class from your attachment. I’m going to add the PyG heterogeneous version as a clean module rather than surgically mutating notebook JSON, then you can import it into the notebook and keep the old NetworkX builder as a baseline.
>
> Two small things popped out already: the old label lookup says `malicious` even though the node attribute is `labels`, and rare-file edges are built as cliques, which can explode quickly. The hetero version should fix both by keeping labels on `process.y` and representing shared artifacts as bipartite process-to-artifact edges instead of process cliques.
>
> I’m going to make the builder accept both the current single dataframe and optional long-format link tables later. That matters because `process_uber_summary.filename` is summary-ish, while `process_file.parquet` is the richer source when you want real process-file bipartite edges.
>
> PyTorch and PyG are already installed here (`torch 2.11`, `torch_geometric 2.8`), so this can be executable now rather than just a sketch. I’m also folding in the data reality: process-file and process-network are big tables, so the builder needs caps/filters to avoid one common artifact merging half the dataset.
>
> Before editing, I’m checking the worktree so I don’t step on anything unrelated. Then I’ll add the hetero builder as a new file with the old edge ideas translated into typed PyG relations.
>
> I’m adding three pieces in the new module: the `HeteroData` builder, session-id extraction over selected strong relations, and small starter encoder/GRU classes so the graph can flow into GraphSAGE/GAT and then sequence pooling without redesigning the data shape later.
>
> <details><summary>Created a file</summary>
>
> - Wrote `./hetero_graph_builder.py` (+757 -0)
>
> </details>
>
> The code is in place. I’m running a smoke test now: compile the module, build a sample `HeteroData`, and push that sample through the starter GraphSAGE encoder so we catch PyG-shape issues immediately.
>
> The first smoke test passed: the sample graph builds, sessions attach to `process.session_id`, GraphSAGE produces process embeddings, and the GRU pools session sequences. I spotted one edge-case in the optional network-destination helper while reviewing it, so I’m tightening that before the final check.
>
> <details><summary>Edited a file</summary>
>
> - Wrote `./hetero_graph_builder.py` (+15 -4)
>
> </details>
>
> I’m doing one more verification pass with synthetic file and network links. That catches the cases where the richer ACME link tables are passed in instead of relying on the summary dataframe.
>
> The richer-link smoke test passed too: parent-child, rare file, rare remote, host, and user relations all show up as typed PyG edges. I’m doing a quick diff/status pass now and then I’ll summarize the implementation plan and the exact mistakes this refactor fixes.
>
> One last thing: since you specifically mentioned GraphSAGE and GAT, I’m checking both paths. SAGE already ran; now I’m poking the GAT variant so the starter encoder isn’t only theoretically compatible.
>
> Both GraphSAGE and GAT smoke tests pass. I’m not touching the modified notebook, since it was already dirty; the reusable refactor lives in a new importable module.

</details>

Implemented the refactor as a reusable module: [hetero_graph_builder.py](~/Desktop/STAMP/Cyber%20Telemetry/hetero_graph_builder.py:1).

What it gives you:

- `HeterogeneousGeometricGraphBuilder` builds a PyG `HeteroData` graph from process telemetry.
- Node types: `process`, `file`, `remote`, `user`, `host`.
- Edge types: `spawns`, `touches`, `connected_to`, `ran_as`, `ran_on`, plus reverse edges for hetero message passing.
- Sessions are attached as `data["process"].session_id`.
- Rare files/remotes are bipartite nodes instead of process-process cliques, which is the big structural fix.
- Added starter `HeteroGraphEncoder` for GraphSAGE/GAT and `SessionGRUAggregator` for ordered session embeddings.

Use it in the notebook like:

```python
from hetero_graph_builder import (
    HeterogeneousGeometricGraphBuilder,
    HeteroGraphEncoder,
    SessionGRUAggregator,
)

builder = HeterogeneousGeometricGraphBuilder(
    rare_file_max_degree=10,
    add_same_user_edges=False,
)

data = builder.build_graph(
    process_df,
    file_links=process_file_df,      # optional but better
    net_links=process_net_conn_df,   # optional
)

print(data)
print(builder.report_)
```

Then the modeling path is:

```python
encoder = HeteroGraphEncoder(data.metadata(), conv="sage")  # or conv="gat"
node_emb = encoder(data.x_dict, data.edge_index_dict)

session_sequences = builder.get_session_sequences()
gru = SessionGRUAggregator(input_channels=node_emb["process"].shape[1])
session_emb = gru(node_emb["process"], session_sequences)
```

Verdict: this is the right refactor direction for GNN/GRU work. The old class was still thinking like NetworkX: one process node type, string attributes, edge labels, and rare-artifact cliques. For PyG, you want typed nodes and typed relations so GraphSAGE/GAT/HGT can learn different message functions per relationship.

I verified:

- `python3 -m py_compile hetero_graph_builder.py`
- Built sample `HeteroData` from ACME process data
- Ran GraphSAGE encoder
- Ran GAT encoder
- Ran GRU session pooling

I left `my_sessionization.ipynb` untouched because it was already modified in the worktree.

> What do I do with it now, what's the next step in this plan
>
> and update it too please
>
> Where we are:
>
> You currently have a NetworkX process-only graph:
>
> process nodes only
> edges have edge_type="parent_child" / "rare_artifact" / "same_user"
> sessions = connected components
> embed_graph() = manual mean/std pooling, not real graph embedding
>
> Where we want to go:
>
> pandas dataframe
> ↓
> PyTorch Geometric HeteroData graph
> ↓
> heterogeneous GNN encoder
> ↓
> process embeddings
> ↓
> session pooling or GRU over session sequences
> ↓
> session classification / anomaly detection
>
> The refactor plan:
>
> 1. Replace nx.Graph with PyG HeteroData
> 2. Make real node types:
>    process
>    file/artifact
>    user
>    host
> 3. Make real typed edges:
>    process -> process : spawned
>    process -> file : touched
>    file -> process : rev_touched
>    user -> process : ran
>    process -> user : rev_ran
>    host -> process : hosted
>    process -> host : rev_hosted
>    process -> process : same_user_temporal
> 4. Stop making rare-artifact process-process cliques.
>    Instead:
>    process_1 -> rare_file <- process_2
> 5. Store process numerical features in:
>    data["process"].x
> 6. Store labels in:
>    data["process"].y
> 7. Extract sessions separately:
>    use process-process projection
>    connected components
>    assign data["process"].session_id
> 8. First model:
>    Hetero GraphSAGE
>    process node classification baseline
> 9. Then:
>    pool process embeddings by session
>    classify session
> 10. Then:
>     sort process embeddings by time within each session
>     feed sequence into GRU
>     classify session
> 11. Later experiments:
>     GAT / GATv2
>     HGT
>     edge features
>     LLM-assisted edge proposal
>
> The immediate next implementation target:
>
> Build HeteroGeometricGraphBuilder that outputs:
> data = HeteroData()
> data["process"].x
> data["process"].y
> data["file"].x
> data["user"].x
> data["host"].x
> data["process", "spawned", "process"].edge_index
> data["process", "touched", "file"].edge_index
> data["user", "ran", "process"].edge_index
> data["host", "hosted", "process"].edge_index
> data["process", "same_user_temporal", "process"].edge_index
>
> Then run:
>
> model = Hetero GraphSAGE
> out = model(data.x_dict, data.edge_index_dict)
> process_logits = out["process"]
>
> Current status in one sentence:
>
> We have the right graph-construction idea, but the current class is a baseline NetworkX/manual-pooling prototype; the next step is turning it into a real PyG heterogeneous graph builder so GNN/GRU models can actually use structure, node types, and edge types properly.

<details><summary>2 previous messages</summary>

> I’ll ground this in the current repo state first, because the answer changes now that the hetero builder module exists. I’m going to inspect the new builder and notebook references, then I’ll give you the updated next-step plan rather than pretending we’re still before the refactor.
>
> Plan
>
> # Updated Next-Step Plan: From Hetero Graph Builder to First GNN Baseline
>
> ## Summary
>
> The original “build `HeteroData`” step is now effectively done: `hetero_graph_builder.py` exists, compiles, and exposes `HeterogeneousGeometricGraphBuilder`, `HeteroGraphEncoder`, and `SessionGRUAggregator`.
>
> The next step is no longer refactoring the graph class. The next step is to **prove the graph is usable**, then train the first **process-level Hetero GraphSAGE baseline**. Only after that should we move to session pooling and GRU.
>
> Updated pipeline:
>
> ```text
> ACME process dataframe
>   ↓
> HeterogeneousGeometricGraphBuilder
>   ↓
> HeteroData sanity report
>   ↓
> Hetero GraphSAGE process classifier
>   ↓
> process embeddings
>   ↓
> session mean/max pooling baseline
>   ↓
> session classifier
>   ↓
> GRU over time-sorted process embeddings
> ```
>
> ## Key Changes To The Plan
>
> - Treat `hetero_graph_builder.py` as the new graph construction path.
> - Keep the old NetworkX `GeometricGraphBuilder` only as a baseline/reference.
> - Stop using `embed_graph()` as the real embedding method; it is manual feature pooling, not graph learning.
> - Use `data["process"].x`, `data["process"].y`, `data["process"].session_id`, and `builder.get_session_sequences()` as the main interface.
> - Prefer richer link tables next:
>   - `process_uber_summary.parquet` for process nodes/features/labels.
>   - `process_file.parquet` for real process-file edges.
>   - `process_net_conn.parquet` for optional process-remote edges.
>
> ## Immediate Implementation Steps
>
> 1. Add a clean notebook section called “PyG Heterogeneous Graph V1”.
> 2. Load a manageable sample first, e.g. `20_000` to `100_000` process rows.
> 3. Build the hetero graph with:
>    - `add_same_user_edges=False` by default.
>    - `rare_file_max_degree=10`.
>    - `rare_remote_max_degree=20`.
>    - file links enabled if memory allows.
> 4. Print and inspect `builder.report_`.
> 5. Do not train until the graph report passes sanity checks:
>    - no massive session component.
>    - file/remote edge counts are nonzero.
>    - parent-child edges are present when parent rows are in the sample.
>    - `data["process"].y` exists.
>    - `data["process"].session_id` exists.
> 6. Train the first model as a process-node classifier:
>    - model: `HeteroGraphEncoder(..., conv="sage")`.
>    - classifier head: linear layer on `out["process"]`.
>    - target: `data["process"].y`.
>    - split: random process-level train/val/test mask for the first baseline.
> 7. Save process embeddings after training:
>    - `process_emb = out["process"].detach()`.
> 8. Build the first session classifier with simple pooling:
>    - mean-pool process embeddings by `session_id`.
>    - label session malicious if any process in session has `red_team == 1`.
> 9. Only after the pooled session baseline works, run the GRU:
>    - use `builder.get_session_sequences()`.
>    - feed time-sorted process embeddings into `SessionGRUAggregator`.
>    - classify session embeddings.
>
> ## Test Plan
>
> - Graph construction smoke test:
>   - build on 500 rows, 20k rows, then larger sample.
>   - verify `data.metadata()` contains process/file/user/host node types.
>   - verify expected edge types exist.
> - Model smoke test:
>   - run one forward pass through GraphSAGE.
>   - confirm `out["process"].shape[0] == data["process"].num_nodes`.
> - Training smoke test:
>   - overfit a tiny subset for 20-50 steps.
>   - loss should decrease.
> - Session smoke test:
>   - verify every process has exactly one `session_id`.
>   - verify session labels can be derived from process labels.
>   - verify pooled session embedding count equals number of sessions.
>
> ## Assumptions And Defaults
>
> - Use `red_team` as the first supervised label.
> - Use `bad_user` later as a cleaner secondary evaluation target.
> - Keep same-user temporal edges disabled by default because they can create overly broad sessions.
> - Keep host/user edges in the hetero graph for GNN context, but do not use them for session connected components.
> - Start with GraphSAGE before GAT/HGT because it is simpler, faster, and easier to debug.
> - Do not jump to anomaly detection yet; first prove supervised process/session classification can learn from the graph.

</details>

The local state says the “immediate next implementation target” is no longer future work: `hetero_graph_builder.py` exists, compiles, and the notebook already has references to importing it. So the next step should move from “build the hetero graph” to “validate graph quality, then train the first process-level GraphSAGE baseline.”

> okay yes but lets do one main thing at a time
>
> So we got the geometric graph library now let's see if it works, like can we try to take a sample out of the acme df and umap or visualize some of the graph embeddings? so I think that's part of what you have but part of step 5

<details><summary>2 previous messages</summary>

> Yes, exactly: make the next unit of work “does the PyG graph produce useful-looking embeddings on an ACME sample?” I’ll quickly check what visualization/embedding tooling is already available locally so the plan lands on something runnable instead of aspirational.
>
> Plan
>
> # Next Single Step: Visualize ACME Hetero Graph Embeddings
>
> ## Summary
>
> The next thing to do is a **graph embedding visualization smoke test**, not the full classifier pipeline yet.
>
> Goal: prove that `HeterogeneousGeometricGraphBuilder` can take an ACME sample, build a valid PyG `HeteroData` graph, run GraphSAGE, produce process embeddings, and project them into 2D so we can inspect whether malicious/red-team points show any structure.
>
> Important correction: do **not** UMAP random GraphSAGE outputs and interpret them. First train a tiny supervised GraphSAGE process classifier for a few epochs, then visualize the learned hidden process embeddings.
>
> ## Implementation Changes
>
> - Add one notebook section: **“PyG Graph Embedding Visualization Smoke Test”**.
> - Use a stratified ACME sample:
>   - include all or most `red_team == 1` rows from the sample budget.
>   - include a larger benign sample.
>   - add one-hop parent rows for sampled processes when available so `spawned` edges are not accidentally erased.
> - Build the graph with:
>   - `HeterogeneousGeometricGraphBuilder`
>   - `rare_file_max_degree=10`
>   - `add_same_user_edges=False`
>   - start without full `process_file.parquet` if memory is annoying; add it once the basic sample works.
> - Run three visualization baselines:
>   - raw `data["process"].x` projected to 2D.
>   - trained GraphSAGE hidden embeddings projected to 2D.
>   - optionally session-pooled GraphSAGE embeddings projected to 2D after the process plot works.
> - Use UMAP with:
>   - `NUMBA_CACHE_DIR=/private/tmp/numba_cache`
>   - this is needed because local `umap` fails without a writable numba cache path.
> - Color plots by:
>   - `red_team`
>   - `bad_user`
>   - optionally `hostname` or `process_name` for sanity checking.
>
> ## Acceptance Criteria
>
> - `data` prints as valid `HeteroData`.
> - `builder.report_` shows nonzero process nodes and at least some meaningful edges.
> - GraphSAGE forward pass returns `out["process"]` with one embedding per process.
> - Tiny training run loss decreases over 10-30 epochs.
> - UMAP plot renders without crashing.
> - The plot is interpreted only as a smoke test: “does structure appear at all?”, not as final evaluation.
>
> ## Test Plan
>
> - First run on about `5k-10k` processes.
> - If that works, run on about `25k-50k` processes.
> - Compare:
>   - raw-feature UMAP vs GraphSAGE-embedding UMAP.
>   - red-team coloring vs bad-user coloring.
> - Check that red-team points are not all hidden inside one giant benign blob.
> - Check `builder.report_["max_session_size"]` to ensure rare-file or same-user logic did not create a huge component.
>
> ## Assumptions
>
> - The immediate target is **process embedding visualization**, not final session classification.
> - GraphSAGE is the first model; GAT/HGT wait until this works.
> - Session pooling comes after process-level embeddings look mechanically valid.
> - The old NetworkX `embed_graph()` is not used for this step.

</details>

A small but important catch: we should not UMAP random/untrained GNN outputs and read meaning into them. The useful demo is: build a sampled hetero graph, train a tiny GraphSAGE classifier for a few epochs, extract the hidden process embeddings, then project those embeddings. Also, local `umap` currently import-fails, so I’m checking whether that’s fixable by environment variable or whether the plan should use a fallback projection first.

> PLEASE IMPLEMENT THIS PLAN:
>
> # Next Single Step: Visualize ACME Hetero Graph Embeddings
>
> ## Summary
>
> The next thing to do is a **graph embedding visualization smoke test**, not the full classifier pipeline yet.
>
> Goal: prove that `HeterogeneousGeometricGraphBuilder` can take an ACME sample, build a valid PyG `HeteroData` graph, run GraphSAGE, produce process embeddings, and project them into 2D so we can inspect whether malicious/red-team points show any structure.
>
> Important correction: do **not** UMAP random GraphSAGE outputs and interpret them. First train a tiny supervised GraphSAGE process classifier for a few epochs, then visualize the learned hidden process embeddings.
>
> ## Implementation Changes
>
> - Add one notebook section: **“PyG Graph Embedding Visualization Smoke Test”**.
> - Use a stratified ACME sample:
>   - include all or most `red_team == 1` rows from the sample budget.
>   - include a larger benign sample.
>   - add one-hop parent rows for sampled processes when available so `spawned` edges are not accidentally erased.
> - Build the graph with:
>   - `HeterogeneousGeometricGraphBuilder`
>   - `rare_file_max_degree=10`
>   - `add_same_user_edges=False`
>   - start without full `process_file.parquet` if memory is annoying; add it once the basic sample works.
> - Run three visualization baselines:
>   - raw `data["process"].x` projected to 2D.
>   - trained GraphSAGE hidden embeddings projected to 2D.
>   - optionally session-pooled GraphSAGE embeddings projected to 2D after the process plot works.
> - Use UMAP with:
>   - `NUMBA_CACHE_DIR=/private/tmp/numba_cache`
>   - this is needed because local `umap` fails without a writable numba cache path.
> - Color plots by:
>   - `red_team`
>   - `bad_user`
>   - optionally `hostname` or `process_name` for sanity checking.
>
> ## Acceptance Criteria
>
> - `data` prints as valid `HeteroData`.
> - `builder.report_` shows nonzero process nodes and at least some meaningful edges.
> - GraphSAGE forward pass returns `out["process"]` with one embedding per process.
> - Tiny training run loss decreases over 10-30 epochs.
> - UMAP plot renders without crashing.
> - The plot is interpreted only as a smoke test: “does structure appear at all?”, not as final evaluation.
>
> ## Test Plan
>
> - First run on about `5k-10k` processes.
> - If that works, run on about `25k-50k` processes.
> - Compare:
>   - raw-feature UMAP vs GraphSAGE-embedding UMAP.
>   - red-team coloring vs bad-user coloring.
> - Check that red-team points are not all hidden inside one giant benign blob.
> - Check `builder.report_["max_session_size"]` to ensure rare-file or same-user logic did not create a huge component.
>
> ## Assumptions
>
> - The immediate target is **process embedding visualization**, not final session classification.
> - GraphSAGE is the first model; GAT/HGT wait until this works.
> - Session pooling comes after process-level embeddings look mechanically valid.
> - The old NetworkX `embed_graph()` is not used for this step.

<details><summary>7 previous messages</summary>

> Absolutely. I’ll add this as a real notebook section, but first I’m checking the notebook’s current cell layout so I insert it in the right place and don’t trample your existing work.
>
> I’m going to replace the current rough PyG cells with a single focused smoke-test section: sample ACME, build `HeteroData`, train a tiny GraphSAGE classifier, then UMAP raw features vs trained embeddings. I’ll keep it self-contained so you can run that section top-to-bottom in the notebook.
>
> I’m going to make the notebook section runnable in chunks rather than one giant cell: setup, sample, graph build, train, visualize. That way if something is slow, you’ll know exactly which stage you’re in.
>
> The notebook now has the new smoke-test section. I’m validating the JSON and then I’ll run a reduced version of the exact workflow outside the notebook so we know the code path works before you run the larger cells interactively.
>
> The section is inserted and the notebook JSON is valid. I’m running a smaller copy of the same workflow now with a reduced sample/epoch count; this is just to catch shape/API mistakes before you run the full 8k/25-epoch notebook version.
>
> The reduced smoke test passed: it built a hetero graph, trained GraphSAGE for 3 mini smoke epochs with decreasing loss, produced process logits/embeddings, and UMAP returned 2D coordinates. I’m adding one tiny guardrail to the notebook output now so it explicitly warns if a sampled graph creates a suspiciously giant session component.
>
> That guardrail is in. I’ll do one final validation pass on the notebook JSON and summarize exactly where the new section is and what to run.

</details>

Implemented the smoke-test section in [my_sessionization.ipynb](~/Desktop/STAMP/Cyber%20Telemetry/my_sessionization.ipynb).

What changed:

- Added **“PyG Graph Embedding Visualization Smoke Test”** after the graph-builder refactor cells.
- Updated the first import cell to set:
  - `MPLCONFIGDIR=/private/tmp/matplotlib`
  - `NUMBA_CACHE_DIR=/private/tmp/numba_cache`
- Added code to:
  - stratify-sample ACME rows with red-team coverage
  - add one-hop parent rows
  - build `HeteroData`
  - print `builder.report_`
  - warn if the biggest session component is suspiciously large
  - train a tiny supervised GraphSAGE process classifier
  - extract trained process embeddings
  - UMAP raw features vs trained GraphSAGE embeddings
  - color plots by `red_team` and `bad_user`
  - optionally visualize mean-pooled session embeddings

I also ran a reduced external smoke test successfully:

- Built a hetero graph from `1,305` sampled ACME processes
- Got nonzero `spawns`, `ran_on`, `ran_as`, and `touches` edges
- Trained GraphSAGE for 3 quick epochs
- Loss decreased: `0.753 -> 0.553 -> 0.495`
- Produced process logits and embeddings
- UMAP returned valid `(1305, 2)` coordinates

Run cells **12 through 18** in the notebook for this step. Restart the kernel first so the UMAP cache environment variable is set before `umap` imports.

> these are the plots
>
> User context
>
> Images:
>
> - `data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAABYAAAAH/CAYAAADuankpAAAAAXNSR0IArs4c6QAAAERlWElmTU0AKgAAAAgAAYdpAAQAAAABAAAAGgAAAAAAA6ABAAMAAAABAAEAAKACAAQAAAABAAAFgKADAAQAAAABAAAB/wAAAAAcqsErAABAAElEQVR4AeydB5wV1dmH36UuHQFFEAVs2EvsgGLUWKOfMRpj7y22iMbYS2KL3VhjwV5jYu+xoIgKKtg7VaQjHRZ2d77zP3quc9vu3b13633e32/3zpw5c8ozM/ee+c877ymJnBkGAQhAAAIQgAAEIAABCEAAAhCAAAQgAAEIQAACzY5Ai2bXIzoEAQhAAAIQgAAEIAABCEAAAhCAAAQgAAEIQAACngACMCcCBCAAAQhAAAIQgAAEIAABCEAAAhCAAAQgAIFmSgABuJkeWLoFAQhAAAIQgAAEIAABCEAAAhCAAAQgAAEIQAABmHMAAhCAAAQgAAEIQAACEIAABCAAAQhAAAIQgEAzJYAA3EwPLN2CAAQgAAEIQAACEIAABCAAAQhAAAIQgAAEIIAAzDkAAQhAAAIQgAAEIAABCEAAAhCAAAQgAAEIQKCZEkAAbqYHlm5BAAIQgAAEIAABCEAAAhCAAAQgAAEIQAACEEAA5hyAAAQgAAEIQAACEIAABCAAAQhAAAIQgAAEINBMCSAAN9MDS7cgAAEIQAACEIAABCAAAQhAAAIQgAAEIAABCCAAcw5AAAIQgAAEIACBBiIwYcIEKykpsauvvrrOW3DPPff4ulRnddavXz87/PDDE9neeOMNv68+i9kCw/fff7+YMeTV93DOi2Vjsu233970V53pej3ppJOqy9Zot+u61vWNQQACEIAABCBQXAQQgIvreNNbCEAAAhAoIIFbbrnFi2JbbbVVlaVOnz7dzjjjDFtnnXWsffv21qFDB9tss83skksusblz5yb2lfggcWGttdZKpMUXXnnlFb9deR5//PHEpiBKKT3T37vvvpvIywIEINC4CTz00EN2/fXX12kjf/jhB7vooots7NixdVoPhTcPAvVxTjYPUvQCAhCAAAQg0HgJtGq8TaNlEIAABCAAgcZN4MEHH7R+zpNq1KhR9u2339qaa66Z1uDRo0fb7rvvbgsXLrSDDz7YC7/KJA/CK664wt588017+eWXE/uVlpb6slTmlltumUjXgurT9qVLlyalh5W//e1v1r9//7Ca+MzUrsRGFiCQA4HtttvOlixZYm3atMkhN1nyISCx7dNPP7U///nP+RRT5b4SgC+++GL//bXJJptUmZeNEKiPcxLKEIAABCAAAQjULQEE4LrlS+kQgAAEINBMCYwfP95Gjhxp//3vf+24447z4uyFF16Y1Ft59/7ud7+zli1b2pgxY7wHcDzDpZdeanfccUc8ydZYYw0rLy+3hx9+OEkAluj7xBNP2B577GH/+c9/kvYJK7vttpttvvnmYbVePhctWuQ9muulMippMAItWrTwDx8arAFUDIEsBIrlO6hY+pnlMJMMAQhAAAIQgECeBAgBkSdAdocABCAAgeIkIG/cFVZYwQuy++67rxeAU0n861//silTpti1116bJv4qb8+ePe28885L3c0OOOAAe/TRR62ysjKx7ZlnnrHFixfbH/7wh0RaoRbkxfzb3/7WeyLLG1Bexuutt54Xt+N1hFATw4cPtz/96U+20korWZ8+fRJZFBJj/fXXt7Zt21rv3r3txBNPTApxETK+99573ita/BQOY6ONNrIbbrghbPafX375pYlrt27dfHskbD/99NNJeZYvX+69GBUyQ23u3r27DR482BQqI9i0adPsiCOO8O1Uu3r16mX/93//Z4pDWp3l0obAZMSIEXbKKafYiiuuaF27dvUPBZYtW+b7f+ihh/pzRf0988wzLYqijFVfd9111rdvX2vXrp0NGTLEe4GmZsylTdrns88+sx122MGXpWOkcCPx8ymUq7Zom/IoPMmvf/1rv2/YHj4zxQBWyJINNtjAPv/8c7+f9l9llVXsyiuvDLslPidOnGh77bWXP946b0477TR76aWXfMiS6uIKL1iwwHvD6jzVMdT+v/nNb+zDDz9MlB/a8vHHH3t2aos830OoFJ2zCtUitgMGDLD//e9/iX3Dgh7S6CFK586drWPHjrbjjjtaLuFTfvzxR/+wRgy/+uorX1xZWZnpgZDaoDavuuqq/tgrvSpTP5577jkTrxDORf0Olmu5ugZ0LehcVF/U53POOccXI95bbLGFX9a1EeqpLiavvsuOOuoof22rT3rb4IQTTjCd58HGjRtn++23n79udQy23npr35+wvarP1157zbbddlt/jqjduk6/+OKLpF0UtkLt1Tl34IEH+utK/Qz2wAMP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F1/M2IeKigrr3bu3Hze/8MILGfOERJW95557Ws+ePa1NmzbWrVs322677eyaa66x+fPnh2z+s58TVrOdX7vuumtSXlYgAAEIQAACjZFAq8bYKNoEAQgkE/jb3/7mB/BLly61d9991z9tHzFihH366ad+UJ+cm7WGJvDaa695of66666r06ZIANY58Oc//7lO66FwCEAAAhCAQGMmcP/99yc177777rNXXnnFUtPXXXfdpHw1XbnjjjussrKyprvVe36184QTTrAePXrYQQcdZOuss47p7SSNGcRGD4+XLFliLVu2rPe21aTCq6++2v7yl794Afjss8/2AvC3335r//vf/+yRRx6xTMKrxmBTp041CbYPPvig7bbbbmlV6hgeddRRfjy94YYb2p/+9Cf/1taCBQvsnXfesfPOO8+ef/55e/XVV5P2lYB++umnJ6VpRYIzVjMCOv/k2IJBAAIQgED9EeBbt/5YUxMEak1Ag9fNN9/c73/00Uf7Af0//vEPe/rpp70XRK0LbuAdFy9e7AfzDdyMglc/Y8YM69q1a8HLra8Cm+txqS9+1AMBCEAAAvVL4OCDD06qUA/LJQCnpidlcis1/b1r3bp1ahGNbn3kyJFe/B00aJA9++yz1qlTp6Q2yrv10ksvTUrLtFJTNpnKyCdNgvXf//53+81vfmMvv/xyWlEaa2WyBx54wH71q1/ZYYcdZuecc44tWrTIOnTokJT1yiuv9OKv3tASD3n2Bjv11FO9gCyhPNX0Fl5151TqPqxnJiCvdAwCEIAABOqXACEg6pc3tUGgIAS23XZbX853332XKG/ZsmV2wQUX2GabbWZdunTxg13le/311xN5tKBB8T777JOUJu8HDX71umCwRx991Kd98cUXISntM8RxU14NsldeeWVf71577WWTJ09Oyr/99tvbBhtsYB988IF/vU6v8WkfmQbx8sTQK3gaEG688cY+dEJSAW5FHhs33HCDqb3Kt+KKK3rvj/fffz8pqwb/4tCuXTv/Ot8f//jHtPZ888039vvf/963WWX16dPHlG/evHmJsnTzOHjwYC/m6tXJAQMGJNqcyBRbCPHQxPyzzz5LvCoYXjlV++V1s/766/v2q7/HHXec/fjjj7FSzJ566inbY489vEdJ27ZtbY011vA3QXqtMZh4PvfcczZx4sREPfJ2kd3j4rHpeKo9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qAjMnLnE3h41yxaXRTZx8mIrschGvj/H5swtT+OwYFGFXXDlly79S9tr55XszJPXTctDAgQgAAEIQAACEIBA4yJQXp7u8XvhlR/bpWdv3LgaSmsgAIEmReDNN9+0q666yj744AObOnWqPfHEE7b33ntX2Yc33njDhg4dap999pmtuuqqdt5559nhhx9e5T51vREBuK4JUz4EIAABCDQogf2OfMemzlxWqzY8/fIMK1tudv5QROBaAWQnCEAAAhCAAAQg0IAE9HYXBgEIQCAfAosWLbKNN97YjjzySNtnn32qLWr8+PG2xx572PHHH28PPvigvfrqq3b00Udbr169bJdddql2/7rKgABcV2QpFwIQgAAEGpzA51/Nr7X4Gxo/fOQsW7PfZNt3z1WsNSEhAhY+IQABCEAAAhCAQKMi7Zyw4AAAQABJREFU0KpVeuiuVi3S0xpVo2kMBCCQM4GlS5fasmW1c+xJrSSKIispSQ4P07ZtW9Nfqu22226mv1zttttus/79+9s111zjd1l33XVtxIgRdt111zWoAMy3Ya5HkHwQgAAEINDkCKS/CFjzLiwtq7R7H5to9/97Ys13Zg8IQAACEIAABCAAgXojMOKZIRbXgb+dsMQeffK7equfiiAAgbohIPG3T5cVrEuXLgX569OnT1o5l19+eUEa/84779hOO+2UVJY8f5XekIYHcEPSp24IQAACEKhTAuuv3cm6dW3tYv26OA552KLFFfbcK9Ns39/2sfGTF9kqK7dzk8WlPx3Oowp2hQAEIAABCEAAAhAoAIHUUMA33vW97b/3GgUomSIgAIGGIiDP39nLltqTg35nHVq1zqsZi8qX295vP2GTJ0+2zp07J8rK5P2b2FiDhWnTplnPnj2T9tD6/PnzbcmSJdauXbukbfW1ggBcX6SpBwIQgAAE6p2AXut56r5tbNc/jrBFi/PzB27TpoVdd/u39umX82zajDLr2rmlPXL71tahPT+l9X5gqRACEIAABCAAAQhkIdDCvdVdGf2ysU1+WtEvBbEEAQg0OIGOpW2cANwmr3aUlP8U+kHib1wAzqvQJrAzISCawEGiiRCAAAQgUHsCEoGv+1vVsz9LzK3K5EW88opt7fURM2zq9DJzIaPsx3kVtsv+b7vl2B1GVYWwDQIQgAAEIAABCECgzgm8+fSQpDoeuWPLpHVWIACBpkugxD3hadEyvz+VUZe28sor2/Tp05Oq0LrE5oby/lVjEICTDgkrEIAABCDQHAmsN6CzPfvg1pZtHpC58yusVavMAwHNDVBREdnoj+ZZeUU6nZGjZ6cnkgIBCEAAAhCAAAQg0GAEFAt4o3V/Gtvtc/goG7zn8AZrCxVDAALFRWCbbbaxV199NanTr7zyiim9IQ0BuCHpUzcEIAABCNQbga6d21qvnqWWWeY1Ky93M8FmaE3Xzq1t/oLyDFt+Snpl+LSs29gAAQhAAAIQgAAEINAwBD7+IvktrRPOaNgJmBqGArVCoJkRkNNOIf5qgGXhwoU2duxY/6fdxo8f75cnTZrkSzn77LPt0EMPTZR4/PHH27hx4+zMM8+0L7/80m655RZ77LHH7LTTTkvkaYgFBOCGoE6dEIAABCDQIATWWbOTlWT55ZMH8Br929sqvUpdfN9WNmiLFezEI1e3ow7qZxmV4Z978L83Z9v0mUsbpD9UCgEIQAACEIAABCCQG4HvJi7LLSO5IACBRkughUJAFOCvJh18//33bdNNN/V/2m/o0KF++YILLvDFTJ061YIYrIT+/fvbc889Z/L63Xjjje2aa66xO++803bZZRefv6H+MXNNQ5GnXghAAAIQqHcCJx+9hovjOzNjvf1XbW83Xr6JtXIxpUpLf4oJXO5CP2j97Xdn2jsfzs24nxInfr/Yeq5YmnU7GyAAAQhAAAIQKC4CF/zjE3ttxBxbvW+p3XfTVsXV+UbSW4X+qozNAfyfuwc2kpbRDAhAoCkR2H777auc9+Wee+5J6472GTNmTFp6QyYgADckfeqGAAQgAIF6JdCjW1tLfhnwl+onTlniHH0rnfj706yyT734g7361kzr0a2N8wRewz745ENbttyFiXBvHaXO+zbmozm25abdfimMJQhAAAIQgAAEipbAVTd94cVfARg3cantdcgIe/r+wUXLo6E6/uZTQ2zylEVu4t6ltv6AbtbSPdTHIACBpk2gxL21qb98rCTKb/986m7IfbO8CNuQTaJuCEAAAhCAQN0R6LZC64yFL1tWaX88brR9+c18W7y43B5/ZoqNn7jI3h/7o937+CQbMrC7u3FIF39V2KPP/JCxTBIhAAEIQAACECg+Ak+/PCOp03PmZphFNikHK3VFYNVVOthG62kMV5yCT11xpVwINBSBEnctF+KvodrfkPXiAdyQ9KkbAhCAAATqncAj/9rSfrPf2xnr/XFeuR09dIyt2a+dTZi0JOEt/L/hLmyEcx3O5j3cuhXPUzMCJRECEIAABCBQhAR23K67/W/47ETPO3VEfEzAYAECEIAABBqEAHesDYKdSiEAAQhAoKEItCttZXvvtnKV1X874RfxVxkV8iGb+Kvtpxy7hj4wCEAAAhCAAAQgYBedsYFtMKCDJ7FC15b2wsPbQQUCEIAABApAoMS5sRbirwBNaXJF4AHc5A4ZDYYABCAAgXwJnPGnAfbKGzNt0ZLCvJK5x4698m0S+0MAAhCAAAQg0IwI3Hb15s2oN3QFAhCAQOMgUNLipxAQ+bSmpKI438rAAzifs4Z9IQABCECgyRI4aN9Vsra9R7fMcYKz7sAGCEAAAhCAAAQgAAEIQAACEIBAIyWAB3AjPTA0CwIQgAAE6pbAGv07W4l7+KvwDrLOnVra8Yetbj26tbFzL/v8p8TY/xbukWlr96tZtiyW6BZ7r1yanMAaBCAAAQhAAAIQgAAEIAABCBScgPcAdl7A+ZjKKEbDA7gYjzp9hgAEIAAB22jdLl4ADigqXDSIvXbpbR98PNcqnSqcaViQKv5q3wdv3SIUwScEIAABCEAAAhCAAAQgAAEI1BGBFk68LcRfHTWvUReLANyoDw+NgwAEIACBuiLQqWMrW6HLL6EeOnf+6aWYefOWW6UTg1MnfausTG/J2qt3cF7B/JSmkyEFAhCAAAQgAAEIQAACEIAABBoLAUJANJYjQTsgAAEIQKDeCSwt+0XVjdzi8Hdm2Yj3ZqeJv5ka1rKl2V3Xb5ZpE2kQgAAEIAABCEAAAhCAAAQgUGgC8r3J1/8m3/0L3ad6Kg8BuJ5AUw0EIAABCDQuAsuXV9qSpc7V92f7ce4y+8+zU2yZS8/F/nriWi6ERKZAEbnsTR4IQAACEIAABCAAAQhAAAIQqAmBEjcxi/7ysXz3z6fuhtw3P2oN2XLqhgAEIACBeiHw0OMT7LCT3rNbhn1rCxeV10ud9VHJ5KmLLR7WIYwj2pVW/9O4et92tvtvetdHM6kDAhCAAAQgAAEIQAACEIAABCCQFwE8gPPCx84QgAAEmieBSqeMHnD8ezZl6rJEB7+bOMXeeGemPfKvrX3g/cSGJrpwzGljklq+3TY9bLVV2tuYT+YmpaeudOzQ0u67acvUZNYhAAEIQAACEIAABCAAAQhAoA4JhAng8qlCZRSjIQAX41GnzxCAAASqIPDn88fa+2PnZcwxfcYyW7i43Dp3/GXytIwZG3ni1OlLrWxZcqiH1VfrYA898b1FqbO/xfqiuL//HbZVLIVFCEAAAhCAAAQgAAEIQAACEKgXAnpZs/oXNqtuSr77V116o92KANxoDw0NgwAEI... (706 KB left)

# Review codex1.md

> please catch up on [codex1.md](codex1.md)

<details><summary>2 previous messages</summary>

> I’m going to read `codex1.md` first and extract the current state so we can pick up cleanly from where things left off.
>
> I’ve got the full thread context now; it’s a long running implementation log, so I’m pulling the last concrete state and the current repo artifacts rather than paraphrasing the whole thing.

</details>

Current state from [codex1.md](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/codex1.md):

- The hetero graph refactor is already in place in [hetero_graph_builder.py](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/hetero_graph_builder.py).
- The next implemented step was the notebook smoke test in [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb).
- That notebook section now:
  - stratifies an ACME sample with red-team coverage
  - adds one-hop parent rows
  - builds `HeteroData`
  - prints `builder.report_`
  - trains a tiny supervised GraphSAGE process classifier
  - projects raw features vs trained embeddings to 2D
  - colors plots by `red_team` and `bad_user`
  - warns if a sampled session component looks suspiciously huge
- The reduced external smoke test passed:
  - built a hetero graph from 1,305 sampled ACME processes
  - got nonzero `spawns`, `ran_on`, `ran_as`, and `touches` edges
  - trained GraphSAGE for 3 quick epochs
  - loss dropped `0.753 -> 0.553 -> 0.495`
  - UMAP returned valid `(1305, 2)` coordinates

So the project is no longer at “build the hetero graph”; it’s at “run and inspect the visualization smoke test on the notebook cells.” The next practical action is to restart the kernel and run notebook cells 12 through 18 in [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb).

Model changed

gpt-5.4-mini -> gpt-5.5

> would you like to take a look now

<details><summary>12 previous messages</summary>

> I’ll take a look at the notebook state now, especially the new PyG visualization cells and any outputs/plots they produced. I’m going to inspect the executed cells first, then extract the plot image if it’s embedded so I can reason from the actual figure rather than guessing.
>
> <details><summary>Ran a command</summary>
>
> - Ran `@'
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb.get('cells', [])):
    src = ''.join(cell.get('source', []))
    outs = cell.get('outputs', [])
    exec_count = cell.get('execution_count')
    title = src.strip().splitlines()[0] if src.strip() else ''
    if 'PyG Graph Embedding Visualization' in src or 'HeterogeneousGeometricGraphBuilder' in src or 'UMAP' in src or outs:
        print(f'CELL {i} exec={exec_count} type={cell.get("cell_type")} title={title[:120]} outputs={len(outs)}')
        for oi, out in enumerate(outs):
            ot = out.get('output_type')
            keys = list(out.keys())
            print(f'  output {oi}: {ot} keys={keys}')
            if ot == 'stream':
                text = ''.join(out.get('text', []))
                print('    stream:', text[:1000].replace('\n',' | '))
            elif ot in ('execute_result','display_data'):
                data = out.get('data', {})
                print('    data keys:', list(data.keys()))
                if 'text/plain' in data:
                    text = ''.join(data['text/plain']) if isinstance(data['text/plain'], list) else data['text/plain']
                    print('    text:', text[:1000].replace('\n',' | '))
            elif ot == 'error':
                print('    error:', out.get('ename'), out.get('evalue'))
'@ | python -`
>
> </details>
>
> The first notebook parser hit a local Python launcher/session issue, so I’m switching to PowerShell’s JSON tooling. Same goal: identify the executed cells and pull out the meaningful outputs.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 0; $i -lt $nb.cells.Count; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $outs = @($cell.outputs)
  if ($src -match 'PyG Graph Embedding Visualization|HeterogeneousGeometricGraphBuilder|UMAP' -or $outs.Count -gt 0) {
    $first = (($src -split "`n") | Select-Object -First 1)
    Write-Output ("CELL {0} exec={1} type={2} title={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(120, $first.Length)), $outs.Count)
    for ($j = 0; $j -lt $outs.Count; $j++) {
      $out = $outs[$j]
      Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
      if ($out.output_type -eq 'stream') {
        $text = ($out.text -join '')
        if ($text.Length -gt 1000) { $text = $text.Substring(0,1000) }
        Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
      } elseif ($out.output_type -eq 'error') {
        Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      } elseif ($out.data) {
        $keys = ($out.data.PSObject.Properties.Name -join ', ')
        Write-Output ('    data keys: ' + $keys)
        if ($out.data.'text/plain') {
          $text = ($out.data.'text/plain' -join '')
          if ($text.Length -gt 1000) { $text = $text.Substring(0,1000) }
          Write-Output ('    text: ' + ($text -replace "`n", ' | '))
        }
      }
    }
  }
}``
>
> </details>
>
> The notebook did run, and it went further than the original smoke test: there are supervised, unsupervised, semantic, kNN, and holdout checks now. I’m going to pull out the key plot outputs next so I can give you a read on what the figures are saying.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
$targets = @(
  @{cell=17; out=1; name='cell17_supervised_red_team.png'},
  @{cell=17; out=2; name='cell17_supervised_bad_user.png'},
  @{cell=22; out=1; name='cell22_unsup_red_team.png'},
  @{cell=27; out=1; name='cell27_feature_recon_red_team.png'},
  @{cell=32; out=1; name='cell32_semantic_raw_red_team.png'},
  @{cell=36; out=1; name='cell36_semantic_graphsage_red_team.png'},
  @{cell=40; out=2; name='cell40_knn_bar.png'},
  @{cell=40; out=5; name='cell40_holdout_umap_1.png'}
)
New-Item -ItemType Directory -Force -Path 'tmp_notebook_plots' | Out-Null
foreach ($t in $targets) {
  $out = $nb.cells[$t.cell].outputs[$t.out]
  if ($out.data.'image/png') {
    $b64 = ($out.data.'image/png' -join '')
    [IO.File]::WriteAllBytes((Join-Path 'tmp_notebook_plots' $t.name), [Convert]::FromBase64String($b64))
    Write-Output (Join-Path 'tmp_notebook_plots' $t.name)
  } else {
    Write-Output ('missing image for cell ' + $t.cell + ' out ' + $t.out)
  }
}`
>
> </details>
>
> The key figures are extracted into `tmp_notebook_plots`. I’m opening the supervised and semantic plots first, since those are the highest-signal ones for deciding whether this path is actually promising.
>
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell17_supervised_red_team.png)
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell17_supervised_bad_user.png)
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell36_semantic_graphsage_red_team.png)
>
> Visually, the supervised GraphSAGE plot is doing exactly the smoke test should: the red-team points become much more concentrated after training. I’m checking the unsupervised variants too, because those tell us whether the graph structure is carrying signal before labels are injected.
>
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell22_unsup_red_team.png)
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell27_feature_recon_red_team.png)
> ![Image](/c:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/tmp_notebook_plots/cell32_semantic_raw_red_team.png)
>
> The plots give a pretty clear story: supervised GraphSAGE separates red-team nicely, but the unsupervised edge-only embedding is weak and the feature-reconstruction version is better but still not as clean. I’m grabbing the numeric kNN summaries now so we can compare the visual impression with a simple neighborhood metric.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
foreach ($idx in @(38,40)) {
  Write-Output ('CELL ' + $idx)
  foreach ($out in @($nb.cells[$idx].outputs)) {
    if ($out.data.'text/plain') {
      $text = ($out.data.'text/plain' -join '')
      Write-Output $text
    }
  }
}`
>
> </details>

</details>

I took a look. Short version: yes, the smoke test worked, and the plots are encouraging, but we should interpret them carefully.

The good news: the supervised GraphSAGE embeddings clearly separate `red_team` much better than raw process features. In the plot, red points move from scattered pockets into a few clean clusters after training. The `bad_user` view is also concentrated, though there are only 69 positives, so that one is more fragile.

The caveat: the strongest supervised plots are trained with the label, so they prove the pipeline can learn a signal, not that the graph alone discovered one. The unsupervised behavioral-edge-only embedding is basically not useful yet: its kNN check shows red-team positives and negatives both near red-team neighborhoods, meaning it is not separating cleanly. The unsupervised feature-reconstruction version is better, but still much weaker than supervised or semantic features.

The most interesting result is actually the semantic side:

- `semantic_text_svd` has very strong kNN enrichment.
- `supervised_graphsage_semantic_x` is also very strong.
- Holdout-style kNN numbers still look good, which is a much better sign than just pretty UMAP clusters.
- But one inspected high-purity neighborhood was dominated by `msedge.exe` renderer tokens, so we should sanity-check whether we’re learning attack behavior or dataset/user/host artifacts.

A few concrete notes:

- Graph built: `8,329` process nodes.
- Red-team rows: `1,721`.
- Bad-user rows: `69`.
- Largest session: `1,250`, which is large enough that I would inspect it before trusting session-level results.
- Supervised GraphSAGE training looks healthy: loss drops to `0.1796`, val accuracy reaches about `0.958`.
- Semantic GraphSAGE gets very high val accuracy, about `0.993`, which is promising but also raises leakage/confounding questions.

There are two harmless notebook issues below the main section:

- Cell 52 errors because `sessions` is undefined.
- Cell 66 errors because `node2vec` is not installed.

Those are in older/extra exploratory cells and don’t invalidate the PyG smoke test.

My read: this is a successful Step 5. The next serious step should be a leakage-resistant validation split, probably by `hostname`, `user`, or time, before we celebrate the semantic/GraphSAGE separation too hard. The plots are not noise; they’re telling us there is signal. Now we need to prove it survives a harder split.

> ok but if in the end I need an unsupervised solution, what should I focus on right now

Focus right now on making the **unsupervised embedding objective meaningful**, not on prettier UMAPs.

The current unsupervised edge-only GraphSAGE is weak because it is mostly learning “these nodes are connected by sparse process/file edges,” and that structure alone is not rich enough yet. For an unsupervised final system, I’d prioritize this order:

1. **Build better self-supervised targets**

   Use multiple reconstruction tasks, not just edge prediction:
   - reconstruct process features
   - reconstruct process name / command family
   - predict masked user/host/process-name tokens
   - predict parent-child links
   - predict rare file/network links
   - contrast nearby-in-session process nodes against random negatives

2. **Use semantic/text features as the anchor**

   Your semantic text/SVD embeddings are currently much stronger than graph-only embeddings. That is not a failure of the graph work. It means the graph should probably refine semantic behavior embeddings rather than replace them.

   Best next model shape:

   ```text
   semantic process features
   + hetero graph structure
   + self-supervised reconstruction / contrastive loss
   -> process embeddings
   -> session embeddings
   -> anomaly score
   ```

3. **Move from process-level plots to session-level anomaly scoring**

   Since your end goal is unsupervised detection, don’t optimize for red/blue separation in process UMAPs too long. Start scoring sessions:
   - mean/max pool process embeddings by session
   - compute distance from benign-looking session clusters
   - use Isolation Forest / LOF / one-class SVM / kNN distance
   - rank sessions by anomaly score
   - then inspect whether red-team sessions rise to the top

4. **Use labels only for evaluation**

   Keep `red_team` and `bad_user` completely out of training. Use them after training to answer:
   - Are red-team sessions enriched in top 1%, 5%, 10% anomaly scores?
   - Are bad-user sessions enriched?
   - Do top anomalies make behavioral sense?
   - Are we just finding one noisy host/user/process family?

5. **Fix the session graph before trusting session anomalies**

   The largest sampled session is `1,250`, which is a warning sign. For unsupervised detection, giant components can wash out the signal.

   Right now I’d inspect and tighten session construction:
   - avoid same-user edges for sessionization
   - cap high-degree artifacts/remotes
   - prefer parent-child, temporal proximity, rare file, rare remote
   - split huge components by time gaps or host boundaries

The next concrete notebook section I’d add is:

```text
Unsupervised Session Anomaly Baseline V1
```

It should train the feature-reconstruction GraphSAGE without labels, pool embeddings by session, score sessions with kNN distance or Isolation Forest, and report red-team enrichment in the top-ranked sessions.

That gives you the thing you actually need: not “does UMAP look separated,” but “does an unsupervised score push suspicious sessions upward?”

> ok

<details><summary>16 previous messages</summary>

> That sounds like a go-ahead to me, so I’ll turn the recommendation into the next notebook step: an unsupervised session anomaly baseline that uses labels only for evaluation. I’ll first inspect the current notebook variables/functions so the new section reuses what already ran instead of duplicating machinery.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 18; $i -le 45; $i++) {
  if ($i -lt $nb.cells.Count) {
    $cell = $nb.cells[$i]
    $src = ($cell.source -join '')
    $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
    if ($null -eq $first) { $first = '' }
    Write-Output ("CELL {0} type={1} exec={2} first={3}" -f $i, $cell.cell_type, $cell.execution_count, $first.Substring(0, [Math]::Min(160, $first.Length)))
  }
}``
> - Ran `rg -n "def mean_pool_by_session|feature_process_embeddings|unsupervised_feature|IsolationForest|LocalOutlierFactor|NearestNeighbors|semantic_svd" my_sessionization.ipynb`
> - Ran `git status --short`
>
> </details>
>
> The notebook already has the ingredients we need: `feature_process_embeddings`, session pooling, kNN evaluation, and holdout checks. I’m going to add a compact section after the feature-reconstruction embeddings that ranks sessions by anomaly score and reports red-team/bad-user enrichment in the top slices.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
foreach ($idx in @(18,25,28,38,40,43,44)) {
  Write-Output ('--- CELL ' + $idx + ' ---')
  $src = ($nb.cells[$idx].source -join '')
  if ($src.Length -gt 7000) { $src = $src.Substring(0,7000) + "`n...TRUNCATED..." }
  Write-Output $src
}``
> - Ran `Get-Command python, py, python3 -ErrorAction SilentlyContinue | Select-Object Name, Source`
>
> </details>
>
> I found the clean insertion point: right after the feature-reconstruction session UMAP. That lets the anomaly baseline consume `feature_session_embeddings` directly and keeps labels out of training, only in the score report.
>
> <details><summary>Ran a command</summary>
>
> - Ran `py -3 --version`
>
> </details>
>
> I’m going to make this a real ranking tool, not just another projection plot: it will score sessions with kNN distance and Isolation Forest, then show enrichment in the top 0.5%, 1%, 2%, 5%, and 10%. That gives us the unsupervised question directly: “do suspicious sessions rise to the top when labels were not used?”
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
foreach ($idx in @(14,15,16)) {
  Write-Output ('--- CELL ' + $idx + ' ---')
  $src = ($nb.cells[$idx].source -join '')
  if ($src.Length -gt 8000) { $src = $src.Substring(0,8000) + "`n...TRUNCATED..." }
  Write-Output $src
}``
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
$src = ($nb.cells[41].source -join '')
Write-Output $src`
>
> </details>
>
> I’m about to edit the notebook by inserting two cells: a short markdown marker and one code cell with reusable anomaly-ranking helpers plus the first report. The section is intentionally label-free until the evaluation table, so it matches the final unsupervised direction.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> markdown_source = [
> > "## Unsupervised Session Anomaly Ranking V1\n",
> > "\n",
> > "Score session embeddings without labels, then use `red_team` and `bad_user` only to evaluate whether the top-ranked sessions are enriched for suspicious activity.\n",
> > ]
>
> code_source = r'''from sklearn.ensemble import IsolationForest
> from sklearn.neighbors import NearestNeighbors
> from scipy import sparse
>
> ANOMALY_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
> ANOMALY_K = 15
> ANOMALY_RANDOM_STATE = RANDOM_STATE + 313
> INCLUDE_SESSION_SIZE_IN_ANOMALY = False
>
> def anomaly_numpy(matrix):
> if torch.is_tensor(matrix):
> matrix = matrix.detach().cpu().numpy()
> elif sparse.issparse(matrix):
> matrix = matrix.toarray()
> arr = np.asarray(matrix, dtype=np.float32)
> arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
> return arr
>
> def standardize_for_anomaly(matrix):
> X = anomaly_numpy(matrix)
> if X.ndim == 1:
> X = X.reshape(-1, 1)
> return StandardScaler().fit_transform(X)
>
> def mean_pool_process_matrix_by_session(matrix, session_id):
> X = anomaly_numpy(matrix)
> sid = anomaly_numpy(session_id).astype(int).reshape(-1)
> if X.shape[0] != len(sid):
> raise ValueError(f"process rows ({X.shape[0]}) != session ids ({len(sid)})")
>
>     n_sessions = int(sid.max()) + 1 if len(sid) else 0
>     pooled = np.zeros((n_sessions, X.shape[1]), dtype=np.float32)
>     counts = np.bincount(sid, minlength=n_sessions).astype(np.float32)
>     np.add.at(pooled, sid, X)
>     pooled = pooled / np.maximum(counts[:, None], 1.0)
>     return pooled
>
> def top_values(series, n=3):
> counts = series.dropna().astype(str).value_counts()
> return ", ".join(counts.head(n).index.tolist()) if len(counts) else ""
>
> def make_session_metadata(process_meta, session_id):
> sid = anomaly_numpy(session_id).astype(int).reshape(-1)
> meta = process_meta.copy().reset_index(drop=True)
> if len(meta) != len(sid):
> raise ValueError(f"metadata rows ({len(meta)}) != session ids ({len(sid)})")
>
>     meta["session_id"] = sid
>     if "bad_user_binary" not in meta.columns:
>         meta["bad_user_binary"] = meta["bad_user"].notna().astype(int) if "bad_user" in meta else 0
>
>     rows = []
>     for session_id_value, group in meta.groupby("session_id", sort=True):
>         row = {
>             "session_id": int(session_id_value),
>             "n_processes": int(len(group)),
>             "red_team": int(group["red_team"].fillna(0).astype(int).max()) if "red_team" in group else 0,
>             "bad_user": int(group["bad_user_binary"].fillna(0).astype(int).max()),
>             "primary_host": top_values(group["hostname"], n=1) if "hostname" in group else "",
>             "top_processes": top_values(group["process_name"], n=4) if "process_name" in group else "",
>             "top_users": top_values(group["user"], n=3) if "user" in group else "",
>         }
>         rows.append(row)
>     return pd.DataFrame(rows).sort_values("session_id").reset_index(drop=True)
>
> def knn*distance_anomaly_scores(session_matrix, k=ANOMALY_K):
> X = standardize_for_anomaly(session_matrix)
> if X.shape[0] <= 1:
> return np.zeros(X.shape[0], dtype=np.float32)
> n_neighbors = min(k + 1, X.shape[0])
> nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
> nn.fit(X)
> distances, * = nn.kneighbors(X)
> neighbor_distances = distances[:, 1:] if distances.shape[1] > 1 else distances
> return neighbor_distances.mean(axis=1)
>
> def isolation_forest_anomaly_scores(session_matrix):
> X = standardize_for_anomaly(session_matrix)
> if X.shape[0] <= 1:
> return np.zeros(X.shape[0], dtype=np.float32)
> model = IsolationForest(
> n_estimators=300,
> contamination="auto",
> random_state=ANOMALY_RANDOM_STATE,
> )
> model.fit(X)
> return -model.decision_function(X)
>
> def percentile_rank(scores):
> return pd.Series(scores).rank(method="average", pct=True).to_numpy()
>
> def build_session_anomaly_table(session_matrix, session_meta, embedding_name):
> X = anomaly_numpy(session_matrix)
> if len(X) != len(session_meta):
> raise ValueError(f"{embedding_name}: session rows ({len(X)}) != metadata rows ({len(session_meta)})")
>
>     score_matrix = X
>     if INCLUDE_SESSION_SIZE_IN_ANOMALY:
>         score_matrix = np.column_stack([score_matrix, np.log1p(session_meta["n_processes"].to_numpy())])
>
>     table = session_meta.copy()
>     table["embedding"] = embedding_name
>     table["knn_distance_score"] = knn_distance_anomaly_scores(score_matrix)
>     table["isolation_forest_score"] = isolation_forest_anomaly_scores(score_matrix)
>     table["rank_ensemble_score"] = (
>         percentile_rank(table["knn_distance_score"]) +
>         percentile_rank(table["isolation_forest_score"])
>     ) / 2.0
>     return table
>
> def anomaly_enrichment_report(score_table, score_col, label_col, top_fractions=ANOMALY_TOP_FRACTIONS):
> y = score_table[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
>
>     rows = []
>     for frac in top_fractions:
>         n_top = max(1, int(np.ceil(len(ranked) * frac)))
>         top = ranked.head(n_top)
>         hits = int(top[label_col].fillna(0).astype(int).sum())
>         precision = hits / n_top
>         recall = hits / total_positive if total_positive else np.nan
>         lift = precision / baseline if baseline > 0 else np.nan
>         rows.append({
>             "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
>             "score": score_col,
>             "label": label_col,
>             "top_fraction": frac,
>             "top_n": n_top,
>             "hits": hits,
>             "total_positive": total_positive,
>             "baseline_positive_rate": baseline,
>             "top_positive_rate": precision,
>             "recall_at_top": recall,
>             "lift_vs_baseline": lift,
>         })
>     return rows
>
> session_meta = make_session_metadata(process_meta, data["process"].session_id.cpu())
>
> session_embedding_sources = {
> "raw_numeric_session_mean": mean_pool_process_matrix_by_session(
> data["process"].x.cpu(),
> data["process"].session_id.cpu(),
> ),
> }
>
> optional_process_embedding_variables = {
> "unsupervised_behavioral_edges_session_mean": "unsup_process_embeddings",
> "feature_recon_graphsage_session_mean": "feature_process_embeddings",
> "semantic_raw_svd_session_mean": "semantic_process_svd",
> "semantic_text_svd_session_mean": "semantic_text_svd",
> }
>
> for embedding_name, variable_name in optional_process_embedding_variables.items():
> if variable_name in globals():
> session_embedding_sources[embedding_name] = mean_pool_process_matrix_by_session(
> globals()[variable_name],
> data["process"].session_id.cpu(),
> )
> else:
> print(f"Skipping {embedding_name}: run the cell that creates `{variable_name}` first.")
>
> session_score_tables = {
> embedding_name: build_session_anomaly_table(session_matrix, session_meta, embedding_name)
> for embedding_name, session_matrix in session_embedding_sources.items()
> }
>
> print("Session anomaly sources:")
> for embedding_name, session_matrix in session_embedding_sources.items():
> print(f" {embedding_name}: {session_matrix.shape}")
> print("Labels are used below only for enrichment checks, not for fitting anomaly scores.")
>
> enrichment_rows = []
> for table in session_score_tables.values():
> for score_col in ["knn_distance_score", "isolation_forest_score", "rank_ensemble_score"]:
> for label_col in ["red_team", "bad_user"]:
> enrichment_rows.extend(anomaly_enrichment_report(table, score_col, label_col))
>
> session_anomaly_enrichment = pd.DataFrame(enrichment_rows)
> summary_view = session_anomaly_enrichment[
> session_anomaly_enrichment["top_fraction"].isin([0.01, 0.05, 0.10])
> ].sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(summary_view)
>
> best_red_team = session_anomaly_enrichment[
> (session_anomaly_enrichment["label"] == "red_team") &
> (session_anomaly_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> if len(best_red_team):
> best_embedding = best_red_team.iloc[0]["embedding"]
> best_score = best_red_team.iloc[0]["score"]
> print(f"Best top-1% red_team ranking: {best_embedding} using {best_score}")
> top_sessions = session_score_tables[best_embedding].sort_values(best_score, ascending=False).head(20)
> display_cols = [
> > "session_id",
> > best_score,
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display(top_sessions[display_cols])
> '''.splitlines(keepends=True)
>
> markdown_cell = {
> "cell_type": "markdown",
> "metadata": {},
> "source": markdown_source,
> }
> code_cell = {
> "cell_type": "code",
> "execution_count": None,
> "metadata": {},
> "outputs": [],
> "source": code_source,
> }
>
> # Replace an existing copy of this section if the cell was already inserted.
>
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Session Anomaly Ranking V1"):
> del nb["cells"][idx:idx + 2]
> break
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Holdout Validation"):
> insert_at = idx
> break
> if insert_at is None:
> insert_at = len(nb["cells"])
>
> nb["cells"][insert_at:insert_at] = [markdown_cell, code_cell]
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted Unsupervised Session Anomaly Ranking V1 at cell index {insert_at}.")
> '@
> $script | py -3 -``
>
> </details>
>
> The notebook section is inserted. I’m validating the notebook JSON and compiling the new code cell now, then I’ll do a quick static pass for any obvious variable-order mistakes.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('from sklearn.ensemble import IsolationForest'):
        ast.parse(src)
        print(f'compiled anomaly cell at index {i}, lines={len(src.splitlines())}')
        break
else:
    raise SystemExit('anomaly cell not found')
'@
$script | py -3 -`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
Write-Output ('cells=' + $nb.cells.Count)
for ($i = 38; $i -le 46; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} first={2}" -f $i, $cell.cell_type, $first.Substring(0, [Math]::Min(140, $first.Length)))
}``
> - Ran `rg -n "Unsupervised Session Anomaly Ranking V1|ANOMALY_TOP_FRACTIONS|Best top-1% red_team" my_sessionization.ipynb`
>
> </details>
>
> The new cell parses cleanly. I spotted one easy robustness improvement before I leave it with you: I’m adding an explicit `StandardScaler` import inside the section so it can run even if you jump to it after a kernel restart and rerun the required upstream cells in chunks.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$script = @'
import json
from pathlib import Path
path = Path('my_sessionization.ipynb')
nb = json.loads(path.read_text(encoding='utf-8'))
for cell in nb['cells']:
    src = ''.join(cell.get('source', []))
    if src.startswith('from sklearn.ensemble import IsolationForest'):
        lines = cell['source']
        import_line = 'from sklearn.preprocessing import StandardScaler\n'
        if import_line not in lines:
            for idx, line in enumerate(lines):
                if line == 'from sklearn.neighbors import NearestNeighbors\n':
                    lines.insert(idx + 1, import_line)
                    break
        break
else:
    raise SystemExit('anomaly cell not found')
path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
print('Added explicit StandardScaler import to anomaly cell.')
'@
$script | py -3 -`
>
> </details>
>
> I’ve made the import tweak. One final validation pass now, then I’ll summarize exactly what to run and what numbers to look at.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('from sklearn.ensemble import IsolationForest'):
        ast.parse(src)
        print(f'compiled anomaly cell at index {i}, lines={len(src.splitlines())}')
        print(src.splitlines()[:4])
        break
else:
    raise SystemExit('anomaly cell not found')
'@
$script | py -3 -`
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
$cell = $nb.cells[43]
Write-Output (($cell.source | Select-Object -First 8) -join '')`
>
> </details>

</details>

I added the next unsupervised-focused section to [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:7907): **Unsupervised Session Anomaly Ranking V1**.

It does the thing we need now:

- pools process embeddings into session embeddings
- scores sessions without labels using `kNN` distance, `IsolationForest`, and a rank ensemble
- uses `red_team` and `bad_user` only afterward for enrichment evaluation
- reports top `0.5%`, `1%`, `2%`, `5%`, and `10%` hit rate, recall, and lift
- shows the top 20 sessions for the best top-1% red-team ranking, including host, users, process names, and labels for inspection

Run the notebook through the kNN contamination section, then run the new section right after it. The main columns to watch are:

- `top_positive_rate`: how many top-ranked sessions are actually red-team/bad-user
- `lift_vs_baseline`: how much better than random the unsupervised ranking is
- `recall_at_top`: how many positives we recover in the top slice

I validated the notebook JSON and compiled the inserted code cell. I didn’t execute it end-to-end from the terminal because it depends on the live notebook variables you already built in-kernel.

> look

<details><summary>4 previous messages</summary>

> I’ll look at the notebook outputs again, especially the new unsupervised anomaly section, and see whether it ran cleanly or surfaced something interesting.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 40; $i -lt [Math]::Min($nb.cells.Count, 50); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(140, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 1800) { $text = $text.Substring(0,1800) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 2200) { $text = $text.Substring(0,2200) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The new section ran. I’m pulling the anomaly table out of the notebook output more precisely now, because the rendered text is truncated and the important part is the top-rate/lift rows.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
$out = $nb.cells[43].outputs[1]
$out.data.PSObject.Properties.Name
$data = $out.data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
if ($data) {
  $data | ConvertTo-Json -Depth 6 | Select-Object -First 1
} else {
  Write-Output 'no datawrangler json'
}`
>
> </details>

</details>

Yep, this is a genuinely interesting result.

The unsupervised session anomaly section ran, and the strongest signal is:

```text
Best top-1% red_team ranking:
semantic_text_svd_session_mean using isolation_forest_score
```

Top 1% session ranking:

```text
68 sessions inspected
61 red-team sessions found
top_positive_rate = 89.7%
baseline red-team session rate = 4.85%
lift = 18.48x
recall = 18.65% of all red-team sessions
```

That is very good for an unsupervised score. It means: without using labels for scoring, the semantic text session embeddings plus Isolation Forest pushed red-team sessions way up into the anomaly tail.

But there’s a pattern we should not ignore: the top sessions are mostly tiny sessions, often `n_processes = 1` or `2`, and many are on `ACME-HH-ZYQ` with `cmd.exe` / `reg.exe`. So the model may be finding a real red-team pocket, but it may also be heavily keyed on one campaign/host/process family. That’s okay at this stage, but it tells us the next validation should be “does this generalize when that host/user/campaign isn’t allowed to dominate?”

For `bad_user`, top 1% found nothing, but top 5% and 10% are strong:

```text
top 5%: 40 / 47 bad-user sessions found, 17.0x lift
top 10%: 47 / 47 bad-user sessions found, 10.0x lift
```

So `bad_user` is not at the very tip of the anomaly ranking, but it is highly enriched once you widen the review queue.

My take: focus on **semantic-text session anomaly scoring** as the current unsupervised winner. The graph feature-reconstruction model is useful, but right now it is not beating semantic text. Next I’d add a stricter evaluation cell: top-k anomaly performance grouped by host, and a leave-one-host-out or train-on-benign-ish-hosts test. That will tell us whether we have a robust unsupervised detector or a very good detector for this particular ACME pocket.

> ok

<details><summary>6 previous messages</summary>

> I’m going to treat that as “yes, add the stricter check.” I’ll insert a follow-up notebook section that tests whether the anomaly ranking is robust or just dominated by one host/process pocket.
>
> The shape I’m adding is deliberately practical: first, a host-dominance report for the current top anomaly slices; second, a within-host re-ranking that asks, “if every host gets its own local baseline, do the same red-team sessions still rise?”
>
> I’m editing the notebook now, placing the robustness section immediately after the session anomaly ranking so it uses the tables from that cell. It won’t disturb the holdout validation section that follows.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> markdown_source = [
> > "## Host Robustness Check for Unsupervised Session Scores\n",
> > "\n",
> > "Check whether the top anomaly queue is robust across hosts or mostly explained by one host/process pocket. Labels are still used only for evaluation.\n",
> > ]
>
> code_source = r'''HOST_ROBUST_TOP_FRACTIONS = [0.01, 0.05, 0.10]
> HOST_ROBUST_SCORE_COLUMNS = [
> > "knn_distance_score",
> > "isolation_forest_score",
> > "rank_ensemble_score",
> > ]
>
> def first_value(value):
> if pd.isna(value):
> return ""
> return str(value).split(",")[0].strip()
>
> def host_session_base_rates(session_meta):
> rows = []
> for host, group in session_meta.groupby("primary_host", dropna=False):
> n_sessions = len(group)
> rows.append({
> "primary_host": host,
> "sessions": n_sessions,
> "red_team_sessions": int(group["red_team"].fillna(0).astype(int).sum()),
> "bad_user_sessions": int(group["bad_user"].fillna(0).astype(int).sum()),
> "red_team_rate": float(group["red_team"].fillna(0).astype(int).mean()) if n_sessions else np.nan,
> "bad_user_rate": float(group["bad_user"].fillna(0).astype(int).mean()) if n_sessions else np.nan,
> "top_process_families": top_values(group["top_processes"].map(first_value), n=5),
> })
> return pd.DataFrame(rows).sort_values(
> ["red_team_sessions", "red_team_rate", "sessions"],
> ascending=[False, False, False],
> ).reset_index(drop=True)
>
> def top_slice_host_dominance(score_table, score_col, label_col, top_fraction):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_value)
>
>     host_counts = top["primary_host"].fillna("").astype(str).value_counts()
>     process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
>     hits = int(top[label_col].fillna(0).astype(int).sum())
>     total_positive = int(ranked[label_col].fillna(0).astype(int).sum())
>     baseline = float(ranked[label_col].fillna(0).astype(int).mean())
>     top_positive_rate = hits / n_top
>
>     return {
>         "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
>         "score": score_col,
>         "label": label_col,
>         "top_fraction": top_fraction,
>         "top_n": n_top,
>         "hits": hits,
>         "total_positive": total_positive,
>         "top_positive_rate": top_positive_rate,
>         "lift_vs_baseline": top_positive_rate / baseline if baseline > 0 else np.nan,
>         "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
>         "top_host": host_counts.index[0] if len(host_counts) else "",
>         "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
>         "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
>         "top_process_family": process_counts.index[0] if len(process_counts) else "",
>         "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
>         "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
>     }
>
> def ranked_enrichment_report(score_table, ranking_cols, label_col, score_name, top_fractions=HOST_ROBUST_TOP_FRACTIONS):
> ranked = score_table.sort_values(ranking_cols, ascending=[False] \* len(ranking_cols)).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
>
>     rows = []
>     for frac in top_fractions:
>         n_top = max(1, int(np.ceil(len(ranked) * frac)))
>         top = ranked.head(n_top)
>         hits = int(top[label_col].fillna(0).astype(int).sum())
>         precision = hits / n_top
>         rows.append({
>             "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
>             "score": score_name,
>             "label": label_col,
>             "top_fraction": frac,
>             "top_n": n_top,
>             "hits": hits,
>             "total_positive": total_positive,
>             "baseline_positive_rate": baseline,
>             "top_positive_rate": precision,
>             "recall_at_top": hits / total_positive if total_positive else np.nan,
>             "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
>         })
>     return rows
>
> def add*within_host_scores(score_table):
> table = score_table.copy()
> for score_col in HOST_ROBUST_SCORE_COLUMNS:
> within_col = f"within_host*{score_col}"
> table[within_col] = table.groupby("primary_host", dropna=False)[score_col].rank(
> method="average",
> pct=True,
> )
> return table
>
> host_base_rates = host_session_base_rates(session_meta)
> print("Session label base rates by host. High rates here can explain strong global anomaly queues.")
> display(host_base_rates)
>
> host_dominance_rows = []
> for table in session_score_tables.values():
> for score_col in HOST_ROBUST_SCORE_COLUMNS:
> for label_col in ["red_team", "bad_user"]:
> for frac in HOST_ROBUST_TOP_FRACTIONS:
> host_dominance_rows.append(top_slice_host_dominance(table, score_col, label_col, frac))
>
> host_dominance_report = pd.DataFrame(host_dominance_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> print("Top anomaly slice host/process concentration. Watch top_host_share and top_process_family_share.")
> display(host_dominance_report)
>
> within_host_score_tables = {
> embedding_name: add_within_host_scores(table)
> for embedding_name, table in session_score_tables.items()
> }
>
> within*host_rows = []
> for table in within_host_score_tables.values():
> for score_col in HOST_ROBUST_SCORE_COLUMNS:
> within_col = f"within_host*{score_col}"
> for label_col in ["red_team", "bad_user"]:
> within_host_rows.extend(
> ranked_enrichment_report(
> table,
> ranking_cols=[within_col, score_col],
> label_col=label_col,
> score_name=within_col,
> )
> )
>
> within_host_anomaly_enrichment = pd.DataFrame(within_host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> print("Within-host anomaly ranking. This asks whether sessions are unusual relative to their own host.")
> display(within_host_anomaly_enrichment)
>
> best_global_red = session_anomaly_enrichment[
> (session_anomaly_enrichment["label"] == "red_team") &
> (session_anomaly_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> best_within_red = within_host_anomaly_enrichment[
> (within_host_anomaly_enrichment["label"] == "red_team") &
> (within_host_anomaly_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> if len(best_global_red) and len(best_within_red):
> comparison = pd.concat(
> [
> > best_global_red.assign(ranking_scope="global_score"),
> > best_within_red.assign(ranking_scope="within_host_score"),
> > ],
> ignore_index=True,
> )
> display_cols = [
> > "ranking_scope",
> > "embedding",
> > "score",
> > "label",
> > "top_fraction",
> > "hits",
> > "top_n",
> > "total_positive",
> > "top_positive_rate",
> > "recall_at_top",
> > "lift_vs_baseline",
> > ]
> print("Best global vs best within-host top-1% red_team ranking:")
> display(comparison[display_cols])
> '''.splitlines(keepends=True)
>
> markdown_cell = {"cell_type": "markdown", "metadata": {}, "source": markdown_source}
> code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_source}
>
> # Replace existing copy if present.
>
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Host Robustness Check for Unsupervised Session Scores"):
> del nb["cells"][idx:idx + 2]
> break
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Holdout Validation"):
> insert_at = idx
> break
> if insert_at is None:
> insert_at = len(nb["cells"])
>
> nb["cells"][insert_at:insert_at] = [markdown_cell, code_cell]
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted Host Robustness Check at cell index {insert_at}.")
> '@
> $script | py -3 -`
>
> </details>
>
> The robustness section is in. I’m validating the notebook JSON and compiling that new code cell so we catch syntax issues before you run it.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('HOST_ROBUST_TOP_FRACTIONS'):
        ast.parse(src)
        print(f'compiled host robustness cell at index {i}, lines={len(src.splitlines())}')
        break
else:
    raise SystemExit('host robustness cell not found')
'@
$script | py -3 -`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 40; $i -le 48; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} first={3}" -f $i, $cell.cell_type, $cell.execution_count, $first.Substring(0, [Math]::Min(140, $first.Length)))
}``
> - Ran `rg -n "Host Robustness Check|HOST_ROBUST_TOP_FRACTIONS|within_host_anomaly_enrichment|host_dominance_report" my_sessionization.ipynb`
>
> </details>

</details>

I added the next check to [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:9761): **Host Robustness Check for Unsupervised Session Scores**.

Run the new cell right after **Unsupervised Session Anomaly Ranking V1**. It gives you three things:

- `host_base_rates`: whether certain hosts already contain most of the red-team/bad-user sessions.
- `host_dominance_report`: whether the top anomaly queue is mostly one host or one process family.
- `within_host_anomaly_enrichment`: re-ranks anomalies relative to each host, which is the key sanity check.

What to look for:

- If global top-1% is strong but within-host top-1% collapses, the score is mostly finding a host/campaign pocket.
- If within-host top-1% stays strong, that’s a much better sign for a deployable unsupervised detector.
- If `top_host_share` is very high, inspect that host before trusting the global ranking.

I validated the inserted code cell parses cleanly.

> take a look

<details><summary>6 previous messages</summary>

> I’ll inspect the latest notebook outputs around the new host robustness section and read the tables back into a concise interpretation.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 42; $i -lt [Math]::Min($nb.cells.Count, 49); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(140, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 1800) { $text = $text.Substring(0,1800) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 2400) { $text = $text.Substring(0,2400) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The robustness cell ran, and the headline is clear already: the global score is very strong but heavily host-concentrated. I’m pulling the exact rows for the best global and within-host comparisons so we can separate “promising detector” from “host pocket detector” cleanly.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$hostBase = RowsToObjects $nb.cells[45].outputs[1].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$hostDom = RowsToObjects $nb.cells[45].outputs[3].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$within = RowsToObjects $nb.cells[45].outputs[5].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$compare = RowsToObjects $nb.cells[45].outputs[7].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
Write-Output 'HOST BASE TOP 4'
$hostBase | Select-Object -First 4 | Format-Table -AutoSize | Out-String -Width 240
Write-Output 'GLOBAL RED TOP 1 ROWS'
$hostDom | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Select-Object -First 8 | Format-Table -AutoSize | Out-String -Width 300
Write-Output 'WITHIN RED TOP 1 ROWS'
$within | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Select-Object -First 8 | Format-Table -AutoSize | Out-String -Width 260
Write-Output 'COMPARE'
$compare | Format-Table -AutoSize | Out-String -Width 260`
>
> </details>
>
> The exact comparison confirms the story: global semantic text ranking is excellent, but once we normalize within host, the graph/behavioral embedding becomes the best top-1% method. I’m checking the concentration columns for the global winner now, since that tells us how much is coming from `ACME-HH-ZYQ`.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$hostDom = RowsToObjects $nb.cells[45].outputs[3].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$hostDom | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Select-Object -First 5 | ConvertTo-Json -Depth 3`
>
> </details>

</details>

This is a really useful result. The robustness check did exactly what we needed: it separated “strong global anomaly signal” from “host-dominated signal.”

The global winner is still excellent:

```text
semantic_text_svd_session_mean + Isolation Forest
top 1%: 61 / 68 red-team sessions
precision: 89.7%
lift: 18.48x
```

But it is very concentrated:

```text
top host: ACME-HH-ZYQ
top host rows: 63 / 68
top host share: 92.6%
distinct hosts: 5
top process family: cmd.exe
```

So the global semantic-text anomaly score is mostly finding the `ACME-HH-ZYQ` campaign pocket. That is not bad. It means the unsupervised method is detecting a real abnormal pocket. But it also means we should not claim it is broadly host-robust yet.

The more exciting robustness finding is this:

```text
best within-host top 1%:
unsupervised_behavioral_edges_session_mean + within_host_knn_distance_score

hits: 19 / 68
precision: 27.9%
baseline: 4.85%
lift: 5.76x
```

That’s weaker than the global semantic score, but much more meaningful for deployability. Once each host gets its own local baseline, the behavioral graph embedding becomes the best red-team ranking method. That suggests the graph is adding a different kind of signal: “unusual relative to this host,” not just “this host/campaign looks different globally.”

Host base rates explain the earlier result:

```text
ACME-HH-ZYQ: 270 / 683 red-team sessions = 39.5%
ACME-HH-DXJ: 47 / 796 red-team sessions = 5.9%, also all 47 bad_user sessions
ACME-HH-EUO: 5 red-team sessions
ACME-HH-ESO: 5 red-team sessions
others: 0 red-team sessions
```

My read:

- For global triage, `semantic_text_svd_session_mean + IsolationForest` is the current winner.
- For a more realistic unsupervised detector, focus on `within-host` ranking.
- The graph/behavioral embedding is now more important than it looked from the global UMAPs, because it wins under the within-host sanity check.

Next thing I’d do: create a final “Unsupervised Detector V1” score that ensembles both:

```text
global semantic anomaly score
+ within-host behavioral graph anomaly score
```

Then evaluate top 1%, 5%, 10% again. That would combine campaign-pocket detection with local host-relative weirdness.

> ok go ahead

<details><summary>5 previous messages</summary>

> I’ll add the ensemble section now, using the two strongest signals we just found: global semantic-text anomaly and within-host behavioral-graph anomaly. It’ll keep labels out of scoring and use them only to evaluate the ranked queue.
>
> The ensemble will compare a few fixed blends rather than tuning one using labels: global-only, within-host-only, 50/50, semantic-heavy, and behavioral-heavy. That way we can see whether the combined detector improves the queue without accidentally baking the answer into the score.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> markdown_source = [
> > "## Unsupervised Detector V1: Global + Within-Host Ensemble\n",
> > "\n",
> > "Combine the strongest global semantic-text anomaly signal with the strongest within-host behavioral graph signal. Labels are used only to evaluate the ranked review queue.\n",
> > ]
>
> code_source = r'''DETECTOR_GLOBAL_SOURCE = "semantic_text_svd_session_mean"
> DETECTOR_GLOBAL_SCORE = "isolation_forest_score"
> DETECTOR_LOCAL_SOURCE = "unsupervised_behavioral_edges_session_mean"
> DETECTOR_LOCAL_SCORE = "within_host_knn_distance_score"
> DETECTOR_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
>
> def rank01(values):
> return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average", pct=True).to_numpy()
>
> def build_detector_v1_table():
> required_globals = ["session_score_tables", "within_host_score_tables", "session_meta"]
> missing = [name for name in required_globals if name not in globals()]
> if missing:
> raise RuntimeError(f"Run the session anomaly and host robustness cells first. Missing: {missing}")
>
>     if DETECTOR_GLOBAL_SOURCE not in session_score_tables:
>         raise KeyError(f"Missing global source: {DETECTOR_GLOBAL_SOURCE}")
>     if DETECTOR_LOCAL_SOURCE not in within_host_score_tables:
>         raise KeyError(f"Missing within-host source: {DETECTOR_LOCAL_SOURCE}")
>
>     base_cols = [
>         "session_id",
>         "n_processes",
>         "red_team",
>         "bad_user",
>         "primary_host",
>         "top_users",
>         "top_processes",
>     ]
>     base = session_meta[base_cols].copy()
>
>     global_scores = session_score_tables[DETECTOR_GLOBAL_SOURCE][[
>         "session_id",
>         DETECTOR_GLOBAL_SCORE,
>     ]].rename(columns={DETECTOR_GLOBAL_SCORE: "global_semantic_score"})
>
>     local_scores = within_host_score_tables[DETECTOR_LOCAL_SOURCE][[
>         "session_id",
>         DETECTOR_LOCAL_SCORE,
>     ]].rename(columns={DETECTOR_LOCAL_SCORE: "within_host_behavioral_score"})
>
>     table = base.merge(global_scores, on="session_id", how="left").merge(local_scores, on="session_id", how="left")
>     table["global_semantic_score"] = table["global_semantic_score"].fillna(table["global_semantic_score"].min())
>     table["within_host_behavioral_score"] = table["within_host_behavioral_score"].fillna(0.0)
>
>     table["global_semantic_rank"] = rank01(table["global_semantic_score"])
>     # The within-host score is already a host-local percentile, but ranking it once more gives a stable global ordering.
>     table["within_host_behavioral_rank"] = rank01(table["within_host_behavioral_score"])
>
>     table["detector_v1_50_50"] = 0.50 * table["global_semantic_rank"] + 0.50 * table["within_host_behavioral_rank"]
>     table["detector_v1_semantic_heavy"] = 0.70 * table["global_semantic_rank"] + 0.30 * table["within_host_behavioral_rank"]
>     table["detector_v1_behavioral_heavy"] = 0.30 * table["global_semantic_rank"] + 0.70 * table["within_host_behavioral_rank"]
>     return table
>
> def detector_enrichment_report(score_table, score_col, label_col, top_fractions=DETECTOR_TOP_FRACTIONS):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
>
>     rows = []
>     for frac in top_fractions:
>         n_top = max(1, int(np.ceil(len(ranked) * frac)))
>         top = ranked.head(n_top)
>         hits = int(top[label_col].fillna(0).astype(int).sum())
>         precision = hits / n_top
>         rows.append({
>             "detector_score": score_col,
>             "label": label_col,
>             "top_fraction": frac,
>             "top_n": n_top,
>             "hits": hits,
>             "total_positive": total_positive,
>             "baseline_positive_rate": baseline,
>             "top_positive_rate": precision,
>             "recall_at_top": hits / total_positive if total_positive else np.nan,
>             "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
>         })
>     return rows
>
> def first_token(value):
> if pd.isna(value):
> return ""
> return str(value).split(",")[0].strip()
>
> def detector_host_dominance(score_table, score_col, label_col, top_fraction):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_token)
>
>     host_counts = top["primary_host"].fillna("").astype(str).value_counts()
>     process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
>     hits = int(top[label_col].fillna(0).astype(int).sum())
>     baseline = float(ranked[label_col].fillna(0).astype(int).mean())
>     top_positive_rate = hits / n_top
>
>     return {
>         "detector_score": score_col,
>         "label": label_col,
>         "top_fraction": top_fraction,
>         "top_n": n_top,
>         "hits": hits,
>         "top_positive_rate": top_positive_rate,
>         "lift_vs_baseline": top_positive_rate / baseline if baseline > 0 else np.nan,
>         "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
>         "top_host": host_counts.index[0] if len(host_counts) else "",
>         "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
>         "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
>         "top_process_family": process_counts.index[0] if len(process_counts) else "",
>         "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
>         "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
>     }
>
> detector_v1_table = build_detector_v1_table()
>
> DETECTOR_SCORE_COLUMNS = [
> > "global_semantic_rank",
> > "within_host_behavioral_rank",
> > "detector_v1_50_50",
> > "detector_v1_semantic_heavy",
> > "detector_v1_behavioral_heavy",
> > ]
>
> print("Detector V1 components:")
> print(f" global semantic: {DETECTOR_GLOBAL_SOURCE} / {DETECTOR_GLOBAL_SCORE}")
> print(f" within-host behavioral: {DETECTOR_LOCAL_SOURCE} / {DETECTOR_LOCAL_SCORE}")
> print("Labels are used below only for evaluation.")
>
> enrichment_rows = []
> for score_col in DETECTOR_SCORE_COLUMNS:
> for label_col in ["red_team", "bad_user"]:
> enrichment_rows.extend(detector_enrichment_report(detector_v1_table, score_col, label_col))
>
> detector_v1_enrichment = pd.DataFrame(enrichment_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(detector_v1_enrichment)
>
> host_rows = []
> for score_col in DETECTOR_SCORE_COLUMNS:
> for label_col in ["red_team", "bad_user"]:
> for frac in [0.01, 0.05, 0.10]:
> host_rows.append(detector_host_dominance(detector_v1_table, score_col, label_col, frac))
>
> detector_v1_host_dominance = pd.DataFrame(host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> print("Detector V1 host/process concentration:")
> display(detector_v1_host_dominance)
>
> best_red = detector_v1_enrichment[
> (detector_v1_enrichment["label"] == "red_team") &
> (detector_v1_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> if len(best_red):
> best_score = best_red.iloc[0]["detector_score"]
> print(f"Best Detector V1 top-1% red_team score: {best_score}")
> top_detector_v1_sessions = detector_v1_table.sort_values(best_score, ascending=False).head(25)
> display_cols = [
> > "session_id",
> > best_score,
> > "global_semantic_rank",
> > "within_host_behavioral_rank",
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display(top_detector_v1_sessions[display_cols])
> '''.splitlines(keepends=True)
>
> markdown_cell = {"cell_type": "markdown", "metadata": {}, "source": markdown_source}
> code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_source}
>
> # Replace existing copy if present.
>
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Detector V1"):
> del nb["cells"][idx:idx + 2]
> break
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Holdout Validation"):
> insert_at = idx
> break
> if insert_at is None:
> insert_at = len(nb["cells"])
>
> nb["cells"][insert_at:insert_at] = [markdown_cell, code_cell]
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted Unsupervised Detector V1 at cell index {insert_at}.")
> '@
> $script | py -3 -`
>
> </details>
>
> The Detector V1 section is inserted. I’m doing the usual parse check now, plus a quick cell-order check so it sits after robustness and before holdout validation.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('DETECTOR_GLOBAL_SOURCE'):
        ast.parse(src)
        print(f'compiled Detector V1 cell at index {i}, lines={len(src.splitlines())}')
        break
else:
    raise SystemExit('Detector V1 cell not found')
'@
$script | py -3 -`
> - Ran `rg -n "Unsupervised Detector V1|DETECTOR_GLOBAL_SOURCE|detector_v1_enrichment|detector_v1_host_dominance" my_sessionization.ipynb`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 42; $i -le 50; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} first={3}" -f $i, $cell.cell_type, $cell.execution_count, $first.Substring(0, [Math]::Min(150, $first.Length)))
}``
>
> </details>

</details>

Added **Unsupervised Detector V1: Global + Within-Host Ensemble** to [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:12922).

It combines:

```text
global semantic score:
semantic_text_svd_session_mean / isolation_forest_score

within-host behavioral score:
unsupervised_behavioral_edges_session_mean / within_host_knn_distance_score
```

The cell evaluates:

- `global_semantic_rank`
- `within_host_behavioral_rank`
- `detector_v1_50_50`
- `detector_v1_semantic_heavy`
- `detector_v1_behavioral_heavy`

It reports top `0.5%`, `1%`, `2%`, `5%`, and `10%` enrichment for `red_team` and `bad_user`, plus host/process concentration for each score. It also displays the top 25 sessions for the best top-1% red-team Detector V1 score.

Run it right after **Host Robustness Check for Unsupervised Session Scores**. I validated the inserted code parses cleanly.

> yeah take a look

<details><summary>6 previous messages</summary>

> I’ll read the latest Detector V1 outputs now and compare the ensemble against the earlier global-only and within-host-only scores. If anything errored, I’ll patch it; otherwise I’ll give you the read on whether the ensemble helped.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 44; $i -lt [Math]::Min($nb.cells.Count, 52); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(150, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 2000) { $text = $text.Substring(0,2000) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 2600) { $text = $text.Substring(0,2600) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The Detector V1 cell ran cleanly, and the first read is surprising in a useful way: the ensemble did not beat the global semantic score for red-team top-1%. I’m extracting the exact rows now so we can decide whether to keep the ensemble, adjust it, or split the detector into two queues.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$enrich = RowsToObjects $nb.cells[47].outputs[1].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$host = RowsToObjects $nb.cells[47].outputs[3].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
Write-Output 'RED TOP 1%'
$enrich | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Format-Table -AutoSize | Out-String -Width 260
Write-Output 'RED TOP 5%'
$enrich | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.05' } | Format-Table -AutoSize | Out-String -Width 260
Write-Output 'BAD_USER TOP 5% and 10%'
$enrich | Where-Object { $_.label -eq 'bad_user' -and ($_.top_fraction -eq '0.05' -or $_.top_fraction -eq '0.1') } | Format-Table -AutoSize | Out-String -Width 260
Write-Output 'HOST RED TOP 1%'
$host | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Format-Table -AutoSize | Out-String -Width 320
Write-Output 'HOST BAD_USER TOP 5%'
$host | Where-Object { $_.label -eq 'bad_user' -and $_.top_fraction -eq '0.05' } | Format-Table -AutoSize | Out-String -Width 320`
>
> </details>
>
> The numbers confirm it: the naive blend is actually worse for red-team because the local score suppresses the big `ACME-HH-ZYQ` campaign pocket. I’m checking the host concentration rows too; PowerShell tripped over its reserved `$Host` variable name, so I’ll rerun that small extraction with a different variable name.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$hostRows = RowsToObjects $nb.cells[47].outputs[3].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
Write-Output 'HOST RED TOP 1%'
$hostRows | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.01' } | Format-Table -AutoSize | Out-String -Width 340
Write-Output 'HOST RED TOP 5%'
$hostRows | Where-Object { $_.label -eq 'red_team' -and $_.top_fraction -eq '0.05' } | Format-Table -AutoSize | Out-String -Width 340
Write-Output 'HOST BAD_USER TOP 5%'
$hostRows | Where-Object { $_.label -eq 'bad_user' -and $_.top_fraction -eq '0.05' } | Format-Table -AutoSize | Out-String -Width 340`
>
> </details>

</details>

Detector V1 ran, and the answer is: the simple ensemble is **not better as one single score**.

For `red_team`, global semantic is still the clear winner:

```text
global_semantic_rank
top 1%: 61 / 68 red-team
precision: 89.7%
recall: 18.7%
```

The blends collapse toward the within-host behavioral result:

```text
detector_v1_50_50
top 1%: 19 / 68 red-team

detector_v1_semantic_heavy
top 1%: 18 / 68 red-team
```

So blending actually hurts red-team detection because the within-host score downranks the big `ACME-HH-ZYQ` campaign pocket that the semantic score catches extremely well.

But there is one good ensemble result: `bad_user`.

```text
detector_v1_semantic_heavy
top 5%: 24 / 47 bad-user sessions
recall: 51.1%
lift: 10.2x
```

That is much better than global semantic alone at top 5%, which only found `2 / 47`.

My read: don’t force this into one score yet. Use a **two-queue unsupervised detector**:

```text
Queue A: global_semantic_rank
Purpose: campaign/pocket detection
Best for: red_team

Queue B: detector_v1_semantic_heavy or within-host behavioral
Purpose: host-relative/user-relative weirdness
Best for: bad_user and local anomalies
```

That’s actually a stronger story than a single blended score: global semantic catches broad campaign pockets; within-host/behavioral catches locally unusual behavior. The next notebook change should make that explicit as “Detector V1 = two ranked queues,” with side-by-side top-k evaluation and top-session inspection.

> okay you can do that but I think we should also actually work on bettering the best method

<details><summary>6 previous messages</summary>

> I’ll do both: make the two-queue detector explicit, and add a concrete “better the best method” experiment. For the improvement, I’m going after the current weakness of the winner: it uses mean-pooled process SVD vectors, so session-level text can get diluted; I’ll add direct session-level semantic text pooling and compare it against the current best.
>
> The notebook already has the variables needed for this, so I’m inserting the new cells after Detector V1 and before the holdout section. The improvement cell will be label-free during fitting and will only use `red_team`/`bad_user` for the same top-k enrichment checks we’ve been using.
>
> I’m adding this as two separate experiments rather than hiding the distinction: one cell for “improve the global semantic method,” and one cell for “two ranked queues.” That gives us cleaner evidence: if the better semantic representation wins, we’ll see it; if the two-queue design is better operationally, we’ll see that too.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> semantic_v2_md = [
> > "## Semantic Text Session Detector V2\n",
> > "\n",
> > "Improve the current global semantic winner by pooling semantic text directly at the session level, instead of only mean-pooling process SVD embeddings. Labels are used only for evaluation.\n",
> > ]
>
> semantic_v2_code = r'''from sklearn.decomposition import TruncatedSVD
> from scipy import sparse
>
> SESSION_TEXT_V2_SVD_COMPONENTS = 64
> SESSION_TEXT_V2_RANDOM_STATE = RANDOM_STATE + 727
> SESSION_TEXT_V2_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
>
> def sparse_mean_pool_by_session(process_matrix, session_id):
> X = process_matrix.tocsr() if sparse.issparse(process_matrix) else sparse.csr_matrix(np.asarray(process_matrix))
> sid = anomaly_numpy(session_id).astype(int).reshape(-1)
> if X.shape[0] != len(sid):
> raise ValueError(f"feature rows ({X.shape[0]}) != session ids ({len(sid)})")
>
>     n_sessions = int(sid.max()) + 1 if len(sid) else 0
>     selector = sparse.coo_matrix(
>         (np.ones(len(sid), dtype=np.float32), (sid, np.arange(len(sid)))),
>         shape=(n_sessions, len(sid)),
>     ).tocsr()
>     pooled = selector @ X
>     counts = np.bincount(sid, minlength=n_sessions).astype(np.float32)
>     inv_counts = sparse.diags(1.0 / np.maximum(counts, 1.0))
>     return (inv_counts @ pooled).tocsr()
>
> def sparse_max_pool_by_session(process_matrix, session_id):
> X = process_matrix.tocsr() if sparse.issparse(process_matrix) else sparse.csr_matrix(np.asarray(process_matrix))
> sid = anomaly_numpy(session_id).astype(int).reshape(-1)
> if X.shape[0] != len(sid):
> raise ValueError(f"feature rows ({X.shape[0]}) != session ids ({len(sid)})")
>
>     n_sessions = int(sid.max()) + 1 if len(sid) else 0
>     order = np.argsort(sid)
>     sorted_sid = sid[order]
>     pooled_rows = []
>     cursor = 0
>     for session_value in range(n_sessions):
>         start = cursor
>         while cursor < len(sorted_sid) and sorted_sid[cursor] == session_value:
>             cursor += 1
>         row_indices = order[start:cursor]
>         if len(row_indices) == 0:
>             pooled_rows.append(sparse.csr_matrix((1, X.shape[1]), dtype=X.dtype))
>         elif len(row_indices) == 1:
>             pooled_rows.append(X[row_indices[0]])
>         else:
>             pooled_rows.append(sparse.csr_matrix(X[row_indices].max(axis=0)))
>     return sparse.vstack(pooled_rows, format="csr")
>
> def svd*session_embedding(matrix, embedding_name, n_components=SESSION_TEXT_V2_SVD_COMPONENTS):
> n_components = min(n_components, matrix.shape[0] - 1, matrix.shape[1] - 1)
> if n_components < 2:
> raise ValueError(f"{embedding_name}: need at least two SVD components, got {n_components}")
> svd = TruncatedSVD(n_components=n_components, random_state=SESSION_TEXT_V2_RANDOM_STATE)
> embedding = svd.fit_transform(matrix)
> print(
> f"{embedding_name}: source={matrix.shape}, embedding={embedding.shape}, "
> f"explained_variance={svd.explained_variance_ratio*.sum():.3f}"
> )
> return embedding, svd
>
> def score_session_embedding(embedding, embedding_name):
> table = session_meta.copy()
> table["embedding"] = embedding_name
> table["knn_distance_score"] = knn_distance_anomaly_scores(embedding)
> table["isolation_forest_score"] = isolation_forest_anomaly_scores(embedding)
> table["rank_ensemble_score"] = (
> percentile_rank(table["knn_distance_score"]) +
> percentile_rank(table["isolation_forest_score"])
> ) / 2.0
> return table
>
> def session_embedding_enrichment(score_table, score_col, label_col, top_fractions=SESSION_TEXT_V2_TOP_FRACTIONS):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
> rows = []
> for frac in top_fractions:
> n_top = max(1, int(np.ceil(len(ranked) \* frac)))
> top = ranked.head(n_top)
> hits = int(top[label_col].fillna(0).astype(int).sum())
> precision = hits / n_top
> rows.append({
> "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
> "score": score_col,
> "label": label_col,
> "top_fraction": frac,
> "top_n": n_top,
> "hits": hits,
> "total_positive": total_positive,
> "baseline_positive_rate": baseline,
> "top_positive_rate": precision,
> "recall_at_top": hits / total_positive if total_positive else np.nan,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> })
> return rows
>
> def session_embedding_host_report(score_table, score_col, label_col, top_fraction):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_token)
> host_counts = top["primary_host"].fillna("").astype(str).value_counts()
> process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
> hits = int(top[label_col].fillna(0).astype(int).sum())
> baseline = float(ranked[label_col].fillna(0).astype(int).mean())
> precision = hits / n_top
> return {
> "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
> "score": score_col,
> "label": label_col,
> "top_fraction": top_fraction,
> "top_n": n_top,
> "hits": hits,
> "top_positive_rate": precision,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
> "top_host": host_counts.index[0] if len(host_counts) else "",
> "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
> "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
> "top_process_family": process_counts.index[0] if len(process_counts) else "",
> "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
> "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
> }
>
> if "semantic_text_features" not in globals():
> raise RuntimeError("Run the Semantic Ablation cell that creates `semantic_text_features` first.")
>
> session_ids = data["process"].session_id.cpu()
> text_session_mean = sparse_mean_pool_by_session(semantic_text_features, session_ids)
> text_session_max = sparse_max_pool_by_session(semantic_text_features, session_ids)
> text_session_mean_max = sparse.hstack([text_session_mean, text_session_max], format="csr")
>
> semantic_text_v2_embeddings = {}
> semantic_text_v2_svd_models = {}
> for embedding_name, matrix in {
> "semantic_text_session_tfidf_mean_svd": text_session_mean,
> "semantic_text_session_tfidf_max_svd": text_session_max,
> "semantic_text_session_tfidf_mean_max_svd": text_session_mean_max,
> }.items():
> semantic_text_v2_embeddings[embedding_name], semantic_text_v2_svd_models[embedding_name] = svd_session_embedding(
> matrix,
> embedding_name,
> )
>
> semantic_text_v2_score_tables = {
> embedding_name: score_session_embedding(embedding, embedding_name)
> for embedding_name, embedding in semantic_text_v2_embeddings.items()
> }
>
> semantic_text_v2_rows = []
> for table in semantic_text_v2_score_tables.values():
> for score_col in ["knn_distance_score", "isolation_forest_score", "rank_ensemble_score"]:
> for label_col in ["red_team", "bad_user"]:
> semantic_text_v2_rows.extend(session_embedding_enrichment(table, score_col, label_col))
>
> semantic_text_v2_enrichment = pd.DataFrame(semantic_text_v2_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(semantic_text_v2_enrichment)
>
> semantic_text_v2_host_rows = []
> for table in semantic_text_v2_score_tables.values():
> for score_col in ["knn_distance_score", "isolation_forest_score", "rank_ensemble_score"]:
> for label_col in ["red_team", "bad_user"]:
> for frac in [0.01, 0.05, 0.10]:
> semantic_text_v2_host_rows.append(session_embedding_host_report(table, score_col, label_col, frac))
>
> semantic_text_v2_host_dominance = pd.DataFrame(semantic_text_v2_host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> display(semantic_text_v2_host_dominance)
>
> baseline_global = detector_v1_enrichment[
> (detector_v1_enrichment["detector_score"] == "global_semantic_rank") &
> (detector_v1_enrichment["label"] == "red_team") &
> (detector_v1_enrichment["top_fraction"] == 0.01)
> ].rename(columns={"detector_score": "score"}).assign(embedding="process_svd_mean_baseline")
>
> best_v2_red = semantic_text_v2_enrichment[
> (semantic_text_v2_enrichment["label"] == "red_team") &
> (semantic_text_v2_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> comparison_cols = [
> > "embedding",
> > "score",
> > "label",
> > "top_fraction",
> > "hits",
> > "top_n",
> > "total_positive",
> > "top_positive_rate",
> > "recall_at_top",
> > "lift_vs_baseline",
> > ]
> print("Current semantic baseline vs best direct session-text V2 at top-1% red_team:")
> display(pd.concat([baseline_global[comparison_cols], best_v2_red[comparison_cols]], ignore_index=True))
>
> if len(best_v2_red):
> best_embedding = best_v2_red.iloc[0]["embedding"]
> best_score = best_v2_red.iloc[0]["score"]
> print(f"Best Semantic Text V2 top-1% red_team score: {best_embedding} / {best_score}")
> top_semantic_text_v2_sessions = semantic_text_v2_score_tables[best_embedding].sort_values(best_score, ascending=False).head(25)
> display_cols = [
> > "session_id",
> > best_score,
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display(top_semantic_text_v2_sessions[display_cols])
> '''.splitlines(keepends=True)
>
> two_queue_md = [
> > "## Detector V1: Two Ranked Queues\n",
> > "\n",
> > "Keep campaign-pocket detection and local-context detection as separate ranked queues instead of forcing them into one blended score.\n",
> > ]
>
> two_queue_code = r'''TWO_QUEUE_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
>
> TWO_QUEUE_SPECS = [
> > {
> > "queue": "campaign_queue_global_semantic",
> > "table_name": "detector_v1_table",
> > "score_col": "global_semantic_rank",
> > "purpose": "global campaign/pocket detection",
> > },
> > {
> > "queue": "local_context_queue_semantic_behavioral",
> > "table_name": "detector_v1_table",
> > "score_col": "detector_v1_semantic_heavy",
> > "purpose": "host-relative and user-context anomalies",
> > },
> > ]
>
> def get_table_by_name(table_name):
> if table_name not in globals():
> raise RuntimeError(f"Run the cell that creates `{table_name}` first.")
> return globals()[table_name]
>
> def two_queue_enrichment(table, score_col, queue_name, purpose, label_col, top_fractions=TWO_QUEUE_TOP_FRACTIONS):
> ranked = table.sort_values(score_col, ascending=False).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
> rows = []
> for frac in top_fractions:
> n_top = max(1, int(np.ceil(len(ranked) \* frac)))
> top = ranked.head(n_top)
> hits = int(top[label_col].fillna(0).astype(int).sum())
> precision = hits / n_top
> rows.append({
> "queue": queue_name,
> "purpose": purpose,
> "score": score_col,
> "label": label_col,
> "top_fraction": frac,
> "top_n": n_top,
> "hits": hits,
> "total_positive": total_positive,
> "baseline_positive_rate": baseline,
> "top_positive_rate": precision,
> "recall_at_top": hits / total_positive if total_positive else np.nan,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> })
> return rows
>
> def two_queue_host_report(table, score_col, queue_name, label_col, top_fraction):
> ranked = table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_token)
> host_counts = top["primary_host"].fillna("").astype(str).value_counts()
> process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
> hits = int(top[label_col].fillna(0).astype(int).sum())
> baseline = float(ranked[label_col].fillna(0).astype(int).mean())
> precision = hits / n_top
> return {
> "queue": queue_name,
> "score": score_col,
> "label": label_col,
> "top_fraction": top_fraction,
> "top_n": n_top,
> "hits": hits,
> "top_positive_rate": precision,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
> "top_host": host_counts.index[0] if len(host_counts) else "",
> "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
> "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
> "top_process_family": process_counts.index[0] if len(process_counts) else "",
> "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
> "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
> }
>
> two_queue_rows = []
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> for label_col in ["red_team", "bad_user"]:
> two_queue_rows.extend(
> two_queue_enrichment(
> table,
> spec["score_col"],
> spec["queue"],
> spec["purpose"],
> label_col,
> )
> )
>
> two_queue_enrichment_report = pd.DataFrame(two_queue_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(two_queue_enrichment_report)
>
> two_queue_host_rows = []
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> for label_col in ["red_team", "bad_user"]:
> for frac in [0.01, 0.05, 0.10]:
> two_queue_host_rows.append(
> two_queue_host_report(table, spec["score_col"], spec["queue"], label_col, frac)
> )
>
> two_queue_host_dominance = pd.DataFrame(two_queue_host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> display(two_queue_host_dominance)
>
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> score_col = spec["score_col"]
> print(f"Top sessions for {spec['queue']} ({spec['purpose']})")
> display_cols = [
> > "session_id",
> > score_col,
> > "global_semantic_rank",
> > "within_host_behavioral_rank",
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display(table.sort_values(score_col, ascending=False).head(20)[display_cols])
> '''.splitlines(keepends=True)
>
> new_sections = [
> > ("## Semantic Text Session Detector V2", semantic_v2_md, semantic_v2_code),
> > ("## Detector V1: Two Ranked Queues", two_queue_md, two_queue_code),
> > ]
>
> # Remove existing copies of these sections if present.
>
> idx = 0
> while idx < len(nb["cells"]):
> cell = nb["cells"][idx]
> src = ''.join(cell.get("source", []))
> matched = False
> for heading, _, _ in new_sections:
> if cell.get("cell_type") == "markdown" and src.startswith(heading):
> del nb["cells"][idx:idx + 2]
> matched = True
> break
> if not matched:
> idx += 1
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> if cell.get("cell_type") == "markdown" and ''.join(cell.get("source", [])).startswith("## Unsupervised Holdout Validation"):
> insert_at = idx
> break
> if insert_at is None:
> insert_at = len(nb["cells"])
>
> cells*to_insert = []
> for *, md, code in new_sections:
> cells_to_insert.append({"cell_type": "markdown", "metadata": {}, "source": md})
> cells_to_insert.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code})
>
> nb["cells"][insert_at:insert_at] = cells_to_insert
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted {len(new_sections)} sections at cell index {insert_at}.")
> '@
> $script | py -3 -``
>
> </details>
>
> Both sections are inserted. I’m validating the notebook cell order and parsing the two new code cells now; these are larger cells, so I want to catch syntax mistakes before you run them.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
starts = ['from sklearn.decomposition import TruncatedSVD', 'TWO_QUEUE_TOP_FRACTIONS']
found = {}
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    for start in starts:
        if src.startswith(start):
            ast.parse(src)
            found[start] = (i, len(src.splitlines()))
for start in starts:
    if start not in found:
        raise SystemExit(f'cell not found: {start}')
    print(f'compiled cell {found[start][0]} lines={found[start][1]} start={start}')
'@
$script | py -3 -`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 44; $i -le 56; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} first={3}" -f $i, $cell.cell_type, $cell.execution_count, $first.Substring(0, [Math]::Min(155, $first.Length)))
}``
> - Ran `rg -n "Semantic Text Session Detector V2|SESSION_TEXT_V2|Detector V1: Two Ranked Queues|TWO_QUEUE" my_sessionization.ipynb`
>
> </details>

</details>

Added both next steps to [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:16835).

New sections:

**Semantic Text Session Detector V2**
This tries to improve the current best method by building session-level semantic text embeddings directly:

```text
process TF-IDF text features
-> session mean pooling
-> session max pooling
-> mean+max pooling
-> SVD
-> kNN / IsolationForest anomaly scores
```

This should tell us whether direct session text pooling beats the current winner, which is mean-pooled process SVD.

**Detector V1: Two Ranked Queues**
This makes the operational version explicit:

```text
Queue A: campaign_queue_global_semantic
Purpose: global campaign/pocket detection

Queue B: local_context_queue_semantic_behavioral
Purpose: host-relative/user-context anomalies
```

It reports top-k enrichment, host concentration, and top sessions for each queue.

Run cells 49 and 51 next. I validated both new code cells parse cleanly.

> look

<details><summary>11 previous messages</summary>

> I’ll look at the new Semantic Text V2 and two-queue outputs now, then compare them against the previous best score so we know whether the new method actually improved anything.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 48; $i -lt [Math]::Min($nb.cells.Count, 56); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(150, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 2500) { $text = $text.Substring(0,2500) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 3200) { $text = $text.Substring(0,3200) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The new Semantic Text V2 did improve the best method: direct session-level TF-IDF mean pooling found 63 red-team sessions in the top 68, up from 61. I’m going to make a small follow-up patch so the two-queue section uses this V2 campaign queue instead of the older baseline queue.
>
> The change is small but important: Queue A should now be the improved direct session-text V2 score, while Queue B stays the host/local context score. That keeps the operational story aligned with the best measured method instead of yesterday’s baseline.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> new_code = r'''TWO_QUEUE_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
> CAMPAIGN_QUEUE_V2_EMBEDDING = "semantic_text_session_tfidf_mean_svd"
> CAMPAIGN_QUEUE_V2_SCORE = "isolation_forest_score"
> LOCAL_CONTEXT_QUEUE_SCORE = "detector_v1_semantic_heavy"
>
> def two_queue_rank01(values):
> return pd.Series(np.asarray(values, dtype=np.float64)).rank(method="average", pct=True).to_numpy()
>
> def build_campaign_queue_v2_table():
> if "semantic_text_v2_score_tables" not in globals():
> raise RuntimeError("Run the Semantic Text Session Detector V2 cell first.")
> if CAMPAIGN_QUEUE_V2_EMBEDDING not in semantic_text_v2_score_tables:
> raise KeyError(f"Missing Semantic Text V2 embedding: {CAMPAIGN_QUEUE_V2_EMBEDDING}")
> table = semantic_text_v2_score_tables[CAMPAIGN_QUEUE_V2_EMBEDDING].copy()
> table["campaign_semantic_v2_rank"] = two_queue_rank01(table[CAMPAIGN_QUEUE_V2_SCORE])
> if "global_semantic_rank" not in table.columns:
> table["global_semantic_rank"] = table["campaign_semantic_v2_rank"]
> if "within_host_behavioral_rank" not in table.columns:
> table["within_host_behavioral_rank"] = np.nan
> return table
>
> campaign_queue_v2_table = build_campaign_queue_v2_table()
>
> TWO_QUEUE_SPECS = [
> > {
> > "queue": "campaign_queue_session_text_v2",
> > "table_name": "campaign_queue_v2_table",
> > "score_col": "campaign_semantic_v2_rank",
> > "purpose": "global campaign/pocket detection with direct session text",
> > },
> > {
> > "queue": "local_context_queue_semantic_behavioral",
> > "table_name": "detector_v1_table",
> > "score_col": LOCAL_CONTEXT_QUEUE_SCORE,
> > "purpose": "host-relative and user-context anomalies",
> > },
> > ]
>
> def get_table_by_name(table_name):
> if table_name not in globals():
> raise RuntimeError(f"Run the cell that creates `{table_name}` first.")
> return globals()[table_name]
>
> def two_queue_enrichment(table, score_col, queue_name, purpose, label_col, top_fractions=TWO_QUEUE_TOP_FRACTIONS):
> ranked = table.sort_values(score_col, ascending=False).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
> rows = []
> for frac in top_fractions:
> n_top = max(1, int(np.ceil(len(ranked) \* frac)))
> top = ranked.head(n_top)
> hits = int(top[label_col].fillna(0).astype(int).sum())
> precision = hits / n_top
> rows.append({
> "queue": queue_name,
> "purpose": purpose,
> "score": score_col,
> "label": label_col,
> "top_fraction": frac,
> "top_n": n_top,
> "hits": hits,
> "total_positive": total_positive,
> "baseline_positive_rate": baseline,
> "top_positive_rate": precision,
> "recall_at_top": hits / total_positive if total_positive else np.nan,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> })
> return rows
>
> def two_queue_host_report(table, score_col, queue_name, label_col, top_fraction):
> ranked = table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_token)
> host_counts = top["primary_host"].fillna("").astype(str).value_counts()
> process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
> hits = int(top[label_col].fillna(0).astype(int).sum())
> baseline = float(ranked[label_col].fillna(0).astype(int).mean())
> precision = hits / n_top
> return {
> "queue": queue_name,
> "score": score_col,
> "label": label_col,
> "top_fraction": top_fraction,
> "top_n": n_top,
> "hits": hits,
> "top_positive_rate": precision,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
> "top_host": host_counts.index[0] if len(host_counts) else "",
> "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
> "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
> "top_process_family": process_counts.index[0] if len(process_counts) else "",
> "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
> "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
> }
>
> two_queue_rows = []
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> for label_col in ["red_team", "bad_user"]:
> two_queue_rows.extend(
> two_queue_enrichment(
> table,
> spec["score_col"],
> spec["queue"],
> spec["purpose"],
> label_col,
> )
> )
>
> two_queue_enrichment_report = pd.DataFrame(two_queue_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(two_queue_enrichment_report)
>
> two_queue_host_rows = []
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> for label_col in ["red_team", "bad_user"]:
> for frac in [0.01, 0.05, 0.10]:
> two_queue_host_rows.append(
> two_queue_host_report(table, spec["score_col"], spec["queue"], label_col, frac)
> )
>
> two_queue_host_dominance = pd.DataFrame(two_queue_host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> display(two_queue_host_dominance)
>
> for spec in TWO_QUEUE_SPECS:
> table = get_table_by_name(spec["table_name"])
> score_col = spec["score_col"]
> print(f"Top sessions for {spec['queue']} ({spec['purpose']})")
> preferred_cols = [
> > "session_id",
> > score_col,
> > "global_semantic_rank",
> > "within_host_behavioral_rank",
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display_cols = []
> for col in preferred_cols:
> if col in table.columns and col not in display_cols:
> display_cols.append(col)
> display(table.sort_values(score_col, ascending=False).head(20)[display_cols])
> '''.splitlines(keepends=True)
>
> for cell in nb['cells']:
> src = ''.join(cell.get('source', []))
> if src.startswith('TWO_QUEUE_TOP_FRACTIONS'):
> cell['source'] = new_code
> cell['execution_count'] = None
> cell['outputs'] = []
> break
> else:
> raise SystemExit('two queue cell not found')
>
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print('Updated two-queue cell to use Semantic Text V2 for campaign queue.')
> '@
> $script | py -3 -``
>
> </details>
>
> The two-queue cell now points at the improved Semantic Text V2 campaign queue. I’m validating the edited cell parses cleanly and checking the cell order/output reset.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('TWO_QUEUE_TOP_FRACTIONS'):
        ast.parse(src)
        print(f'compiled updated two-queue cell at index {i}, lines={len(src.splitlines())}, exec={cell.get("execution_count")}, outputs={len(cell.get("outputs", []))}')
        break
else:
    raise SystemExit('two queue cell not found')
'@
$script | py -3 -`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 48; $i -le 52; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} outputs={3} first={4}" -f $i, $cell.cell_type, $cell.execution_count, @($cell.outputs).Count, $first.Substring(0, [Math]::Min(150, $first.Length)))
}``
> - Ran `rg -n "campaign_queue_session_text_v2|CAMPAIGN_QUEUE_V2|campaign_queue_global_semantic" my_sessionization.ipynb`
>
> </details>
>
> The V2 result is clean enough to call: direct session-level text pooling improved the campaign detector. I’m doing one last exact-row extraction for the key V2 metrics so the summary has the real numbers, not just the truncated notebook view.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$v2 = RowsToObjects $nb.cells[49].outputs[1].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
Write-Output 'V2 RED TOP 1/5/10'
$v2 | Where-Object { $_.label -eq 'red_team' -and ($_.top_fraction -eq '0.01' -or $_.top_fraction -eq '0.05' -or $_.top_fraction -eq '0.1') } | Select-Object -First 15 | Format-Table -AutoSize | Out-String -Width 280
Write-Output 'V2 BAD_USER TOP 1/5/10'
$v2 | Where-Object { $_.label -eq 'bad_user' -and ($_.top_fraction -eq '0.01' -or $_.top_fraction -eq '0.05' -or $_.top_fraction -eq '0.1') } | Select-Object -First 15 | Format-Table -AutoSize | Out-String -Width 280`
>
> </details>
>
> PowerShell’s string matching was being annoyingly literal on the red-team fractions, so I’m checking those rows directly. The important number is already in the notebook comparison, but I want the broader top-k rows too.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
function RowsToObjects($viewer) {
  $cols = @($viewer.columns | ForEach-Object { $_.name })
  $objs = @()
  foreach ($row in $viewer.rows) {
    $obj = [ordered]@{}
    for ($i=0; $i -lt $cols.Count; $i++) { $obj[$cols[$i]] = $row[$i] }
    $objs += [pscustomobject]$obj
  }
  return $objs
}
$v2 = RowsToObjects $nb.cells[49].outputs[1].data.'application/vnd.microsoft.datawrangler.viewer.v0+json'
$v2 | Where-Object { $_.label -eq 'red_team' } | Select-Object -First 20 | Format-Table -AutoSize | Out-String -Width 300`
>
> </details>

</details>

The new V2 semantic method actually improved the current best.

Best previous campaign score:

```text
process_svd_mean_baseline / global_semantic_rank
top 1% red_team: 61 / 68
precision: 89.7%
lift: 18.48x
```

New best:

```text
semantic_text_session_tfidf_mean_svd / isolation_forest_score
top 1% red_team: 63 / 68
precision: 92.6%
lift: 19.08x
```

Even stronger at the very top:

```text
top 0.5% red_team: 34 / 34
precision: 100%
```

So yes, directly pooling semantic text at the session level beats mean-pooling process SVD. Nice. That was the right direction.

Caveat: it is still mostly the `ACME-HH-ZYQ` pocket, with lots of `powershell.exe`, `cmd.exe`, and `reg.exe`. That means it is an excellent campaign detector, but still host/campaign concentrated.

Bad-user did not improve with V2. The better bad-user queue is still the semantic-heavy local/context score:

```text
detector_v1_semantic_heavy
top 5% bad_user: 24 / 47
recall: 51.1%
lift: 10.2x
```

I also patched the **Detector V1: Two Ranked Queues** cell so Queue A now uses the improved V2 campaign detector instead of the older global semantic baseline. Run cell 51 again; its previous output is stale because I updated the code after reading the results.

> ok look and change whats next

<details><summary>9 previous messages</summary>

> I’ll look at the freshly rerun two-queue outputs first, then I’ll update the notebook with the next concrete step rather than leaving this as analysis-only. My hunch is the next change should move the improved V2 detector into a harder validation/check section, but I’ll let the outputs decide.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 48; $i -lt [Math]::Min($nb.cells.Count, 56); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(150, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 2500) { $text = $text.Substring(0,2500) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 3600) { $text = $text.Substring(0,3600) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The latest output confirms the two-queue version is now aligned with the improved V2 campaign queue: top 1% red-team is `63 / 68`, and top 0.5% is `34 / 34`. The next change should be a harder validation of V2, because right now the strongest result is still mostly the `ACME-HH-ZYQ` pocket.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
foreach ($idx in @(43,47,49,53,54)) {
  if ($idx -lt $nb.cells.Count) {
    Write-Output ('--- CELL ' + $idx + ' ---')
    $src = ($nb.cells[$idx].source -join '')
    if ($src.Length -gt 10000) { $src = $src.Substring(0,10000) + "`n...TRUNCATED..." }
    Write-Output $src
  }
}``
> - Ran `rg -n "class SemanticProcessFeatureBuilder|SemanticFeatureConfig|holdout_train_fit|holdout_test_sample|make_stratified_process_sample|build_graph\(" my_sessionization.ipynb semantic_feature_baseline.py hetero_graph_builder.py`
>
> </details>
>
> I found the right place to change the notebook: after the two-queue section, before the older process-level holdout checks. I’m going to add a new holdout cell for **session-level V2** specifically: fit text features/SVD/anomaly model on `df_train`, score `df_test` sessions, and report top-k enrichment.
>
> I’m making the next cell fit on `df_train` and score `df_test` sessions, which is a much stricter question than the current in-sample V2 result. It will still use labels only for the final enrichment table, not to fit TF-IDF, SVD, Isolation Forest, or kNN scoring.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> markdown_source = [
> > "## Holdout Session Detector V2: Train-Fit / Test-Score\n",
> > "\n",
> > "Fit the direct session-text detector on `df_train` and score `df_test` sessions. This is the next hard check for whether the improved V2 method survives outside the in-sample ACME slice.\n",
> > ]
>
> code_source = r'''from sklearn.ensemble import IsolationForest
> from sklearn.neighbors import NearestNeighbors
> from sklearn.preprocessing import StandardScaler
>
> HOLDOUT_SESSION_V2_K = 15
> HOLDOUT_SESSION_V2_SVD_COMPONENTS = 64
> HOLDOUT_SESSION_V2_RANDOM_STATE = RANDOM_STATE + 909
> HOLDOUT_SESSION_V2_TOP_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10]
>
> def build*holdout_session_graph(sample_df, name):
> graph_builder = HeterogeneousGeometricGraphBuilder(
> rare_file_max_degree=10,
> add_same_user_edges=False,
> add_network_edges=False,
> )
> graph_data = graph_builder.build_graph(sample_df)
> print(
> f"{name}: processes={graph_builder.report*['node_types']['process']:,}, "
> f"sessions={graph*builder.report*['n_sessions']:,}, "
> f"max*session={graph_builder.report*['max_session_size']:,}"
> )
> return graph*data, graph_builder.report*
>
> def graph_process_ids(graph_data):
> return pd.Index([str(pid) for pid in graph_data["process"].external_id], name="pid_hash")
>
> def make_process_meta_for_graph(sample_df, graph_data):
> return make_holdout_metadata(sample_df, graph_process_ids(graph_data))
>
> def fit*text_session_svd(train_session_features, test_session_features):
> n_components = min(
> HOLDOUT_SESSION_V2_SVD_COMPONENTS,
> train_session_features.shape[0] - 1,
> train_session_features.shape[1] - 1,
> )
> if n_components < 2:
> raise ValueError(f"Need at least two SVD components, got {n_components}")
> svd = TruncatedSVD(n_components=n_components, random_state=HOLDOUT_SESSION_V2_RANDOM_STATE)
> train_emb = svd.fit_transform(train_session_features)
> test_emb = svd.transform(test_session_features)
> print(
> f"Holdout session text SVD: train={train_session_features.shape}->{train_emb.shape}, "
> f"test={test_session_features.shape}->{test_emb.shape}, "
> f"explained_variance={svd.explained_variance_ratio*.sum():.3f}"
> )
> return train_emb, test_emb, svd
>
> def train*fit_knn_distance_scores(train_matrix, test_matrix, k=HOLDOUT_SESSION_V2_K):
> scaler = StandardScaler()
> train_x = scaler.fit_transform(anomaly_numpy(train_matrix))
> test_x = scaler.transform(anomaly_numpy(test_matrix))
> n_neighbors = min(k, len(train_x))
> nn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
> nn.fit(train_x)
> distances, * = nn.kneighbors(test_x)
> return distances.mean(axis=1)
>
> def train_fit_isolation_scores(train_matrix, test_matrix):
> scaler = StandardScaler()
> train_x = scaler.fit_transform(anomaly_numpy(train_matrix))
> test_x = scaler.transform(anomaly_numpy(test_matrix))
> model = IsolationForest(
> n_estimators=400,
> contamination="auto",
> random_state=HOLDOUT_SESSION_V2_RANDOM_STATE,
> )
> model.fit(train_x)
> return -model.decision_function(test_x), model
>
> def build_holdout_session_score_table(test_session_meta, train_embedding, test_embedding, embedding_name):
> table = test_session_meta.copy()
> table["embedding"] = embedding_name
> table["knn_train_distance_score"] = train_fit_knn_distance_scores(train_embedding, test_embedding)
> table["isolation_forest_train_score"], isolation_model = train_fit_isolation_scores(train_embedding, test_embedding)
> table["rank_ensemble_train_score"] = (
> percentile_rank(table["knn_train_distance_score"]) +
> percentile_rank(table["isolation_forest_train_score"])
> ) / 2.0
> return table, isolation_model
>
> def holdout_session_enrichment(score_table, score_col, label_col, top_fractions=HOLDOUT_SESSION_V2_TOP_FRACTIONS):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> y = ranked[label_col].fillna(0).astype(int).to_numpy()
> baseline = float(y.mean()) if len(y) else np.nan
> total_positive = int(y.sum())
> rows = []
> for frac in top_fractions:
> n_top = max(1, int(np.ceil(len(ranked) \* frac)))
> top = ranked.head(n_top)
> hits = int(top[label_col].fillna(0).astype(int).sum())
> precision = hits / n_top
> rows.append({
> "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
> "score": score_col,
> "label": label_col,
> "top_fraction": frac,
> "top_n": n_top,
> "hits": hits,
> "total_positive": total_positive,
> "baseline_positive_rate": baseline,
> "top_positive_rate": precision,
> "recall_at_top": hits / total_positive if total_positive else np.nan,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> })
> return rows
>
> def holdout_session_host_dominance(score_table, score_col, label_col, top_fraction):
> ranked = score_table.sort_values(score_col, ascending=False).reset_index(drop=True)
> n_top = max(1, int(np.ceil(len(ranked) \* top_fraction)))
> top = ranked.head(n_top).copy()
> top["top_process_family"] = top["top_processes"].map(first_token)
> host_counts = top["primary_host"].fillna("").astype(str).value_counts()
> process_counts = top["top_process_family"].fillna("").astype(str).value_counts()
> hits = int(top[label_col].fillna(0).astype(int).sum())
> baseline = float(ranked[label_col].fillna(0).astype(int).mean())
> precision = hits / n_top
> return {
> "embedding": ranked.loc[0, "embedding"] if len(ranked) else "",
> "score": score_col,
> "label": label_col,
> "top_fraction": top_fraction,
> "top_n": n_top,
> "hits": hits,
> "top_positive_rate": precision,
> "lift_vs_baseline": precision / baseline if baseline > 0 else np.nan,
> "distinct_hosts": int(top["primary_host"].nunique(dropna=False)),
> "top_host": host_counts.index[0] if len(host_counts) else "",
> "top_host_rows": int(host_counts.iloc[0]) if len(host_counts) else 0,
> "top_host_share": float(host_counts.iloc[0] / n_top) if len(host_counts) else np.nan,
> "top_process_family": process_counts.index[0] if len(process_counts) else "",
> "top_process_family_rows": int(process_counts.iloc[0]) if len(process_counts) else 0,
> "top_process_family_share": float(process_counts.iloc[0] / n_top) if len(process_counts) else np.nan,
> }
>
> holdout_train_graph, holdout_train_graph_report = build_holdout_session_graph(holdout_train_fit, "holdout train-fit")
> holdout_test_graph, holdout_test_graph_report = build_holdout_session_graph(holdout_test_sample, "holdout test-score")
>
> holdout_train_process_meta = make_process_meta_for_graph(holdout_train_fit, holdout_train_graph)
> holdout_test_process_meta = make_process_meta_for_graph(holdout_test_sample, holdout_test_graph)
> holdout_train_session_meta = make_session_metadata(
> holdout_train_process_meta,
> holdout_train_graph["process"].session_id.cpu(),
> )
> holdout_test_session_meta = make_session_metadata(
> holdout_test_process_meta,
> holdout_test_graph["process"].session_id.cpu(),
> )
>
> holdout_train_text_features_for_graph = holdout_text_builder.transform(
> holdout_train_fit,
> process_ids=graph_process_ids(holdout_train_graph),
> )
> holdout_test_text_features_for_graph = holdout_text_builder.transform(
> holdout_test_sample,
> process_ids=graph_process_ids(holdout_test_graph),
> )
>
> holdout_train_session_text_mean = sparse_mean_pool_by_session(
> holdout_train_text_features_for_graph,
> holdout_train_graph["process"].session_id.cpu(),
> )
> holdout_test_session_text_mean = sparse_mean_pool_by_session(
> holdout_test_text_features_for_graph,
> holdout_test_graph["process"].session_id.cpu(),
> )
>
> holdout_train_session_text_svd, holdout_test_session_text_svd, holdout_session_text_svd_model = fit_text_session_svd(
> holdout_train_session_text_mean,
> holdout_test_session_text_mean,
> )
>
> holdout_session_v2_table, holdout_session_v2_isolation_model = build_holdout_session_score_table(
> holdout_test_session_meta,
> holdout_train_session_text_svd,
> holdout_test_session_text_svd,
> "holdout_session_text_v2_mean_svd",
> )
>
> print(
> "Holdout test sessions: "
> f"{len(holdout_session_v2_table):,}; "
> f"red_team={int(holdout_session_v2_table['red_team'].sum()):,}; "
> f"bad_user={int(holdout_session_v2_table['bad_user'].sum()):,}"
> )
> print(
> "Train-fit labels shown only for context: "
> f"red_team={int(holdout_train_session_meta['red_team'].sum()):,}; "
> f"bad_user={int(holdout_train_session_meta['bad_user'].sum()):,}"
> )
>
> holdout_session_v2_rows = []
> for score_col in ["knn_train_distance_score", "isolation_forest_train_score", "rank_ensemble_train_score"]:
> for label_col in ["red_team", "bad_user"]:
> holdout_session_v2_rows.extend(holdout_session_enrichment(holdout_session_v2_table, score_col, label_col))
>
> holdout_session_v2_enrichment = pd.DataFrame(holdout_session_v2_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(holdout_session_v2_enrichment)
>
> holdout_session_v2_host_rows = []
> for score_col in ["knn_train_distance_score", "isolation_forest_train_score", "rank_ensemble_train_score"]:
> for label_col in ["red_team", "bad_user"]:
> for frac in [0.01, 0.05, 0.10]:
> holdout_session_v2_host_rows.append(
> holdout_session_host_dominance(holdout_session_v2_table, score_col, label_col, frac)
> )
>
> holdout_session_v2_host_dominance = pd.DataFrame(holdout_session_v2_host_rows).sort_values(
> ["label", "top_fraction", "lift_vs_baseline"],
> ascending=[True, True, False],
> ).reset_index(drop=True)
>
> display(holdout_session_v2_host_dominance)
>
> best_holdout_red = holdout_session_v2_enrichment[
> (holdout_session_v2_enrichment["label"] == "red_team") &
> (holdout_session_v2_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False).head(1)
>
> if len(best_holdout_red):
> best_score = best_holdout_red.iloc[0]["score"]
> print(f"Best holdout top-1% red_team score: {best_score}")
> display_cols = [
> > "session_id",
> > best_score,
> > "n_processes",
> > "red_team",
> > "bad_user",
> > "primary_host",
> > "top_users",
> > "top_processes",
> > ]
> display(holdout_session_v2_table.sort_values(best_score, ascending=False).head(25)[display_cols])
> '''.splitlines(keepends=True)
>
> markdown_cell = {"cell_type": "markdown", "metadata": {}, "source": markdown_source}
> code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_source}
>
> # Replace existing section if present.
>
> idx = 0
> while idx < len(nb["cells"]):
> cell = nb["cells"][idx]
> src = ''.join(cell.get("source", []))
> if cell.get("cell_type") == "markdown" and src.startswith("## Holdout Session Detector V2"):
> del nb["cells"][idx:idx + 2]
> continue
> idx += 1
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> src = ''.join(cell.get("source", []))
> if src.startswith("from hetero_graph_builder import DEFAULT_PROCESS_FEATURES"):
> insert_at = idx + 1
> break
> if insert_at is None:
> raise SystemExit("Could not find holdout setup cell insertion point")
>
> nb["cells"][insert_at:insert_at] = [markdown_cell, code_cell]
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted Holdout Session Detector V2 section at cell index {insert_at}.")
> '@
> $script | py -3 -``
>
> </details>
>
> The holdout V2 section is inserted after the holdout setup cell, so it can reuse the existing train/test samples and text builder. I’m validating parse and cell order now.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('from sklearn.ensemble import IsolationForest') and 'HOLDOUT_SESSION_V2' in src:
        ast.parse(src)
        print(f'compiled holdout session V2 cell at index {i}, lines={len(src.splitlines())}, exec={cell.get("execution_count")}, outputs={len(cell.get("outputs", []))}')
        break
else:
    raise SystemExit('holdout session V2 cell not found')
'@
$script | py -3 -`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 50; $i -le 58; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} outputs={3} first={4}" -f $i, $cell.cell_type, $cell.execution_count, @($cell.outputs).Count, $first.Substring(0, [Math]::Min(160, $first.Length)))
}``
> - Ran `rg -n "Holdout Session Detector V2|HOLDOUT_SESSION_V2|holdout_session_v2_enrichment|holdout_session_v2_host_dominance" my_sessionization.ipynb`
>
> </details>

</details>

The rerun looks good: the two-queue cell is now using the improved campaign queue.

Current best in-sample queue:

```text
campaign_queue_session_text_v2
top 0.5% red_team: 34 / 34
top 1% red_team: 63 / 68
top 5% red_team: 277 / 337
top 10% red_team: 327 / 674
```

So V2 is now the best campaign detector. The local/context queue is still better for `bad_user` at top 5%:

```text
local_context_queue_semantic_behavioral
top 5% bad_user: 24 / 47
```

I changed the notebook’s next step by adding **Holdout Session Detector V2: Train-Fit / Test-Score** in [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:24454).

That new section does the harder test:

```text
fit TF-IDF/SVD/anomaly model on df_train sessions
score df_test sessions
evaluate top-k enrichment with labels only afterward
```

Run cell 55 next, after the existing holdout setup cell 53. This is the right next question: does the improved V2 method survive a train-fit/test-score setup, or was it mostly an in-sample ACME pocket win?

> look

<details><summary>8 previous messages</summary>

> I’ll check the holdout V2 section now. The main thing I’m looking for is whether the train-fit/test-score version ran cleanly and whether the V2 campaign detector still has top-k enrichment once it’s scored on held-out sessions.
>
> <details><summary>Ran a command</summary>
>
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 52; $i -lt [Math]::Min($nb.cells.Count, 59); $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  $outs = @($cell.outputs)
  Write-Output ("CELL {0} exec={1} type={2} first={3} outputs={4}" -f $i, $cell.execution_count, $cell.cell_type, $first.Substring(0, [Math]::Min(160, $first.Length)), $outs.Count)
  for ($j = 0; $j -lt $outs.Count; $j++) {
    $out = $outs[$j]
    Write-Output ("  output {0}: {1}" -f $j, $out.output_type)
    if ($out.output_type -eq 'stream') {
      $text = ($out.text -join '')
      if ($text.Length -gt 3000) { $text = $text.Substring(0,3000) }
      Write-Output ('    stream: ' + ($text -replace "`n", ' | '))
    } elseif ($out.output_type -eq 'error') {
      Write-Output ('    error: ' + $out.ename + ' ' + $out.evalue)
      if ($out.traceback) { Write-Output (($out.traceback -join "`n") -replace "`e\[[0-9;]*m", '') }
    } elseif ($out.data) {
      $keys = ($out.data.PSObject.Properties.Name -join ', ')
      Write-Output ('    data keys: ' + $keys)
      if ($out.data.'text/plain') {
        $text = ($out.data.'text/plain' -join '')
        if ($text.Length -gt 4200) { $text = $text.Substring(0,4200) }
        Write-Output ('    text: ' + ($text -replace "`n", ' | '))
      }
    }
  }
}``
>
> </details>
>
> The holdout V2 cell ran, but the result surfaced a bigger issue than detector quality: the holdout sessionization collapsed many labeled processes into only a few sessions. I’m checking the graph builder parameters now so we can tighten that session construction instead of trusting a misleading “5 red-team sessions” evaluation.
>
> <details><summary>Ran 2 commands</summary>
>
> - Ran `Get-Content -Path 'hetero_graph_builder.py' -TotalCount 280`
> - Ran `Get-Content -Path 'hetero_graph_builder.py' -TotalCount 620 | Select-Object -Last 260`
>
> </details>
>
> I’m going to add one more section that directly addresses what the holdout run just revealed: sessionization sensitivity. It will compare the current artifact-based sessions against stricter sessionization configs, because the current holdout test graph has a giant 5,623-process session and only 5 red-team sessions, which makes the session-level metric too compressed to trust.
>
> <details><summary>Ran a command</summary>
>
> - Ran `$script = @'
>   import json
>   from pathlib import Path
>
> path = Path('my_sessionization.ipynb')
> nb = json.loads(path.read_text(encoding='utf-8'))
>
> markdown_source = [
> > "## Holdout Sessionization Sensitivity Check\n",
> > "\n",
> > "The first holdout V2 run exposed a giant session component. This section compares stricter sessionization choices and reruns the holdout session detector on those alternatives.\n",
> > ]
>
> code_source = r'''HOLDOUT_SESSIONIZATION_STRICT_CONFIGS = [
> > {
> > "name": "artifact_deg2_strict",
> > "builder_kwargs": {
> > "add_file_edges": True,
> > "rare_file_min_degree": 2,
> > "rare_file_max_degree": 2,
> > },
> > },
> > {
> > "name": "parent_child_only",
> > "builder_kwargs": {
> > "add_file_edges": False,
> > },
> > },
> > ]
>
> HOLDOUT_SESSIONIZATION_SCORE_COLUMNS = [
> > "knn_train_distance_score",
> > "isolation_forest_train_score",
> > "rank_ensemble_train_score",
> > ]
>
> def required_holdout_session_vars():
> required = [
> > "holdout_train_fit",
> > "holdout_test_sample",
> > "holdout_text_builder",
> > "holdout_session_v2_enrichment",
> > "holdout_session_v2_table",
> > "HeterogeneousGeometricGraphBuilder",
> > "make_process_meta_for_graph",
> > "make_session_metadata",
> > "graph_process_ids",
> > "sparse_mean_pool_by_session",
> > "fit_text_session_svd",
> > "build_holdout_session_score_table",
> > "holdout_session_enrichment",
> > "holdout_session_host_dominance",
> > ]
> missing = [name for name in required if name not in globals()]
> if missing:
> raise RuntimeError(f"Run the holdout setup and Holdout Session Detector V2 cells first. Missing: {missing}")
>
> def build*session_graph_for_config(sample_df, config_name, builder_kwargs):
> kwargs = {
> "rare_file_max_degree": 10,
> "add_same_user_edges": False,
> "add_network_edges": False,
> }
> kwargs.update(builder_kwargs)
> graph_builder = HeterogeneousGeometricGraphBuilder(\*\*kwargs)
> graph_data = graph_builder.build_graph(sample_df)
> process_meta_for_graph = make_process_meta_for_graph(sample_df, graph_data)
> session_meta_for_graph = make_session_metadata(
> process_meta_for_graph,
> graph_data["process"].session_id.cpu(),
> )
> return graph_data, graph_builder.report*, process_meta_for_graph, session_meta_for_graph
>
> def session_quality_row(split_name, config_name, graph_report, process_meta_for_graph, session_meta_for_graph):
> process_count = int(graph_report["node_types"]["process"])
> process_red_rows = int(process_meta_for_graph["red_team"].fillna(0).astype(int).sum())
> process_bad_rows = int(process_meta_for_graph["bad_user_binary"].fillna(0).astype(int).sum())
> red_sessions = int(session_meta_for_graph["red_team"].fillna(0).astype(int).sum())
> bad_sessions = int(session_meta_for_graph["bad_user"].fillna(0).astype(int).sum())
> return {
> "split": split_name,
> "sessionization_config": config_name,
> "processes": process_count,
> "sessions": int(graph_report["n_sessions"]),
> "singletons": int(graph_report["singletons"]),
> "median_session_size": float(graph_report["median_session_size"]),
> "p95_session_size": float(graph_report["p95_session_size"]),
> "max_session_size": int(graph_report["max_session_size"]),
> "max_session_fraction": int(graph_report["max_session_size"]) / max(process_count, 1),
> "process_red_rows": process_red_rows,
> "red_sessions": red_sessions,
> "red_processes_per_session": process_red_rows / max(red_sessions, 1),
> "process_bad_user_rows": process_bad_rows,
> "bad_user_sessions": bad_sessions,
> "bad_user_processes_per_session": process_bad_rows / max(bad_sessions, 1),
> }
>
> def score_strict_holdout_config(config_name, builder_kwargs):
> train_graph, train_report, train_process_meta, train_session_meta = build_session_graph_for_config(
> holdout_train_fit,
> f"{config_name}\_train",
> builder_kwargs,
> )
> test_graph, test_report, test_process_meta, test_session_meta = build_session_graph_for_config(
> holdout_test_sample,
> f"{config_name}\_test",
> builder_kwargs,
> )
>
>     train_text_features = holdout_text_builder.transform(
>         holdout_train_fit,
>         process_ids=graph_process_ids(train_graph),
>     )
>     test_text_features = holdout_text_builder.transform(
>         holdout_test_sample,
>         process_ids=graph_process_ids(test_graph),
>     )
>
>     train_session_text = sparse_mean_pool_by_session(
>         train_text_features,
>         train_graph["process"].session_id.cpu(),
>     )
>     test_session_text = sparse_mean_pool_by_session(
>         test_text_features,
>         test_graph["process"].session_id.cpu(),
>     )
>
>     train_embedding, test_embedding, svd_model = fit_text_session_svd(train_session_text, test_session_text)
>     score_table, isolation_model = build_holdout_session_score_table(
>         test_session_meta,
>         train_embedding,
>         test_embedding,
>         f"holdout_session_text_v2_{config_name}",
>     )
>     score_table["sessionization_config"] = config_name
>
>     quality_rows = [
>         session_quality_row("train", config_name, train_report, train_process_meta, train_session_meta),
>         session_quality_row("test", config_name, test_report, test_process_meta, test_session_meta),
>     ]
>
>     enrichment_rows = []
>     for score_col in HOLDOUT_SESSIONIZATION_SCORE_COLUMNS:
>         for label_col in ["red_team", "bad_user"]:
>             for row in holdout_session_enrichment(score_table, score_col, label_col):
>                 row["sessionization_config"] = config_name
>                 enrichment_rows.append(row)
>
>     host_rows = []
>     for score_col in HOLDOUT_SESSIONIZATION_SCORE_COLUMNS:
>         for label_col in ["red_team", "bad_user"]:
>             for frac in [0.01, 0.05, 0.10]:
>                 row = holdout_session_host_dominance(score_table, score_col, label_col, frac)
>                 row["sessionization_config"] = config_name
>                 host_rows.append(row)
>
>     return {
>         "score_table": score_table,
>         "quality_rows": quality_rows,
>         "enrichment_rows": enrichment_rows,
>         "host_rows": host_rows,
>         "svd_model": svd_model,
>         "isolation_model": isolation_model,
>     }
>
> required_holdout_session_vars()
>
> current_quality_rows = []
> if "holdout_train_graph_report" in globals() and "holdout_test_graph_report" in globals():
> current_quality_rows.append(
> session_quality_row(
> "train",
> "artifact_deg10_current",
> holdout_train_graph_report,
> holdout_train_process_meta,
> holdout_train_session_meta,
> )
> )
> current_quality_rows.append(
> session_quality_row(
> "test",
> "artifact_deg10_current",
> holdout_test_graph_report,
> holdout_test_process_meta,
> holdout_test_session_meta,
> )
> )
>
> strict_results = {}
> for config in HOLDOUT_SESSIONIZATION_STRICT_CONFIGS:
> print(f"Scoring holdout config: {config['name']}")
> strict_results[config["name"]] = score_strict_holdout_config(
> config["name"],
> config["builder_kwargs"],
> )
>
> holdout_sessionization_quality = pd.DataFrame(
> current_quality_rows + [
> row
> for result in strict_results.values()
> for row in result["quality_rows"]
> ]
> )
>
> display(
> holdout_sessionization_quality.sort_values(
> ["split", "max_session_fraction", "red_processes_per_session"],
> ascending=[True, True, True],
> ).reset_index(drop=True)
> )
>
> current_enrichment = holdout_session_v2_enrichment.copy()
> current_enrichment["sessionization_config"] = "artifact_deg10_current"
>
> strict_enrichment = pd.DataFrame([
> row
> for result in strict_results.values()
> for row in result["enrichment_rows"]
> ])
> holdout_sessionization_enrichment = pd.concat(
> [current_enrichment, strict_enrichment],
> ignore_index=True,
> ).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(holdout_sessionization_enrichment)
>
> current_host = holdout_session_v2_host_dominance.copy()
> current_host["sessionization_config"] = "artifact_deg10_current"
>
> strict_host = pd.DataFrame([
> row
> for result in strict_results.values()
> for row in result["host_rows"]
> ])
> holdout_sessionization_host_dominance = pd.concat(
> [current_host, strict_host],
> ignore_index=True,
> ).sort_values(
> ["label", "top_fraction", "lift_vs_baseline", "top_positive_rate"],
> ascending=[True, True, False, False],
> ).reset_index(drop=True)
>
> display(holdout_sessionization_host_dominance)
>
> best_rows = holdout_sessionization_enrichment[
> (holdout_sessionization_enrichment["label"] == "red_team") &
> (holdout_sessionization_enrichment["top_fraction"] == 0.01)
> ].sort_values("lift_vs_baseline", ascending=False)
>
> print("Best holdout top-1% red_team by sessionization config:")
> display(best_rows.head(10))
>
> holdout_strict_score_tables = {
> name: result["score_table"]
> for name, result in strict_results.items()
> }
> '''.splitlines(keepends=True)
>
> markdown_cell = {"cell_type": "markdown", "metadata": {}, "source": markdown_source}
> code_cell = {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code_source}
>
> idx = 0
> while idx < len(nb["cells"]):
> cell = nb["cells"][idx]
> src = ''.join(cell.get("source", []))
> if cell.get("cell_type") == "markdown" and src.startswith("## Holdout Sessionization Sensitivity Check"):
> del nb["cells"][idx:idx + 2]
> continue
> idx += 1
>
> insert_at = None
> for idx, cell in enumerate(nb["cells"]):
> src = ''.join(cell.get("source", []))
> if src.startswith("from sklearn.neighbors import NearestNeighbors") and idx > 55:
> insert_at = idx
> break
> if insert_at is None:
> for idx, cell in enumerate(nb["cells"]):
> src = ''.join(cell.get("source", []))
> if src.startswith("## Holdout Semantic Text Neighbor Inspection"):
> insert_at = idx
> break
> if insert_at is None:
> insert_at = len(nb["cells"])
>
> nb["cells"][insert_at:insert_at] = [markdown_cell, code_cell]
> path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding='utf-8')
> print(f"Inserted Holdout Sessionization Sensitivity Check at cell index {insert_at}.")
> '@
> $script | py -3 -`
>
> </details>
>
> The sensitivity section is inserted. I’m validating it now; it reuses the existing holdout setup and will compare the current `rare_file_max_degree=10` sessionization against stricter artifact-degree-2 and parent-child-only alternatives.
>
> <details><summary>Ran 3 commands</summary>
>
> - Ran `$script = @'
import ast
import json
from pathlib import Path
nb = json.loads(Path('my_sessionization.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell.get('source', []))
    if src.startswith('HOLDOUT_SESSIONIZATION_STRICT_CONFIGS'):
        ast.parse(src)
        print(f'compiled holdout sessionization sensitivity cell at index {i}, lines={len(src.splitlines())}, exec={cell.get("execution_count")}, outputs={len(cell.get("outputs", []))}')
        break
else:
    raise SystemExit('holdout sessionization sensitivity cell not found')
'@
$script | py -3 -`
> - Ran `rg -n "Holdout Sessionization Sensitivity Check|HOLDOUT_SESSIONIZATION|holdout_sessionization_enrichment|holdout_strict_score_tables" my_sessionization.ipynb`
> - Ran ``$nb = Get-Content -Raw -Path 'my_sessionization.ipynb' | ConvertFrom-Json
for ($i = 52; $i -le 61; $i++) {
  $cell = $nb.cells[$i]
  $src = ($cell.source -join '')
  $first = (($src -split "`n") | Where-Object { $_.Trim().Length -gt 0 } | Select-Object -First 1)
  if ($null -eq $first) { $first = '' }
  Write-Output ("CELL {0} type={1} exec={2} outputs={3} first={4}" -f $i, $cell.cell_type, $cell.execution_count, @($cell.outputs).Count, $first.Substring(0, [Math]::Min(170, $first.Length)))
}``
>
> </details>

</details>

The holdout V2 ran, and it exposed the thing we needed to catch:

```text
holdout test-score:
processes = 11,295
sessions = 5,004
max_session = 5,623
red_team sessions = 5
bad_user sessions = 1
```

That means the current holdout sessionization is collapsing a huge chunk of the test set into one giant component. So the holdout metric is technically strong, but not trustworthy yet:

```text
top 1% red_team: 5 / 5 sessions found
```

The issue is that `1,188` red-team process rows became only `5` red-team sessions. Same problem for bad-user: `970` bad-user process rows became only `1` bad-user session. That is too compressed to use as the main session-level validation.

I added the next fix section: [my_sessionization.ipynb](/C:/Users/mekae/Desktop/CS/Masters/Tutte/Cyber-Telemetry/my_sessionization.ipynb:27384) **Holdout Sessionization Sensitivity Check**.

It compares:

```text
artifact_deg10_current
artifact_deg2_strict
parent_child_only
```

and reruns the holdout session detector for the stricter options. The main table to watch is `holdout_sessionization_quality`, especially:

```text
max_session_fraction
red_sessions
red_processes_per_session
bad_user_sessions
bad_user_processes_per_session
```

Run cell 57 next. The goal is to find a sessionization setting that avoids the giant component before we trust the V2 holdout detector numbers.
