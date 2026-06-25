import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "pipeline_temporal_clean.ipynb"
STATE_PATH = ROOT / "tmp/jupyter-notebook/discovery_state.npz"


def run_notebook_prefix():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    namespace = {"__name__": "__main__"}
    for index, cell in enumerate(notebook["cells"]):
        if index > 19:
            break
        if cell.get("cell_type") != "code":
            continue
        print(f"\n--- executing cell {index} ---", flush=True)
        if index == 17:
            exec(
                """
real_edges = add_reverse_edges(base_edge_dict(graph_data))
real_edges_no_host = {
    et: ei for et, ei in real_edges.items()
    if "same_host_time_window" not in et[1]
}
x_dict = make_x_dict(graph_data)
sage_real_no_host_by_seed = {}
sage_shuffled_no_host_by_seed = {}
for seed in EXPERIMENT_SEEDS:
    shuffled_edges = add_reverse_edges(shuffled_base_edge_dict(graph_data, seed=seed))
    shuffled_edges_no_host = {
        et: ei for et, ei in shuffled_edges.items()
        if "same_host_time_window" not in et[1]
    }
    z_real = train_sage(
        f"sage_real_no_host seed={seed}",
        real_edges_no_host,
        x_dict,
        seed,
    )
    z_shuffled = train_sage(
        f"sage_shuffled_no_host seed={seed}",
        shuffled_edges_no_host,
        x_dict,
        seed,
    )
    sage_real_no_host_by_seed[seed] = session_embedding_from_z(z_real, sessions)
    sage_shuffled_no_host_by_seed[seed] = session_embedding_from_z(z_shuffled, sessions)
""",
                namespace,
            )
        else:
            exec("".join(cell.get("source", [])), namespace)
        if index == 3:
            namespace["ROWS"] = 100_000
    return namespace


ns = run_notebook_prefix()

real = np.stack(
    [ns["sage_real_no_host_by_seed"][seed] for seed in ns["EXPERIMENT_SEEDS"]],
    axis=0,
)
shuffled = np.stack(
    [ns["sage_shuffled_no_host_by_seed"][seed] for seed in ns["EXPERIMENT_SEEDS"]],
    axis=0,
)

np.savez_compressed(
    STATE_PATH,
    real=real,
    shuffled=shuffled,
    node_stats=ns["X_node_stats"],
    labels=ns["labels_eval"],
    sessions=np.asarray(ns["sessions"], dtype=object),
    process_name=ns["process_df"]["process_name"].fillna("unknown").astype(str).to_numpy(),
    hostname=ns["process_df"]["hostname"].fillna("unknown").astype(str).to_numpy(),
)
print(f"\nSaved {STATE_PATH}", flush=True)
