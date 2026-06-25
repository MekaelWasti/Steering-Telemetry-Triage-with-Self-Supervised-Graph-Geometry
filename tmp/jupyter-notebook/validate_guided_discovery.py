import json
import os
import tempfile
from itertools import combinations
from pathlib import Path

os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(tempfile.gettempdir(), "numba_cache"),
)

import datamapplot
import numpy as np
import pandas as pd
import umap
from IPython.display import display
from scipy.spatial import cKDTree
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[2]
notebook = json.loads((ROOT / "pipeline_temporal_clean.ipynb").read_text(encoding="utf-8"))
state = np.load(
    ROOT / "tmp/jupyter-notebook/discovery_state.npz",
    allow_pickle=True,
)

seeds = (42, 43, 44)
real = state["real"]
shuffled = state["shuffled"]
labels = state["labels"].astype(str)
sessions = [list(map(int, session)) for session in state["sessions"]]

namespace = {
    "__name__": "__main__",
    "np": np,
    "pd": pd,
    "cKDTree": cKDTree,
    "combinations": combinations,
    "normalize": normalize,
    "display": display,
    "datamapplot": datamapplot,
    "Path": Path,
    "SEED": 42,
    "K": 15,
    "TOP_FRACTIONS": (0.01, 0.05, 0.10),
    "EXPERIMENT_SEEDS": seeds,
    "X_node_stats": state["node_stats"],
    "labels_eval": labels,
    "y_eval": labels,
    "sage_real_no_host_by_seed": {
        seed: real[index] for index, seed in enumerate(seeds)
    },
    "sage_shuffled_no_host_by_seed": {
        seed: shuffled[index] for index, seed in enumerate(seeds)
    },
    "sessions": sessions,
    "process_df": pd.DataFrame({
        "process_name": state["process_name"],
        "hostname": state["hostname"],
    }),
}

evaluation_cell = next(
    cell for cell in notebook["cells"] if cell.get("id") == "b8efad28"
)
exec("".join(evaluation_cell["source"]), namespace)

discovery_cell = next(
    cell for cell in notebook["cells"] if cell.get("id") == "guided-discovery-code"
)
exec("".join(discovery_cell["source"]), namespace)

X_viz = np.hstack([
    normalize(np.nan_to_num(namespace["sage_real_no_host_by_seed"][seed]))
    for seed in seeds
])
namespace["embedding_2d"] = umap.UMAP(
    n_components=2,
    n_neighbors=15,
    min_dist=0.05,
    random_state=42,
).fit_transform(X_viz)

map_cell = next(
    cell for cell in notebook["cells"] if cell.get("id") == "guided-discovery-datamap-code"
)
os.chdir(ROOT)
exec("".join(map_cell["source"]), namespace)

print("GUIDED_DISCOVERY_VALIDATION_COMPLETE")
