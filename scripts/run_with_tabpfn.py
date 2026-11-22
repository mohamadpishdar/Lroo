"""
Run the full LROO pipeline with Real-TabPFN included.

The Prior Labs access key is read from the TABPFN_TOKEN environment variable
and is never written to disk or to the result files.

Usage
    export TABPFN_TOKEN=...
    python scripts/run_with_tabpfn.py data/TM-RugPull_v2.xlsx results
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import lroo_pipeline as L  # noqa: E402
import tabpfn_client  # noqa: E402
from tabpfn_client import TabPFNClassifier, set_access_token  # noqa: E402

MODEL = "v2.5_real"   # Real-TabPFN checkpoint used in the paper


def tabpfn_factory():
    return TabPFNClassifier(model_path=MODEL, random_state=L.SEED)


if __name__ == "__main__":
    token = os.environ.get("TABPFN_TOKEN")
    if not token:
        sys.exit("Set the TABPFN_TOKEN environment variable first.")
    set_access_token(token)
    data = sys.argv[1] if len(sys.argv) > 1 else "data/TM-RugPull_v2.xlsx"
    out = sys.argv[2] if len(sys.argv) > 2 else "results"
    start = time.time()
    L.main(data, out_dir=out, tabpfn_factory=tabpfn_factory, tune=True, permutation_for_tabpfn=False)  # as in the paper: quota
    info = {"tabpfn_client_version": tabpfn_client.__version__, "model": MODEL,
            "params": {k: str(v) for k, v in tabpfn_factory().get_params().items()},
            "runtime_s": round(time.time() - start)}
    with open(os.path.join(out, "tabpfn_info.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))
