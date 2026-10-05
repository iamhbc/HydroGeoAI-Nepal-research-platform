"""Reproduce the main results of HydroGeoAI-Nepal.

    make reproduce-main-results            # or
    python scripts/reproduce_main_results.py --config configs/experiments/main.yaml

Steps: (1) ensure the versioned dataset exists, (2) record environment, (3) run every split mode with
baselines, deep models, SSL pretraining, ablation A-F, uncertainty, failure analysis, explainability,
robustness and hypothesis assessment, (4) write tables/figures/metrics/models/supplementary and register
the experiment + model. Outputs: results/<experiment_id>/ (see SUMMARY.md there).
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import hydrogeoai  # noqa: E402  (imports torch first; see package docstring)
from hydrogeoai.data.pipeline import build_dataset  # noqa: E402
from hydrogeoai.experiments.runner import run  # noqa: E402


def environment() -> dict:
    import numpy
    import pandas
    import sklearn
    import torch
    return {"python": sys.version, "platform": platform.platform(), "hydrogeoai": hydrogeoai.__version__,
            "numpy": numpy.__version__, "pandas": pandas.__version__, "sklearn": sklearn.__version__,
            "torch": torch.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/main.yaml")
    ap.add_argument("--modes", nargs="*")
    ap.add_argument("--rebuild-data", action="store_true")
    a = ap.parse_args()
    if a.rebuild_data or not (ROOT / "data" / "processed" / "LATEST").exists():
        build_dataset(force=a.rebuild_data)
    res = run(a.config, a.modes)
    out = Path(res["results_dir"])
    (out / "supplementary" / "environment.json").write_text(json.dumps(environment(), indent=2))
    print(json.dumps(res, indent=2))
    print(f"\nSummary: {out / 'SUMMARY.md'}")


if __name__ == "__main__":
    main()
