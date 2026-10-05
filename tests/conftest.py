import os
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import hydrogeoai  # noqa: E402,F401  (torch first)


@pytest.fixture(scope="session")
def small_synthetic():
    from hydrogeoai.data import synthetic
    return synthetic.generate({"n_stations": 10, "start": "2000-01-01", "end": "2012-12-31", "seed": 7,
                               "inject_inhomogeneities": 2})


@pytest.fixture(scope="session")
def workspace(tmp_path_factory):
    """Isolated data/results/registry directories with a small processed dataset."""
    tmp = tmp_path_factory.mktemp("hgai")
    os.environ["HYDROGEOAI_DATA_DIR"] = str(tmp / "data")
    os.environ["HYDROGEOAI_RESULTS_DIR"] = str(tmp / "results")
    os.environ["HYDROGEOAI_REGISTRY_URL"] = f"sqlite:///{tmp / 'registry.db'}"
    data_cfg = yaml.safe_load((ROOT / "configs/data/default.yaml").read_text())
    data_cfg["synthetic"].update({"n_stations": 12, "start": "2000-01-01", "end": "2012-12-31"})
    data_cfg["reference_period"] = ["2000-01-01", "2007-12-31"]
    (tmp / "data.yaml").write_text(yaml.safe_dump(data_cfg))
    exp = {
        "inherits": str(ROOT / "configs/experiments/quick.yaml"), "name": "pytest",
        "data_config": str(tmp / "data.yaml"),
        "splits": {"temporal": {"train": ["2000-01-01", "2007-12-31"], "val": ["2008-01-01", "2009-12-31"],
                                "test": ["2010-01-01", "2012-12-31"], "gap_days": 30},
                   "spatial": {"strategy": "basin", "test_groups": ["Koshi"], "val_fraction_of_train_stations": 0.2},
                   "modes": ["temporal"]},
        "max_train_samples": 1500, "max_eval_samples": 2000, "window_days": 30,
        "baselines": ["climatology", "persistence", "logistic", "hist_gbm"], "deep_models": ["lstm"],
        "deep": {"epochs": 1, "patience": 1, "batch_size": 128}, "pretrain": {"epochs": 1, "max_samples": 1000},
        "uncertainty": {"ensemble_members": 2, "mc_dropout_samples": 2}, "bootstrap": {"n": 10},
        "robustness": {"missing_rates": [0.0, 0.3]}, "label_fractions": [0.5, 1.0],
    }
    (tmp / "exp.yaml").write_text(yaml.safe_dump(exp))
    from hydrogeoai.data.pipeline import build_dataset
    ds = build_dataset(tmp / "data.yaml")
    return {"tmp": tmp, "dataset_dir": ds, "exp_config": tmp / "exp.yaml"}
