"""Stage and (manually) publish Hugging Face artifacts.

Nothing is ever uploaded automatically. `export()` builds three local folders under hf/export/:
    hydrogeoai-nepal-dataset/  (README.md dataset card + parquet/geojson/metadata files)
    hydrogeoai-nepal-model/    (README.md model card + config + weights + encoder)
    hydrogeoai-nepal-demo/     (Gradio Space app + requirements + a small demo data slice + model)
`publish(..., push=True)` uploads them with huggingface_hub; it needs a login (`hf auth login`) or HF_TOKEN.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import pandas as pd

from ..config import resolve
from ..data.pipeline import latest_dataset_dir
from ..experiments.registry import Registry
from .cards import dataset_card

EXPORT = resolve("hf/export")
DATASET_FILES = ["observations.parquet", "stations.parquet", "thresholds.parquet", "qc_flags.parquet",
                 "homogeneity.csv", "trends.csv", "etccdi_annual.parquet", "whiplash_events.parquet",
                 "whiplash_annual.csv", "basins.geojson", "rivers.geojson", "outline.geojson", "metadata.json",
                 "provenance.json", "qc_summary.json", "station_issues.csv", "dem.npz", "dem.json"]


def export(namespace: str | None = None) -> dict:
    namespace = namespace or os.environ.get("HF_NAMESPACE", "your-hf-username")
    ds = latest_dataset_dir()
    out = {}
    d_out = EXPORT / "hydrogeoai-nepal-dataset"
    shutil.rmtree(d_out, ignore_errors=True)
    d_out.mkdir(parents=True)
    for f in DATASET_FILES:
        if (ds / f).exists():
            shutil.copy(ds / f, d_out / f)
    (d_out / "README.md").write_text(dataset_card(ds, namespace))
    out["dataset"] = str(d_out)

    rec = Registry().deployed_model("hydrogeoai-nepal-model")
    if rec:
        m_out = EXPORT / "hydrogeoai-nepal-model"
        shutil.rmtree(m_out, ignore_errors=True)
        shutil.copytree(rec["path"], m_out)
        out["model"] = str(m_out)
        s_out = EXPORT / "hydrogeoai-nepal-demo"
        shutil.rmtree(s_out, ignore_errors=True)
        shutil.copytree(resolve("hf/space"), s_out)
        shutil.copytree(rec["path"], s_out / "model")
        demo = s_out / "demo_data"
        demo.mkdir()
        obs = pd.read_parquet(ds / "observations.parquet")
        cols = ["station_id", "date", "precip_qc", "tmax_qc", "tmin_qc", "extreme_wet_day", "spi30"]
        obs[obs.date >= "2019-01-01"][cols].to_parquet(demo / "observations.parquet", index=False)
        shutil.copy(ds / "stations.parquet", demo / "stations.parquet")
        # bundle the package source so the Space runs the identical inference code
        shutil.copytree(resolve("src/hydrogeoai"), s_out / "hydrogeoai",
                        ignore=shutil.ignore_patterns("__pycache__", "api"))
        out["space"] = str(s_out)
    return out


def whoami() -> dict | None:
    """Return the logged-in Hugging Face account (HF_TOKEN env var or `hf auth login`), or None."""
    try:
        from huggingface_hub import HfApi
        return HfApi(token=os.environ.get("HF_TOKEN") or None).whoami()
    except Exception:
        return None


def publish(namespace: str | None = None, push: bool = False, private: bool = False) -> dict:
    staged = {k: EXPORT / v for k, v in (("dataset", "hydrogeoai-nepal-dataset"), ("model", "hydrogeoai-nepal-model"),
                                          ("space", "hydrogeoai-nepal-demo"))}
    missing = [k for k, p in staged.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Run `hydrogeoai hf export` first (missing: {missing})")
    me = whoami()
    namespace = namespace or (me or {}).get("name") or os.environ.get("HF_NAMESPACE", "your-hf-username")
    plan = {k: {"repo_id": f"{namespace}/{p.name}", "repo_type": {"dataset": "dataset", "model": "model",
                                                                  "space": "space"}[k],
                "folder": str(p), "files": sum(1 for _ in p.rglob("*") if _.is_file()),
                "size_mb": round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 1)}
            for k, p in staged.items()}
    login = {"logged_in_as": (me or {}).get("name")}
    if not push:
        return {"dry_run": True, **login, "plan": plan,
                "note": "Nothing uploaded. Log in with `hf auth login`, then re-run with --push."}
    if me is None:
        raise RuntimeError("Not logged in to Hugging Face: run `hf auth login` (or set HF_TOKEN)")
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    urls = {}
    for k, info in plan.items():
        kw = {"space_sdk": "gradio"} if k == "space" else {}
        api.create_repo(info["repo_id"], repo_type=info["repo_type"], private=private, exist_ok=True, **kw)
        api.upload_folder(folder_path=info["folder"], repo_id=info["repo_id"], repo_type=info["repo_type"],
                          commit_message="HydroGeoAI-Nepal release")
        prefix = {"dataset": "datasets/", "space": "spaces/", "model": ""}[k]
        urls[k] = f"https://huggingface.co/{prefix}{info['repo_id']}"
    return {"dry_run": False, **login, "uploaded": plan, "urls": urls}
