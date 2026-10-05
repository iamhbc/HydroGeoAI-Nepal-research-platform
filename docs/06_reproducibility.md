# 6. Reproducibility

## Environment
* `pyproject.toml` (pip extras: `api, geo, gbm, explain, hf, io, db, dev, all`) or `environment.yml` (conda).
* `Dockerfile` (CPU torch) and `docker-compose.yml` (PostGIS + API + web).
* Tested locally on macOS (Apple silicon) with Python 3.14, torch 2.14, numpy 2.4, pandas 3.0, scikit-learn 1.9,
  Node 24 / Next.js 16. CI workflow (`.github/workflows/ci.yml`) targets Python 3.11 and runs once pushed.

## One command
```bash
make reproduce-main-results
```
This (1) builds the dataset if missing, (2) runs `configs/experiments/main.yaml` for all splits, (3) writes
`results/<experiment_id>/{tables,figures,metrics,models,supplementary}` plus `SUMMARY.md` and
`supplementary/environment.json`, and (4) registers the experiment, runs and model in the registry and deploys the
model.

`make reproduce-quick` runs the same pipeline with a small budget to check that everything works.

## What is recorded per experiment
Experiment ID, dataset version (content hash), code version (git commit, or `nogit-<source hash>` before git is
initialised), full config, hyper-parameters, train/val/test periods, geographic split (station lists), seed,
metrics per run, artifacts, timestamps. Each API prediction is logged with model version, dataset version and
parameters (`GET /predictions/log`).

## Determinism
Seeds are set for Python, NumPy and PyTorch. CPU is the default device; set `HYDROGEOAI_USE_ACCELERATOR=1` to use
CUDA/MPS (faster, not bit-reproducible). Bootstrap and permutation tests use fixed seeds.

## Platform notes
* **macOS + conda**: PyTorch and scikit-learn can load different OpenMP runtimes, which segfaults. The package
  imports torch first (`hydrogeoai/__init__.py`) and runs scikit-learn under `utils.sklearn_threads()`. If you write
  your own scripts, `import hydrogeoai` (or `torch`) before pandas/scikit-learn.
* **MapLibre v6** loads its worker from `public/maplibre/` (copied by `npm install` via `postinstall`).

## Publishing later
Nothing is pushed or uploaded automatically. When the work is finished and tested:
```bash
git init && git add -A && git commit -m "HydroGeoAI-Nepal: initial research platform"
# create an empty repository on GitHub, then:
git remote add origin git@github.com:<you>/HydroGeoAI-Nepal.git && git push -u origin main
hydrogeoai hf export                                   # stage dataset/model/Space under hf/export/
hydrogeoai hf publish --namespace <hf-user>            # dry run: shows what would be uploaded
HF_TOKEN=... hydrogeoai hf publish --namespace <hf-user> --push
```
Before publishing real data, check the providers' redistribution terms. The `.gitignore` excludes data, results,
registries and weights. Release those via Hugging Face (or Zenodo for a DOI).
