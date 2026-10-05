# HydroGeoAI-Nepal — reproducible research workflow
# All targets run locally. Nothing is pushed to GitHub or Hugging Face automatically.

PY      ?= .venv/bin/python
PIP     ?= .venv/bin/pip
CLI      = $(PY) -m hydrogeoai.cli
export PYTHONWARNINGS = ignore

.PHONY: help setup setup-full data qc-report reproduce-main-results reproduce-quick experiment \
        api web web-install space test lint cards hf-export hf-publish-dry db-up db-load docker-up clean-results

help:            ## list targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-24s\033[0m %s\n",$$1,$$2}'

setup:           ## create .venv (reusing system torch/numpy) and install the package + API deps
	python3 -m venv --system-site-packages .venv
	$(PIP) install -q -e ".[api,dev]"

setup-full:      ## install every optional dependency (geo, gbm, explain, hf, io, db)
	$(PIP) install -q -e ".[all]"

data:            ## build the versioned dataset (synthetic by default): ingest -> QC -> homogeneity -> events -> features
	$(CLI) data build

data-force:      ## rebuild the dataset, overwriting the same version directory
	$(CLI) data build --force

reproduce-quick: data  ## fast end-to-end run of the full experiment pipeline, all splits (~1 h CPU)
	$(CLI) experiment run --config configs/experiments/quick.yaml

reproduce-main-results: data  ## the main experiment pipeline (all splits, ablation, uncertainty, XAI, H1-H6; ~2-4 h CPU)
	$(PY) scripts/reproduce_main_results.py --config configs/experiments/main.yaml

experiment:      ## run a single split: make experiment MODES=temporal CONFIG=configs/experiments/quick.yaml
	$(CLI) experiment run --config $(or $(CONFIG),configs/experiments/main.yaml) --modes $(or $(MODES),temporal)

api:             ## start the FastAPI backend on http://localhost:8000 (docs at /docs)
	$(PY) -m uvicorn hydrogeoai.api.main:app --reload --port 8000

web-install:     ## install frontend dependencies
	cd web && npm install

web:             ## start the Next.js research dashboard on http://localhost:3000
	cd web && npm run dev

space:           ## run the Hugging Face Space (Gradio) locally — needs `pip install gradio`
	$(PY) hf/space/app.py

test:            ## run the test suite
	$(PY) -m pytest

lint:
	$(PY) -m ruff check src tests || true

cards:           ## regenerate dataset card into hf/export (model card is written by each experiment)
	$(CLI) hf export

hf-export:       ## stage dataset/model/space folders under hf/export (no upload)
	$(CLI) hf export

hf-publish-dry:  ## show what WOULD be uploaded to Hugging Face (dry run)
	$(CLI) hf publish

db-up:           ## start PostGIS in Docker
	docker compose up -d db

db-load:         ## load the processed dataset into PostGIS (needs psycopg + geoalchemy2)
	$(PY) scripts/load_postgis.py

docker-up:       ## full stack in Docker: PostGIS + API + web
	docker compose up --build

clean-results:   ## delete experiment outputs (keeps data and registry)
	rm -rf results/*/ && touch results/.gitkeep

hf-login:        ## log in to Hugging Face (you type your own token)
	.venv/bin/hf auth login

hf-publish:      ## upload dataset, model and Space to Hugging Face (requires hf-login)
	$(CLI) hf publish --push
