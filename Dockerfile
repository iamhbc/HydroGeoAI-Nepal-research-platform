# HydroGeoAI-Nepal API / pipeline image (CPU)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -e ".[api,db,dev]"

COPY configs ./configs
COPY scripts ./scripts
COPY tests ./tests
COPY Makefile ./

EXPOSE 8000
# Build the dataset on first start if missing, then serve the API
CMD ["sh", "-c", "test -f data/processed/LATEST || python -m hydrogeoai.cli data build; uvicorn hydrogeoai.api.main:app --host 0.0.0.0 --port 8000"]
