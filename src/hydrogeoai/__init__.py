"""HydroGeoAI-Nepal: multimodal geospatial-temporal AI for hydroclimatic extremes.

Package layout
--------------
data/         ingestion, provenance, dataset catalog, synthetic generator
qc/           quality control + climate homogeneity
features/     temporal, extreme-event, spatial features and climate indices (SPI, ETCCDI)
events/       event taxonomy labelling + hydroclimatic whiplash engine
gis/          CRS-aware rasters, terrain derivatives, spatial operations
models/       baselines, deep sequence models, Hydroclimatic Encoder (SSL), multimodal fusion
uncertainty/  ensembles, MC dropout, conformal prediction, calibration
explain/      permutation importance, integrated gradients, modality ablation
evaluation/   metrics, benchmark, failure analysis
experiments/  registry, split builders, experiment runner
reporting/    paper-ready tables and figures
hf/           Hugging Face dataset/model cards and (manual) publishing
api/          FastAPI backend
"""

__version__ = "0.1.0"

# Load PyTorch (and its OpenMP runtime) before pandas/pyarrow/scipy/scikit-learn. On macOS with
# conda builds, loading another OpenMP runtime first causes segfaults inside torch kernels.
# scikit-learn calls are additionally wrapped in `utils.sklearn_threads()`.
try:  # pragma: no cover - import side effect only
    import torch as _torch  # noqa: F401
except ImportError:  # pragma: no cover
    pass

ATTRIBUTION_DISCLAIMER = (
    "Feature attributions describe what the statistical model relied on; they are NOT evidence of "
    "physical causation. Physical interpretation requires process-based reasoning and independent evidence."
)
