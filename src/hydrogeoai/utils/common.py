"""Shared utilities: logging, seeding, hashing, device selection, code versioning."""
from __future__ import annotations

import datetime as dt
import hashlib
import logging
import os
import random
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

_LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def get_logger(name: str = "hydrogeoai") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logging.getLogger().handlers and not logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(_LOG_FORMAT, "%H:%M:%S"))
        logger.addHandler(h)
        logger.setLevel(os.environ.get("HYDROGEOAI_LOG_LEVEL", "INFO"))
        logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(False)
    except ImportError:  # pragma: no cover
        pass


def get_device(pref: str = "auto"):
    import torch

    if pref != "auto":
        return torch.device(pref)
    # MPS is fast but not bit-reproducible for every op; CPU is the reproducible default.
    if os.environ.get("HYDROGEOAI_USE_ACCELERATOR") == "1":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device("cpu")


def utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def file_hash(path: str | os.PathLike, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataframe_hash(df: pd.DataFrame) -> str:
    """Deterministic content hash of a DataFrame (values + columns + index)."""
    h = hashlib.sha256()
    h.update(",".join(map(str, df.columns)).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()


def code_version(root: str | os.PathLike | None = None) -> str:
    """Git commit if available, otherwise a content hash of the source tree (`nogit-<hash>`)."""
    root = Path(root) if root else Path(__file__).resolve().parents[3]
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True, timeout=5
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        h = hashlib.sha256()
        for p in sorted((root / "src").rglob("*.py")):
            h.update(p.read_bytes())
        return "nogit-" + h.hexdigest()[:12]


def sklearn_threads():
    """Context manager limiting OpenMP to one thread for scikit-learn calls.

    On macOS, PyTorch and scikit-learn can load different OpenMP runtimes; running multithreaded
    HistGradientBoosting after torch has initialised its runtime segfaults. One OpenMP thread inside
    sklearn calls avoids this while torch keeps its own intra-op threads.
    """
    from contextlib import nullcontext

    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1, user_api="openmp")
    except ImportError:  # pragma: no cover
        return nullcontext()
