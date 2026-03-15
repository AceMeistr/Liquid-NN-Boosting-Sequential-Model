from __future__ import annotations

import copy
import json
import logging
import random
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generator, TypeVar

import numpy as np
import torch
import torch.nn as nn


_LOGGING_INITIALIZED = False
T = TypeVar("T")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    if json_format:
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":"%(message)s"}'
    else:
        fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        stream=sys.stdout,
    )
    _LOGGING_INITIALIZED = True


# ---------------------------------------------------------------------------
# Device management
# ---------------------------------------------------------------------------

def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        dev = torch.device("cuda")
        logger.info(
            "CUDA device selected: %s (%.0f MB free)",
            torch.cuda.get_device_name(dev),
            (torch.cuda.get_device_properties(dev).total_memory - torch.cuda.memory_allocated(dev)) / 1024**2,
        )
        return dev
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        logger.info("MPS (Apple Silicon) device selected")
        return torch.device("mps")
    logger.info("No GPU detected - falling back to CPU")
    return torch.device("cpu")


def get_device_info(device: torch.device) -> dict[str, Any]:
    info: dict[str, Any] = {"device": str(device)}
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
        info["cuda_version"] = torch.version.cuda or "N/A"
        props = torch.cuda.get_device_properties(device)
        info["gpu_total_memory_mb"] = round(props.total_memory / 1024**2, 2)
        info["memory_allocated_mb"] = round(torch.cuda.memory_allocated(device) / 1024**2, 2)
        info["memory_reserved_mb"] = round(torch.cuda.memory_reserved(device) / 1024**2, 2)
    return info


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def enable_performance_mode() -> None:
    """Enable maximum GPU performance by allowing non-deterministic ops and benchmarking."""
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("GPU performance mode enabled - using fastest kernels")


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)
    tmp.replace(path)  # atomic on same filesystem


def read_json(path: Path, missing_ok: bool = False) -> dict:
    if missing_ok and not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fp:
        return json.load(fp)


def retry(operation: Callable[[], T], retries: int = 3, delay_seconds: float = 0.15) -> T:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            return operation()
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                break
            time.sleep(delay_seconds * attempt)
    raise RuntimeError(f"Operation failed after {retries} retries") from last_error


def safe_torch_load(path: Path, map_location: torch.device) -> dict:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------------
# EMA (Exponential Moving Average) of model weights for stable inference
# ---------------------------------------------------------------------------

class ModelEMA:
    """Maintains an exponential moving average of model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        self.decay = decay
        self.shadow: dict[str, torch.Tensor] = {}
        self.backup: dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup.clear()

    @contextmanager
    def average_parameters(self, model: nn.Module) -> Generator[None, None, None]:
        self.apply(model)
        try:
            yield
        finally:
            self.restore(model)


# ---------------------------------------------------------------------------
# Learning rate schedulers
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    epochs: int,
    warmup_epochs: int = 0,
    **kwargs: Any,
) -> torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau:
    """Build LR scheduler with optional linear warmup."""
    if scheduler_type == "plateau":
        base = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )
    elif scheduler_type == "cosine":
        base = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(epochs - warmup_epochs, 1), eta_min=1e-6,
        )
    elif scheduler_type == "cosine_warm":
        base = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=max((epochs - warmup_epochs) // 3, 1), T_mult=2, eta_min=1e-6,
        )
    else:
        base = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=3,
        )


    if warmup_epochs > 0 and scheduler_type != "plateau":
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, total_iters=warmup_epochs,
        )
        return torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, base], milestones=[warmup_epochs],
        )

    return base

# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def is_cuda_oom(exc: Exception) -> bool:
    """Check if an exception is a CUDA OutOfMemoryError."""
    err_str = str(exc)
    return "CUDA out of memory" in err_str or "OutOfMemoryError" in err_str

def build_sqlite_storage_url(db_path: str) -> str:
    """Convert a file path to a proper sqlite:/// URL."""
    p = Path(db_path).resolve()
    # Windows paths need extra slashes for SQLAlchemy
    return f"sqlite:///{p.as_posix()}"
