from __future__ import annotations

import json
import logging
import random
import time
from pathlib import Path
from typing import Callable, TypeVar

import numpy as np
import torch


_LOGGING_INITIALIZED = False
T = TypeVar("T")


def configure_logging(level: str = "INFO") -> None:
    global _LOGGING_INITIALIZED
    if _LOGGING_INITIALIZED:
        return
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    _LOGGING_INITIALIZED = True


def pick_device(force_cpu: bool = False) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)


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
