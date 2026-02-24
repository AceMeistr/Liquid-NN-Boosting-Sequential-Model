from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

from modelmk1.common.runtime import retry


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AppPaths:
    root: Path
    training_data: Path
    outputs: Path


def get_app_paths() -> AppPaths:
    root = Path(__file__).resolve().parents[3]
    training_data = root / "Training Data"
    outputs = root / "outputs"
    training_data.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
    return AppPaths(root=root, training_data=training_data, outputs=outputs)


def resolve_data_file(data_path: str | None = None) -> Path:
    paths = get_app_paths()
    if data_path:
        candidate = Path(data_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
        raise FileNotFoundError(f"Provided dataset does not exist: {candidate}")

    discovered = [*paths.training_data.glob("*.parquet"), *paths.training_data.glob("*.csv")]
    candidates: list[tuple[float, Path]] = []
    for item in discovered:
        try:
            mtime = retry(lambda: item.stat().st_mtime, retries=2, delay_seconds=0.05)
            candidates.append((mtime, item))
        except FileNotFoundError:
            logger.warning("Skipped dataset candidate removed during scan: %s", item)

    candidates = [item for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True)]
    if not candidates:
        raise FileNotFoundError(
            "No dataset found in 'Training Data'. Add CSV/Parquet and rerun."
        )
    return candidates[0]
