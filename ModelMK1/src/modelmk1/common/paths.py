from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


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

    candidates = sorted(
        [
            *paths.training_data.glob("*.parquet"),
            *paths.training_data.glob("*.csv"),
        ],
        key=lambda item: item.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(
            "No dataset found in 'Training Data'. Add CSV/Parquet and rerun."
        )
    return candidates[0]
