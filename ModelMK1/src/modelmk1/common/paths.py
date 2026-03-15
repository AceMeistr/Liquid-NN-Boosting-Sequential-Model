from __future__ import annotations

from dataclasses import dataclass
import logging


logger = logging.getLogger(__name__)

<<<<<<< HEAD
=======
# ---------------------------------------------------------------------------
# DATA ENGINE location (sibling project in ~/Downloads/MFT)
# ---------------------------------------------------------------------------
_DATA_ENGINE_ROOT = Path("C:/Users/sasan/Downloads/MFT/DATA ENGINE")
_DATA_ENGINE_TRAINING = _DATA_ENGINE_ROOT / "TRAINING DATA"

>>>>>>> main

@dataclass(frozen=True)
class AppPaths:
    root: Path
    training_data: Path
    outputs: Path
<<<<<<< HEAD
=======
    data_engine_root: Path
    data_engine_raw: Path
    data_engine_features: Path
    data_engine_clean: Path
>>>>>>> main


def get_app_paths() -> AppPaths:
    root = Path(__file__).resolve().parents[3]
    training_data = root / "Training Data"
    outputs = root / "outputs"
    training_data.mkdir(parents=True, exist_ok=True)
    outputs.mkdir(parents=True, exist_ok=True)
<<<<<<< HEAD
    return AppPaths(root=root, training_data=training_data, outputs=outputs)
=======

    de_root = _DATA_ENGINE_ROOT
    de_raw = _DATA_ENGINE_TRAINING / "raw"
    de_features = _DATA_ENGINE_TRAINING / "features"
    de_clean = _DATA_ENGINE_TRAINING / "clean"

    return AppPaths(
        root=root,
        training_data=training_data,
        outputs=outputs,
        data_engine_root=de_root,
        data_engine_raw=de_raw,
        data_engine_features=de_features,
        data_engine_clean=de_clean,
    )


def discover_data_engine_parquets(
    folder: Path,
    limit: int = 0,
) -> list[Path]:
    """Recursively find all .parquet files under *folder*, sorted by name.

    If *limit* > 0, return at most *limit* files (newest first by name).
    """
    if not folder.exists():
        logger.warning("DATA ENGINE folder not found: %s", folder)
        return []
    files = sorted(folder.rglob("*.parquet"))
    if limit > 0:
        files = files[-limit:]
    logger.info("Discovered %d parquet files in %s", len(files), folder)
    return files
>>>>>>> main


def resolve_data_file(data_path: str | None = None) -> Path:
    paths = get_app_paths()
    if data_path:
        candidate = Path(data_path).expanduser().resolve()
        if candidate.exists() and candidate.is_file():
            return candidate
<<<<<<< HEAD
        raise FileNotFoundError(f"Provided dataset does not exist: {candidate}")

=======
        if candidate.exists() and candidate.is_dir():
            # Allow pointing to a directory - return as marker for multi-file load
            return candidate
        raise FileNotFoundError(f"Provided dataset does not exist: {candidate}")

    # 1) Check local Training Data folder first
>>>>>>> main
    discovered = [*paths.training_data.glob("*.parquet"), *paths.training_data.glob("*.csv")]
    candidates: list[tuple[float, Path]] = []
    for item in discovered:
        try:
            mtime = retry(lambda: item.stat().st_mtime, retries=2, delay_seconds=0.05)
            candidates.append((mtime, item))
        except FileNotFoundError:
            logger.warning("Skipped dataset candidate removed during scan: %s", item)

<<<<<<< HEAD
    candidates = [item for _, item in sorted(candidates, key=lambda pair: pair[0], reverse=True)]
    if not candidates:
        raise FileNotFoundError(
            "No dataset found in 'Training Data'. Add CSV/Parquet and rerun."
        )
    return candidates[0]
=======
    if candidates:
        result = max(candidates, key=lambda pair: pair[0])[1]
        return result

    # 2) Fall back to DATA ENGINE raw data
    if paths.data_engine_raw.exists():
        logger.info("No local data found - using DATA ENGINE raw data at %s", paths.data_engine_raw)
        return paths.data_engine_raw  # return directory for multi-file loading

    raise FileNotFoundError(
        "No dataset found. Place CSV/Parquet in 'Training Data' or ensure DATA ENGINE is in ~/Downloads."
    )
>>>>>>> main
