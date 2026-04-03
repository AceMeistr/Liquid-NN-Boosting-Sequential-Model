from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.preprocessing import RobustScaler, StandardScaler
from typing import Iterator
from torch.utils.data import Dataset

from modelmk1.common.paths import discover_data_engine_parquets
from modelmk1.features.indicators import add_all_indicators


logger = logging.getLogger(__name__)


REQUIRED_COLUMNS = {"timestamp", "price", "high", "low", "volume"}

# Column mapping from DATA ENGINE raw schema to ModelMK1 schema
_DATA_ENGINE_COL_MAP = {
    "idx_close": "price",
    "idx_high": "high",
    "idx_low": "low",
    "idx_volume": "volume",
}


@dataclass
class DataBundle:
    sequences: np.ndarray
    xgb_features: np.ndarray
    targets: np.ndarray
    feature_columns: list[str]


@dataclass
class SplitBundle:
    """Pre-split data bundle for train/val with fitted scaler."""
    x_seq_train: np.ndarray
    x_seq_val: np.ndarray
    x_xgb_train: np.ndarray
    x_xgb_val: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    scaler: StandardScaler | RobustScaler
    feature_columns: list[str]


class TickDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


# ---------------------------------------------------------------------------
# Loading and validation
# ---------------------------------------------------------------------------

def _map_data_engine_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Map DATA ENGINE column names to ModelMK1 expected names."""
    rename = {}
    for src, dst in _DATA_ENGINE_COL_MAP.items():
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    return df


def load_data_engine_dir(directory: str | Path, limit: int = 0) -> pd.DataFrame:
    """Load and concatenate all parquet files from a DATA ENGINE directory.

    Handles column mapping (idx_close to price, etc.) and sorts by timestamp.
    *limit*: when > 0, use only the most recent *limit* files.
    """
    folder = Path(directory)
    files = discover_data_engine_parquets(folder, limit=limit)
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder}")

    frames: list[pd.DataFrame] = []
    for f in files:
        try:
            part = pd.read_parquet(f)
            part = _map_data_engine_columns(part)
            frames.append(part)
        except Exception as exc:
            logger.warning("Skipping unreadable file %s: %s", f.name, exc)

    if not frames:
        raise ValueError(f"All parquet files in {folder} were unreadable")

    df = pd.concat(frames, ignore_index=True)

    # Ensure timestamp column exists and is proper datetime
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    else:
        raise ValueError("DATA ENGINE files missing 'timestamp' column")

    # Cast numeric columns
    for col in ["price", "high", "low", "volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["timestamp", "price", "high", "low", "volume"])
    df = df[(df["price"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["volume"] >= 0)]
    df = df[df["high"] >= df["low"]]
    df = df.sort_values("timestamp").reset_index(drop=True)

    logger.info(
        "Loaded %d rows from %d DATA ENGINE files (%s to %s)",
        len(df), len(files),
        df["timestamp"].iloc[0].strftime("%Y-%m-%d") if len(df) > 0 else "?",
        df["timestamp"].iloc[-1].strftime("%Y-%m-%d") if len(df) > 0 else "?",
    )
    return df


def load_tick_df(path: str) -> pd.DataFrame:
    """Load tick data from a file *or* a DATA ENGINE directory."""
    p = Path(path)

    # If path is a directory, treat as DATA ENGINE multi-file source
    if p.is_dir():
        return load_data_engine_dir(p)

    source = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)

    # Auto-map DATA ENGINE columns if present
    source = _map_data_engine_columns(source)

    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = source.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    for column in ["price", "high", "low", "volume"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    start_rows = len(df)
    df = df.dropna(subset=["timestamp", "price", "high", "low", "volume"])
    df = df[(df["price"] > 0) & (df["high"] > 0) & (df["low"] > 0) & (df["volume"] >= 0)]
    df = df[df["high"] >= df["low"]]
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

    dropped = start_rows - len(df)
    if dropped > 0:
        logger.info("Dropped %s invalid rows during tick-data validation", dropped)

    if df.empty:
        raise ValueError("Dataset is empty after cleaning.")
    return df


def resample_ticks(df: pd.DataFrame, freq: str = "1min") -> pd.DataFrame:
    out = df.copy().set_index("timestamp")
    agg = out.resample(freq).agg({"price": "last", "high": "max", "low": "min", "volume": "sum"})
    agg = agg.dropna().reset_index()
    if agg.empty:
        raise ValueError("No rows after resampling. Check frequency and source data coverage.")
    return agg


# ---------------------------------------------------------------------------
# Target computation - VECTORIZED (50-100x faster than loop)
# ---------------------------------------------------------------------------

def _compute_target_signed_range(price: np.ndarray, horizon: int) -> np.ndarray:
    """Vectorized signed-range target using sliding window view.

    For each index i, computes:
      range = max(price[i+1..i+horizon]) - min(price[i+1..i+horizon])
      direction = sign(mean(price[i+1..i+horizon]) - price[i])
      target = range * direction
    """
    n = len(price)
    targets = np.full(n, np.nan, dtype=np.float64)

    if n <= horizon:
        return targets

    # Build sliding windows of the *future* prices
    future = price[1:]  # shift by 1
    if len(future) < horizon:
        return targets

    windows = sliding_window_view(future, window_shape=horizon)  # (n-horizon, horizon)
    w_max = windows.max(axis=1)
    w_min = windows.min(axis=1)
    w_mean = windows.mean(axis=1)

    ranges = w_max - w_min
    directions = np.sign(w_mean - price[: len(windows)])
    # Normalise by current price â†’ dimensionless fraction (~0.002â€“0.015)
    # Makes targets stationary across price regimes (16k vs 24k Nifty etc.)
    current_price = np.clip(np.abs(price[: len(windows)]), 1e-8, None)
    targets[: len(windows)] = (ranges * directions) / current_price

    return targets


# ---------------------------------------------------------------------------
# Supervised data construction - pre-allocated arrays
# ---------------------------------------------------------------------------

def build_supervised_data(
    df: pd.DataFrame,
    seq_len: int,
    horizon: int,
    include_stoch_rsi: bool = True,
) -> DataBundle:
    if seq_len <= 0:
        raise ValueError("seq_len must be > 0")
    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    feat_df = add_all_indicators(df, include_stoch_rsi=include_stoch_rsi)
    feat_df["target"] = _compute_target_signed_range(feat_df["price"].to_numpy(), horizon=horizon)
    feat_df = feat_df.dropna().reset_index(drop=True)

    if len(feat_df) <= seq_len:
        raise ValueError("Not enough rows after feature engineering for selected seq_len/horizon.")

    feature_columns = [col for col in feat_df.columns if col not in {"timestamp", "target"}]
    if not feature_columns:
        raise ValueError("No feature columns available after preprocessing.")

    x_values = feat_df[feature_columns].to_numpy(dtype=np.float32)
    y_values = feat_df["target"].to_numpy(dtype=np.float32)

    n_samples = len(feat_df) - seq_len
    n_features = x_values.shape[1]

    # Pre-allocate arrays instead of list.append loop
    sequences = np.empty((n_samples, seq_len, n_features), dtype=np.float32)
    xgb_features = np.empty((n_samples, n_features), dtype=np.float32)
    targets = np.empty(n_samples, dtype=np.float32)

    for i, end_idx in enumerate(range(seq_len, len(feat_df))):
        sequences[i] = x_values[end_idx - seq_len : end_idx]
        xgb_features[i] = x_values[end_idx - 1]
        targets[i] = y_values[end_idx]

    if n_samples == 0:
        raise ValueError("Not enough rows for selected seq_len/horizon.")

    return DataBundle(
        sequences=sequences,
        xgb_features=xgb_features,
        targets=targets,
        feature_columns=feature_columns,
    )


# ---------------------------------------------------------------------------
# Scaling helper
# ---------------------------------------------------------------------------

def scale_sequences(scaler: StandardScaler | RobustScaler, seq: np.ndarray) -> np.ndarray:
    """Scale 3D sequence array (B, T, F) using a fitted 2D scaler."""
    shape = seq.shape
    flat = seq.reshape(-1, shape[-1])
    return scaler.transform(flat).reshape(shape)


# ---------------------------------------------------------------------------
# Train / Validation split with scaling
# ---------------------------------------------------------------------------

def prepare_split(
    bundle: DataBundle,
    train_ratio: float = 0.8,
    use_robust_scaler: bool = False,
) -> SplitBundle:
    """Split data chronologically and fit scaler on training set only."""
    split = int(train_ratio * len(bundle.targets))

    x_seq_train, x_seq_val = bundle.sequences[:split], bundle.sequences[split:]
    x_xgb_train, x_xgb_val = bundle.xgb_features[:split], bundle.xgb_features[split:]
    y_train, y_val = bundle.targets[:split], bundle.targets[split:]

    scaler = RobustScaler() if use_robust_scaler else StandardScaler()
    scaler.fit(x_seq_train.reshape(-1, x_seq_train.shape[-1]))

    return SplitBundle(
        x_seq_train=scale_sequences(scaler, x_seq_train),
        x_seq_val=scale_sequences(scaler, x_seq_val),
        x_xgb_train=scaler.transform(x_xgb_train),
        x_xgb_val=scaler.transform(x_xgb_val),
        y_train=y_train,
        y_val=y_val,
        scaler=scaler,
        feature_columns=bundle.feature_columns,
    )


def walk_forward_splits(
    bundle: DataBundle,
    train_bars: int = 60 * 375,
    test_bars: int = 5 * 375,
    use_robust_scaler: bool = False,
) -> Iterator[SplitBundle]:
    """Yields rolling walk-forward SplitBundles (e.g., 60-day train, 5-day test)."""
    step_size = test_bars
    total_samples = len(bundle.targets)

    if total_samples < train_bars + test_bars:
        # Fallback to standard 80/20 split if there isn't enough data for walk-forward
        yield prepare_split(bundle, train_ratio=0.8, use_robust_scaler=use_robust_scaler)
        return

    for start_idx in range(0, total_samples - train_bars - test_bars + 1, step_size):
        train_end = start_idx + train_bars
        test_end = train_end + test_bars

        x_seq_train = bundle.sequences[start_idx:train_end]
        x_seq_val = bundle.sequences[train_end:test_end]
        x_xgb_train = bundle.xgb_features[start_idx:train_end]
        x_xgb_val = bundle.xgb_features[train_end:test_end]
        y_train = bundle.targets[start_idx:train_end]
        y_val = bundle.targets[train_end:test_end]

        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        # Scale only on training fold to prevent data leakage
        scaler.fit(x_seq_train.reshape(-1, x_seq_train.shape[-1]))

        yield SplitBundle(
            x_seq_train=scale_sequences(scaler, x_seq_train),
            x_seq_val=scale_sequences(scaler, x_seq_val),
            x_xgb_train=scaler.transform(x_xgb_train),
            x_xgb_val=scaler.transform(x_xgb_val),
            y_train=y_train,
            y_val=y_val,
            scaler=scaler,
            feature_columns=bundle.feature_columns,
        )


def anchored_walk_forward_splits(
    bundle: DataBundle,
    min_train_bars: int = 40 * 375,
    test_bars: int = 10 * 375,
    n_splits: int = 5,
    use_robust_scaler: bool = False,
) -> Iterator[SplitBundle]:
    """Yields anchored expanding train windows with rolling validation windows."""
    total_samples = len(bundle.targets)
    
    available_for_test = total_samples - min_train_bars
    if available_for_test < test_bars * n_splits:
        test_bars = max(available_for_test // n_splits, 375)
        
    if min_train_bars + test_bars > total_samples:
        yield prepare_split(bundle, train_ratio=0.8, use_robust_scaler=use_robust_scaler)
        return

    for i in range(n_splits):
        train_end = min_train_bars + (i * test_bars)
        test_end = min(train_end + test_bars, total_samples)
        
        if train_end >= total_samples or test_end == train_end:
            break

        x_seq_train = bundle.sequences[0:train_end]
        x_seq_val = bundle.sequences[train_end:test_end]
        x_xgb_train = bundle.xgb_features[0:train_end]
        x_xgb_val = bundle.xgb_features[train_end:test_end]
        y_train = bundle.targets[0:train_end]
        y_val = bundle.targets[train_end:test_end]

        scaler = RobustScaler() if use_robust_scaler else StandardScaler()
        scaler.fit(x_seq_train.reshape(-1, x_seq_train.shape[-1]))

        yield SplitBundle(
            x_seq_train=scale_sequences(scaler, x_seq_train),
            x_seq_val=scale_sequences(scaler, x_seq_val),
            x_xgb_train=scaler.transform(x_xgb_train),
            x_xgb_val=scaler.transform(x_xgb_val),
            y_train=y_train,
            y_val=y_val,
            scaler=scaler,
            feature_columns=bundle.feature_columns,
        )
