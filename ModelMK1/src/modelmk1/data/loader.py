from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from modelmk1.features.indicators import add_all_indicators


REQUIRED_COLUMNS = {"timestamp", "price", "high", "low", "volume"}


@dataclass
class DataBundle:
    sequences: np.ndarray
    xgb_features: np.ndarray
    targets: np.ndarray
    feature_columns: list[str]


class TickDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray) -> None:
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32).unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.sequences[idx], self.targets[idx]


def load_tick_df(path: str) -> pd.DataFrame:
    source = pd.read_parquet(path) if path.lower().endswith((".parquet", ".pq")) else pd.read_csv(path)

    missing = REQUIRED_COLUMNS.difference(source.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = source.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=False, errors="coerce")
    df = df.dropna(subset=["timestamp", "price", "high", "low", "volume"])
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"]).reset_index(drop=True)

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


def _compute_target_signed_range(price: np.ndarray, horizon: int) -> np.ndarray:
    targets = np.full(shape=(len(price),), fill_value=np.nan, dtype=np.float64)
    for idx in range(len(price) - horizon):
        future = price[idx + 1 : idx + 1 + horizon]
        future_range = future.max() - future.min()
        direction = np.sign(future.mean() - price[idx])
        targets[idx] = future_range if direction >= 0 else -future_range
    return targets


def build_supervised_data(df: pd.DataFrame, seq_len: int, horizon: int, include_stoch_rsi: bool = True) -> DataBundle:
    feat_df = add_all_indicators(df, include_stoch_rsi=include_stoch_rsi)
    feat_df["target"] = _compute_target_signed_range(feat_df["price"].to_numpy(), horizon=horizon)
    feat_df = feat_df.dropna().reset_index(drop=True)

    feature_columns = [col for col in feat_df.columns if col not in {"timestamp", "target"}]
    x_values = feat_df[feature_columns].to_numpy(dtype=np.float32)
    y_values = feat_df["target"].to_numpy(dtype=np.float32)

    sequences: list[np.ndarray] = []
    xgb_features: list[np.ndarray] = []
    targets: list[float] = []
    for end_idx in range(seq_len, len(feat_df)):
        sequences.append(x_values[end_idx - seq_len : end_idx])
        xgb_features.append(x_values[end_idx - 1])
        targets.append(float(y_values[end_idx]))

    if not sequences:
        raise ValueError("Not enough rows for selected seq_len/horizon.")

    return DataBundle(
        sequences=np.stack(sequences),
        xgb_features=np.stack(xgb_features),
        targets=np.array(targets, dtype=np.float32),
        feature_columns=feature_columns,
    )
