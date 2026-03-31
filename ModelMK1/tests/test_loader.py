from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from modelmk1.data.loader import (
    build_supervised_data,
    load_tick_df,
    prepare_split,
    scale_sequences,
)


@pytest.fixture
def minimal_tick_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(42)
    n = 400
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
            "price": 100 + np.cumsum(rng.normal(0, 0.1, n)),
            "high": 100 + np.cumsum(rng.normal(0, 0.1, n)) + 0.5,
            "low": 100 + np.cumsum(rng.normal(0, 0.1, n)) - 0.5,
            "volume": rng.integers(10, 1000, size=n).astype(float),
        }
    )
    target = tmp_path / "ticks.csv"
    frame.to_csv(target, index=False)
    return target


def test_load_tick_df_filters_invalid_rows(minimal_tick_csv: Path) -> None:
    df = load_tick_df(str(minimal_tick_csv))
    assert not df.empty
    assert {"timestamp", "price", "high", "low", "volume"}.issubset(df.columns)


def test_build_supervised_data_invalid_args(minimal_tick_csv: Path) -> None:
    df = load_tick_df(str(minimal_tick_csv))
    with pytest.raises(ValueError):
        build_supervised_data(df, seq_len=0, horizon=15)
    with pytest.raises(ValueError):
        build_supervised_data(df, seq_len=64, horizon=0)


def test_build_supervised_data_shapes(minimal_tick_csv: Path) -> None:
    df = load_tick_df(str(minimal_tick_csv))
    bundle = build_supervised_data(df, seq_len=32, horizon=10)
    assert bundle.sequences.ndim == 3
    assert bundle.sequences.shape[1] == 32
    assert bundle.xgb_features.ndim == 2
    assert len(bundle.targets) == len(bundle.sequences)
    assert len(bundle.feature_columns) > 0


def test_prepare_split_scaling(minimal_tick_csv: Path) -> None:
    df = load_tick_df(str(minimal_tick_csv))
    bundle = build_supervised_data(df, seq_len=32, horizon=10)
    split = prepare_split(bundle, train_ratio=0.8)

    # Train set should be ~80%
    total = len(split.y_train) + len(split.y_val)
    assert abs(len(split.y_train) / total - 0.8) < 0.02

    # Scaled data should have reasonable range (not raw)
    assert split.x_seq_train.std() < 10.0


def test_scale_sequences_preserves_shape() -> None:
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(0)
    data = rng.normal(size=(50, 10, 5)).astype(np.float32)
    scaler = StandardScaler()
    scaler.fit(data.reshape(-1, 5))
    result = scale_sequences(scaler, data)
    assert result.shape == data.shape
