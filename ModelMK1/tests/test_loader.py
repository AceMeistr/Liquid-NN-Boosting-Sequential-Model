from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from modelmk1.data.loader import build_supervised_data, load_tick_df


@pytest.fixture
def minimal_tick_csv(tmp_path: Path) -> Path:
    frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-01", periods=300, freq="min"),
            "price": [100 + i * 0.1 for i in range(300)],
            "high": [100 + i * 0.1 + 0.05 for i in range(300)],
            "low": [100 + i * 0.1 - 0.05 for i in range(300)],
            "volume": [10 + i for i in range(300)],
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
