from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modelmk1.models.xgb_spectral import load_constituent_price_panel
from modelmk1.tuning.optuna_xgb_spectral import _build_sqlite_storage_url, _parse_tickers


def _write_nested_price_parquet(path: Path, seed: int) -> None:
    rng = np.random.default_rng(seed)
    timestamps = pd.date_range("2025-01-01", periods=600, freq="min")
    returns = 0.00005 + rng.normal(0.0, 0.0012, size=len(timestamps))
    price = 100.0 * np.cumprod(1.0 + returns)
    spread = np.abs(rng.normal(0.001, 0.0003, size=len(timestamps)))

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "idx_close": price,
            "idx_high": price * (1.0 + spread),
            "idx_low": price * np.maximum(1.0 - spread, 0.0001),
            "idx_volume": rng.integers(500, 4000, size=len(timestamps)),
        }
    )
    frame.to_parquet(path, index=False)


def test_load_constituent_price_panel_supports_nested_directory(tmp_path: Path) -> None:
    nested = tmp_path / "features" / "2025" / "01"
    nested.mkdir(parents=True)

    _write_nested_price_parquet(nested / "part_a.parquet", seed=11)
    _write_nested_price_parquet(nested / "part_b.parquet", seed=13)

    market = load_constituent_price_panel(tmp_path / "features")

    assert len(market) > 500
    assert {"timestamp", "price", "high", "low", "volume"}.issubset(set(market.columns))


def test_build_sqlite_storage_url_normalizes_windows_path() -> None:
    raw = r"C:\\repo\\ModelMK1\\outputs\\model\\optuna_xgb_study.db"
    storage = _build_sqlite_storage_url(raw)
    assert storage == "sqlite:///C:/repo/ModelMK1/outputs/model/optuna_xgb_study.db"


def test_parse_tickers_returns_clean_values() -> None:
    assert _parse_tickers("  rsi_14 , macd_hist,vol_z_20 ,, ") == ["rsi_14", "macd_hist", "vol_z_20"]
    assert _parse_tickers(None) is None
    assert _parse_tickers(" , , ") is None
