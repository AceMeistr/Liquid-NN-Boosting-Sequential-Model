from __future__ import annotations

import numpy as np
import pandas as pd

from modelmk1.features.indicators import (
    add_all_indicators,
    adx,
    cmf,
    get_feature_schema,
    obv,
    parkinson_volatility,
    vwap,
)


def test_feature_schema_sizes() -> None:
    full = get_feature_schema(include_stoch_rsi=True)
    base = get_feature_schema(include_stoch_rsi=False)

    assert len(full) == len(base) + 3
    assert "stoch_rsi" in full
    assert "stoch_rsi" not in base


def test_feature_schema_has_advanced_indicators() -> None:
    schema = get_feature_schema(include_stoch_rsi=True)
    advanced = ["vwap_20", "obv", "cmf_20", "adx", "plus_di", "minus_di",
                "parkinson_vol", "kc_upper", "bb_pct_b", "momentum_10",
                "price_sma_ratio", "tick_intensity"]
    for feat in advanced:
        assert feat in schema, f"Missing advanced indicator: {feat}"


def _make_df(n: int = 300) -> pd.DataFrame:
    return pd.DataFrame({
        "timestamp": pd.date_range("2025-01-01", periods=n, freq="min"),
        "price": 100 + np.cumsum(np.random.randn(n) * 0.1),
        "high": 100 + np.cumsum(np.random.randn(n) * 0.1) + 0.5,
        "low": 100 + np.cumsum(np.random.randn(n) * 0.1) - 0.5,
        "volume": np.random.randint(10, 1000, size=n).astype(float),
    })


def test_add_all_indicators_output_columns() -> None:
    df = _make_df(300)
    result = add_all_indicators(df, include_stoch_rsi=True)
    schema = get_feature_schema(include_stoch_rsi=True)
    for col in schema:
        assert col in result.columns, f"Missing column in output: {col}"
    assert not result.isnull().any().any(), "Output contains NaN values"


def test_vwap_is_finite() -> None:
    df = _make_df(100)
    result = vwap(df, 20)
    assert result.dropna().shape[0] > 0
    assert np.isfinite(result.dropna().values).all()


def test_obv_monotonic_on_constant_direction() -> None:
    df = pd.DataFrame({
        "price": np.arange(1, 101, dtype=float),
        "high": np.arange(1.5, 101.5, dtype=float),
        "low": np.arange(0.5, 100.5, dtype=float),
        "volume": np.ones(100) * 10,
    })
    result = obv(df)
    # Price always increasing â†’ OBV should be non-decreasing
    assert (result.diff().dropna() >= 0).all()
