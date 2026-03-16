from __future__ import annotations

import pandas as pd

from preprocessing_engine import PreprocessingEngine


def test_validate_ohlcv_quarantines_invalid_bar(pipeline_config, synthetic_ohlcv_5d):
    engine = PreprocessingEngine(pipeline_config)
    df = synthetic_ohlcv_5d.copy()

    bad_idx = df.index[10]
    df.loc[bad_idx, "idx_high"] = df.loc[bad_idx, "idx_close"] - 10

    out = engine.validate_ohlcv(df, "idx")
    assert out.loc[bad_idx, "idx_is_invalid_bar"]
    assert pd.isna(out.loc[bad_idx, "idx_open"])
    assert pd.isna(out.loc[bad_idx, "idx_high"])
    assert pd.isna(out.loc[bad_idx, "idx_low"])
    assert pd.isna(out.loc[bad_idx, "idx_close"])


def test_compute_log_returns_resets_on_session_open(pipeline_config, synthetic_ohlcv_5d):
    engine = PreprocessingEngine(pipeline_config)
    df = synthetic_ohlcv_5d.copy()

    out = engine.compute_log_returns(df, "idx")
    first_bars = out["bar_index_in_session"] == 0

    assert out.loc[first_bars, "idx_log_ret"].isna().all()
    assert "idx_overnight_ret" in out.columns


def test_detect_stale_feeds_flags_runs(pipeline_config, synthetic_ohlcv_5d):
    engine = PreprocessingEngine(pipeline_config)
    df = synthetic_ohlcv_5d.copy()

    idxs = df.index[20:28]
    const_price = float(df.loc[idxs[0], "reliance_close"])
    for col in ["reliance_open", "reliance_high", "reliance_low", "reliance_close"]:
        df.loc[idxs, col] = const_price

    stale = engine.detect_stale_feeds(df, "reliance", window=5)
    assert stale.loc[idxs[-1]]
