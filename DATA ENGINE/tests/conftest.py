from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import PipelineConfig


@pytest.fixture()
def pipeline_config(tmp_path, monkeypatch) -> PipelineConfig:
    monkeypatch.setenv("UPSTOX_API_KEY", "test_key")
    monkeypatch.setenv("UPSTOX_API_SECRET", "test_secret")

    data_root = tmp_path / "data"
    log_root = tmp_path / "logs"

    cfg = PipelineConfig(
        DATA_ROOT=data_root,
        LOG_ROOT=log_root,
        RFR_PATH=data_root / "rfr" / "rbi_tbill_91d.csv",
        WEIGHTS_PATH=data_root / "instruments" / "sensex_weights.csv",
        REGIME_TABLE_PATH=data_root / "regimes" / "expiry_regimes.csv",
        CA_TABLE_PATH=data_root / "corporate_actions" / "ca_table.csv",
        INSTRUMENT_MASTER_PATH=data_root / "instruments" / "bse_instruments.csv",
    )
    cfg.ensure_directories()

    (cfg.RFR_PATH).write_text("date,rate\n2026-01-01,0.07\n", encoding="utf-8")
    (cfg.WEIGHTS_PATH).write_text(
        "symbol,weight,sector\nRELIANCE,0.6,energy\nHDFCBANK,0.4,bank\n",
        encoding="utf-8",
    )
    (cfg.REGIME_TABLE_PATH).write_text(
        "date_from,date_to,expiry_weekday,regime_flag\n2020-01-01,2099-12-31,3,2\n",
        encoding="utf-8",
    )
    (cfg.CA_TABLE_PATH).write_text("symbol,ex_date,split_ratio,dividend_amount\n", encoding="utf-8")
    (cfg.INSTRUMENT_MASTER_PATH).write_text(
        "trading_symbol,segment,instrument_key\nSENSEX,BSE_INDEX,BSE_INDEX|SENSEX\nRELIANCE,BSE_EQ,BSE_EQ|RELIANCE\nHDFCBANK,BSE_EQ,BSE_EQ|HDFCBANK\n",
        encoding="utf-8",
    )

    return cfg


@pytest.fixture()
def synthetic_ohlcv_5d() -> pd.DataFrame:
    trading_days = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]

    idx = []
    for d in trading_days:
        idx.extend(
            pd.date_range(
                start=f"{d} 09:15:00",
                end=f"{d} 15:29:00",
                freq="1min",
                tz="Asia/Kolkata",
            )
        )
    index = pd.DatetimeIndex(idx)

    n = len(index)
    rng = np.random.default_rng(42)

    base = np.linspace(78000, 78500, n, dtype=np.float32)
    noise = rng.normal(0, 15, n).astype(np.float32)
    idx_close = base + noise

    df = pd.DataFrame(index=index)
    df["session_date"] = pd.to_datetime(pd.Index(index.date))
    df["bar_index_in_session"] = np.tile(np.arange(375, dtype=np.int16), len(trading_days))

    df["idx_open"] = (idx_close - 2).astype(np.float32)
    df["idx_high"] = (idx_close + 3).astype(np.float32)
    df["idx_low"] = (idx_close - 4).astype(np.float32)
    df["idx_close"] = idx_close.astype(np.float32)
    df["idx_volume"] = rng.integers(1000, 5000, n, dtype=np.int32)

    rel_close = (2500 + np.linspace(0, 20, n) + rng.normal(0, 3, n)).astype(np.float32)
    hdfc_close = (1650 + np.linspace(0, 15, n) + rng.normal(0, 2, n)).astype(np.float32)

    for sym, close in (("reliance", rel_close), ("hdfcbank", hdfc_close)):
        df[f"{sym}_open"] = (close - 1).astype(np.float32)
        df[f"{sym}_high"] = (close + 2).astype(np.float32)
        df[f"{sym}_low"] = (close - 3).astype(np.float32)
        df[f"{sym}_close"] = close.astype(np.float32)
        df[f"{sym}_volume"] = rng.integers(100, 2000, n, dtype=np.int32)
        df[f"{sym}_is_stale"] = False
        df[f"{sym}_log_ret"] = np.log((close + 1e-8) / (np.roll(close, 1) + 1e-8)).astype(np.float32)

    df["idx_log_ret"] = np.log((df["idx_close"] + 1e-8) / (df["idx_close"].shift(1) + 1e-8)).astype("float32")
    df.loc[df["bar_index_in_session"] == 0, "idx_log_ret"] = np.nan
    df["idx_overnight_ret"] = np.nan
    df["idx_vol_zscore"] = ((df["idx_volume"] - df["idx_volume"].mean()) / df["idx_volume"].std()).astype("float32")

    df["synth_atm_iv"] = np.float32(0.22)
    df["iv_skew_25d"] = np.float32(0.03)
    df["iv_butterfly"] = np.float32(0.01)
    df["opt_delta_net"] = np.float32(0.12)
    df["opt_gamma_net"] = np.float32(0.004)
    df["opt_vega_net"] = np.float32(0.08)
    df["gex"] = np.float32(19000)
    df["total_pcr_vol"] = np.float32(1.1)
    df["total_pcr_oi"] = np.float32(1.05)
    df["oi_imbalance"] = np.float32(0.02)
    df["oi_flow"] = np.float32(0.01)
    df["days_to_expiry"] = np.float32(2.0)
    df["dte_normalized"] = np.float32(2.0 / 7.0)
    df["is_expiry_day"] = False
    df["regime_flag"] = np.int8(2)

    return df
