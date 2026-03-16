from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

import ingestion_engine as ingestion_module
from ingestion_engine import IngestionEngine
from normalization_engine import NormalizationEngine


def _build_engine_stub(pipeline_config) -> IngestionEngine:
    engine = object.__new__(IngestionEngine)
    engine.cfg = pipeline_config
    engine.instruments = pd.read_csv(pipeline_config.INSTRUMENT_MASTER_PATH)
    engine.weights = pd.read_csv(pipeline_config.WEIGHTS_PATH)
    engine._build_instrument_lookup()
    return engine


def test_option_chain_sdk_fallback_call(monkeypatch, pipeline_config):
    class DummyOptionsApi:
        def get_put_call_option_chain(self, instrument_key, expiry_date):
            assert instrument_key == "BSE_INDEX|SENSEX"
            assert expiry_date == "2026-01-30"
            return {"data": []}

    monkeypatch.setattr(ingestion_module, "handle_401_retry", lambda fn, *a, **k: fn(*a, **k))

    engine = _build_engine_stub(pipeline_config)
    engine.option_api = DummyOptionsApi()

    resp = engine._call_option_chain("2026-01-30")
    assert isinstance(resp, dict)


def test_fetch_option_chain_handles_nan_spot(monkeypatch, pipeline_config):
    engine = _build_engine_stub(pipeline_config)

    def _next_expiry(trading_date: date) -> date:
        _ = trading_date
        return date(2026, 1, 30)

    def _call_option_chain(expiry_str: str):
        _ = expiry_str
        return {
        "data": [
            {
                "strike_price": 80000,
                "call_options": {
                    "iv": 0.22,
                    "delta": 0.40,
                    "gamma": 0.01,
                    "theta": -0.02,
                    "vega": 0.12,
                    "last_price": 120.0,
                    "volume": 100,
                    "oi": 500,
                },
                "put_options": {
                    "iv": 0.24,
                    "delta": -0.35,
                    "gamma": 0.01,
                    "theta": -0.03,
                    "vega": 0.11,
                    "last_price": 115.0,
                    "volume": 90,
                    "oi": 520,
                },
            },
            {
                "strike_price": 81000,
                "call_options": {
                    "iv": 0.23,
                    "delta": 0.30,
                    "gamma": 0.01,
                    "theta": -0.02,
                    "vega": 0.10,
                    "last_price": 95.0,
                    "volume": 80,
                    "oi": 400,
                },
                "put_options": {
                    "iv": 0.25,
                    "delta": -0.45,
                    "gamma": 0.01,
                    "theta": -0.03,
                    "vega": 0.13,
                    "last_price": 130.0,
                    "volume": 75,
                    "oi": 410,
                },
            },
        ]
    }

    monkeypatch.setattr(engine, "_next_expiry", _next_expiry)
    monkeypatch.setattr(engine, "_call_option_chain", _call_option_chain)

    out = engine.fetch_option_chain(date(2026, 1, 5), float("nan"))
    assert not out.empty
    assert {"iv", "delta", "gamma", "theta", "vega", "volume", "oi"}.issubset(out.columns)


def test_normalization_run_generates_fold_outputs(pipeline_config, synthetic_ohlcv_5d):
    engine = NormalizationEngine(pipeline_config)

    df = synthetic_ohlcv_5d.reset_index().rename(columns={"index": "timestamp"}).copy()
    df["idx_target_1m"] = (df["idx_log_ret"].shift(-1) > 0).astype("float32")
    df = df[df["idx_target_1m"].notna()].copy()

    # Add mandatory engineered columns used by scaler groups.
    for col in [
        "wrs_residual",
        "sector_bank_flow",
        "sector_it_flow",
        "sector_energy_flow",
        "sector_industrial_flow",
        "top5_dispersion",
        "MDS",
        "M_5",
        "M_15",
        "M_30",
        "idx_mom_decay",
        "idx_vwap_dev",
        "idx_vrp",
        "idx_rv5m",
        "idx_rv30m",
        "const_adv_dec",
        "adr",
    ]:
        if col not in df.columns:
            df[col] = np.float32(0.0)

    # Persist one feature file per trading day.
    for sess_date, part in df.groupby(pd.to_datetime(df["session_date"]).dt.date):
        out_path = pipeline_config.feature_root / f"{sess_date.year}" / f"{sess_date.month:02d}" / f"features_{sess_date.isoformat()}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        part.to_parquet(out_path, index=False)

    dates = sorted(pd.to_datetime(df["session_date"]).dt.date.unique())
    engine.run(pipeline_config.feature_root, pipeline_config.normalized_root, pipeline_config.scaler_root, dates)

    for fold in range(1, 6):
        split_dir = pipeline_config.normalized_root / f"split_{fold}"
        assert (split_dir / "lnn_train.parquet").exists()
        assert (split_dir / "lnn_val.parquet").exists()
        assert (split_dir / "xgb_train.parquet").exists()
        assert (split_dir / "xgb_val.parquet").exists()
