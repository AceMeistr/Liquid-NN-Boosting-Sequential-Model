from __future__ import annotations

from feature_engine import FeatureEngine


def test_compute_momentum_features_columns(pipeline_config, synthetic_ohlcv_5d):
    engine = FeatureEngine(pipeline_config)
    out = engine.compute_momentum_features(synthetic_ohlcv_5d["idx_log_ret"].fillna(0.0))

    assert {"M_5", "M_15", "M_30", "MDS", "idx_mom_decay"}.issubset(out.columns)
    assert len(out) == len(synthetic_ohlcv_5d)


def test_compute_target_sets_last_bar_nan(pipeline_config, synthetic_ohlcv_5d):
    engine = FeatureEngine(pipeline_config)
    df = synthetic_ohlcv_5d[["session_date", "idx_log_ret"]].copy()

    out = engine.compute_target(df)
    assert "idx_target_1m" in out.columns

    last_rows = out.groupby("session_date").tail(1)
    assert last_rows["idx_target_1m"].isna().all()


def test_compute_constituent_features_has_core_columns(pipeline_config, synthetic_ohlcv_5d):
    engine = FeatureEngine(pipeline_config)
    out = engine.compute_constituent_features(synthetic_ohlcv_5d)

    expected = {"wrs_residual", "cssd", "concentration_ratio", "stale_feed_count"}
    assert expected.issubset(out.columns)
    assert len(out) == len(synthetic_ohlcv_5d)
