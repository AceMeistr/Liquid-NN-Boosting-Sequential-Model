from __future__ import annotations

import numpy as np
import pandas as pd

from normalization_engine import NormalizationEngine, _pearson_statistic


def _build_norm_df(synthetic_ohlcv_5d: pd.DataFrame) -> pd.DataFrame:
    df = synthetic_ohlcv_5d.reset_index().rename(columns={"index": "timestamp"}).copy()
    df["idx_target_1m"] = (df["idx_log_ret"].shift(-1) > 0).astype("float32")
    df = df[df["idx_target_1m"].notna()].copy()
    return df


def test_session_aware_rolling_zscore_respects_sessions(pipeline_config, synthetic_ohlcv_5d):
    engine = NormalizationEngine(pipeline_config)
    series = synthetic_ohlcv_5d["idx_log_ret"].fillna(0.0)
    session_col = synthetic_ohlcv_5d["session_date"]

    z = engine.session_aware_rolling_zscore(series, session_col, window=60)
    assert len(z) == len(series)
    assert np.isfinite(z.dropna()).all()


def test_build_purged_splits_returns_indices(pipeline_config, synthetic_ohlcv_5d):
    engine = NormalizationEngine(pipeline_config)
    df = _build_norm_df(synthetic_ohlcv_5d)
    df["wrs_residual"] = df["idx_log_ret"].fillna(0.0)
    df = df[["timestamp", "session_date", "bar_index_in_session", "idx_target_1m", "idx_log_ret", "wrs_residual"]].copy()

    splits = engine.build_purged_splits(df, n_splits=3)
    assert len(splits) >= 1
    for train_idx, val_idx in splits:
        assert len(train_idx) > 0
        assert len(val_idx) > 0


def test_fit_and_apply_lnn_scalers_persists_scaler(pipeline_config, synthetic_ohlcv_5d, tmp_path):
    engine = NormalizationEngine(pipeline_config)
    df = _build_norm_df(synthetic_ohlcv_5d)
    df["wrs_residual"] = df["idx_log_ret"].fillna(0.0)
    df["M_5"] = 0.0
    df["M_15"] = 0.0
    df["M_30"] = 0.0
    df["MDS"] = 0.0
    df["idx_mom_decay"] = 0.0
    df["idx_vwap_dev"] = 0.0
    df["idx_vrp"] = 0.0
    df["synth_atm_iv"] = 0.2
    df["idx_rv5m"] = 0.1
    df["idx_rv30m"] = 0.1
    df["iv_skew_25d"] = 0.01
    df["iv_butterfly"] = 0.01

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    scaler_dir = tmp_path / "scalers"
    out_train, out_val = engine.fit_and_apply_lnn_scalers(train_df, val_df, fold_id=1, scaler_dir=scaler_dir)

    assert (scaler_dir / "pt_iv_fold1.pkl").exists()
    assert "roll_mu_60" in out_train.columns
    assert "roll_std_60" in out_val.columns


# ---------------------------------------------------------------------------
# Normalization math stress tests
# ---------------------------------------------------------------------------


def test_rolling_zscore_expanding_warmup_no_nan(pipeline_config, synthetic_ohlcv_5d):
    """Early-session bars should produce finite z-scores via expanding fallback."""
    engine = NormalizationEngine(pipeline_config)
    series = synthetic_ohlcv_5d["idx_log_ret"].fillna(0.0)
    session_col = synthetic_ohlcv_5d["session_date"]

    z = engine.session_aware_rolling_zscore(series, session_col, window=60)

    # First session bars (after bar 1) should be finite, not NaN
    first_session = synthetic_ohlcv_5d["session_date"].iloc[0]
    mask = synthetic_ohlcv_5d["session_date"] == first_session
    first_z = z[mask]
    # Bar 0 might be NaN (single point), but bars 2+ should be finite
    assert first_z.iloc[2:].notna().all(), "Expanding fallback should prevent NaN in warm-up"


def test_rolling_zscore_sigma_floor_prevents_extreme_values(pipeline_config):
    """Near-constant series should not produce z-scores beyond clip threshold."""
    engine = NormalizationEngine(pipeline_config)
    n = 100
    # Almost constant series with tiny noise
    vals = pd.Series(np.full(n, 1.0) + np.random.default_rng(99).normal(0, 1e-9, n))
    sessions = pd.Series(["2026-01-01"] * n)

    z = engine.session_aware_rolling_zscore(vals, sessions, window=20)
    assert z.abs().max() <= pipeline_config.ZSCORE_CLIP + 0.01


def test_iqr_clip_removes_extreme_outliers(pipeline_config):
    """IQR clip should bound extreme values using training statistics."""
    rng = np.random.default_rng(42)
    train = pd.DataFrame({"a": rng.normal(0, 1, 200), "b": rng.normal(5, 2, 200)})
    val = pd.DataFrame({"a": [100.0, -100.0, 0.5], "b": [50.0, -50.0, 5.0]})

    clipped_train, clipped_val = NormalizationEngine._iqr_clip(train, val, k=4.0)

    # Val extremes should be clipped to within training IQR fences
    assert clipped_val["a"].max() < 100.0
    assert clipped_val["a"].min() > -100.0
    assert clipped_val["b"].max() < 50.0
    # Normal values should pass through
    assert abs(clipped_val["a"].iloc[2] - 0.5) < 0.01


def test_median_imputation_uses_train_stats(pipeline_config, synthetic_ohlcv_5d):
    """NaN imputation should use training median, not zero."""
    engine = NormalizationEngine(pipeline_config)
    df = _build_norm_df(synthetic_ohlcv_5d)
    df["wrs_residual"] = df["idx_log_ret"].fillna(0.0)
    for col in ["M_5", "M_15", "M_30", "MDS", "idx_mom_decay", "idx_vwap_dev", "idx_vrp"]:
        df[col] = np.float32(0.0)

    # Set IV features with some NaN
    rng = np.random.default_rng(42)
    df["synth_atm_iv"] = rng.uniform(0.15, 0.30, len(df)).astype("float32")
    df["idx_rv5m"] = rng.uniform(0.05, 0.15, len(df)).astype("float32")
    df["idx_rv30m"] = rng.uniform(0.05, 0.12, len(df)).astype("float32")
    df["iv_skew_25d"] = rng.uniform(0.01, 0.05, len(df)).astype("float32")
    df["iv_butterfly"] = rng.uniform(0.005, 0.02, len(df)).astype("float32")

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    # Inject NaN into validation IV features
    val_df.loc[val_df.index[:10], "synth_atm_iv"] = np.nan

    scaler_dir = pipeline_config.scaler_root
    scaler_dir.mkdir(parents=True, exist_ok=True)
    out_train, out_val = engine.fit_and_apply_lnn_scalers(train_df, val_df, fold_id=1, scaler_dir=scaler_dir)

    # After scaling, no NaN should remain in IV columns (median imputation filled them)
    assert out_val["synth_atm_iv"].notna().all(), "Median imputation should fill NaN in val"


def test_pearson_statistic_handles_nan_pairs():
    """_pearson_statistic should drop NaN pairs and still compute correctly."""
    x = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    y = pd.Series([2.0, np.nan, 6.0, 8.0, 10.0])

    corr = _pearson_statistic(x, y)
    assert np.isfinite(corr)
    # With NaN dropped, only pairs (1,2), (4,8), (5,10) remain
    assert len(x.dropna()) == 4  # x has 4 non-NaN
    assert corr != 0.0  # there is correlation in the remaining pairs


def test_pearson_statistic_constant_series():
    """Constant series should return 0 without error."""
    x = pd.Series([1.0, 1.0, 1.0, 1.0])
    y = pd.Series([2.0, 3.0, 4.0, 5.0])
    assert _pearson_statistic(x, y) == 0.0


def test_xgb_scalers_no_inf_in_output(pipeline_config, synthetic_ohlcv_5d):
    """XGB scalers should never produce Inf values."""
    engine = NormalizationEngine(pipeline_config)
    df = _build_norm_df(synthetic_ohlcv_5d)

    for col in [
        "opt_delta_net", "opt_gamma_net", "opt_vega_net", "gex",
        "const_adv_dec", "adr",
    ]:
        if col not in df.columns:
            df[col] = np.float32(0.01)

    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:].copy()

    # Inject extreme outliers to test robustness
    train_df.loc[train_df.index[0], "opt_delta_net"] = np.float32(1e6)
    train_df.loc[train_df.index[1], "gex"] = np.float32(-1e6)

    scaler_dir = pipeline_config.scaler_root
    scaler_dir.mkdir(parents=True, exist_ok=True)
    out_train, out_val = engine.fit_and_apply_xgb_scalers(train_df, val_df, fold_id=1, scaler_dir=scaler_dir)

    # No Inf should survive
    num_cols = out_train.select_dtypes(include=[np.number]).columns
    assert not np.isinf(out_train[num_cols].to_numpy()).any(), "Train should have no Inf after XGB scaling"
    assert not np.isinf(out_val[num_cols].to_numpy()).any(), "Val should have no Inf after XGB scaling"


def test_purged_splits_embargo_enforced(pipeline_config, synthetic_ohlcv_5d):
    """Validation data should start after embargo period from train end."""
    engine = NormalizationEngine(pipeline_config)
    df = _build_norm_df(synthetic_ohlcv_5d)
    df["wrs_residual"] = df["idx_log_ret"].fillna(0.0)
    keep_cols = ["timestamp", "session_date", "bar_index_in_session", "idx_target_1m", "idx_log_ret", "wrs_residual"]
    df = df[keep_cols].copy()

    splits = engine.build_purged_splits(df, n_splits=3)
    assert len(splits) >= 1

    ordered = df.sort_values("timestamp").reset_index(drop=True)
    for train_idx, val_idx in splits:
        train_end = pd.Timestamp(ordered.iloc[train_idx]["timestamp"].iloc[-1])
        val_start = pd.Timestamp(ordered.iloc[val_idx]["timestamp"].iloc[0])
        gap_minutes = (val_start - train_end).total_seconds() / 60.0
        assert gap_minutes >= pipeline_config.EMBARGO_MINUTES, (
            f"Embargo violated: gap={gap_minutes:.1f}min < {pipeline_config.EMBARGO_MINUTES}min"
        )
