from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from modelmk1.models.xgb_spectral import (
    build_latest_correlation_spectral,
    build_spectral_training_matrices,
    load_constituent_price_panel,
    train_xgb_with_dmatrix,
)


def _make_tick_csv(path: Path, n_steps: int = 1800) -> None:
    rng = np.random.default_rng(7)
    timestamps = pd.date_range("2025-01-01", periods=n_steps, freq="min")

    drift = 0.00003
    noise = rng.normal(0.0, 0.0015, size=n_steps)
    returns = drift + noise
    price = 20000.0 * np.cumprod(1.0 + returns)

    spread = np.abs(rng.normal(0.0009, 0.0003, size=n_steps))
    high = price * (1.0 + spread)
    low = price * np.maximum(1.0 - spread, 0.0001)
    volume = rng.integers(1000, 5000, size=n_steps)

    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "price": price,
            "high": high,
            "low": low,
            "volume": volume,
        }
    )
    frame.to_csv(path, index=False)


def test_load_constituent_price_panel_returns_market_frame(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    _make_tick_csv(source)

    market = load_constituent_price_panel(source)

    assert {"timestamp", "price", "high", "low", "volume"}.issubset(set(market.columns))
    assert len(market) > 500


def test_build_latest_correlation_spectral_topn(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    _make_tick_csv(source)

    market = load_constituent_price_panel(source)
    latest = build_latest_correlation_spectral(
        market,
        top_n=8,
        corr_window=45,
        seq_len=48,
        horizon=2,
        robust_quantile_low=15,
        robust_quantile_high=85,
    )

    assert len(latest.constituents) == 8
    assert latest.correlation_matrix.shape == (8, 8)
    assert latest.eigenvalues.shape == (8,)
    assert np.all(np.isfinite(latest.correlation_matrix))
    assert np.allclose(np.diag(latest.correlation_matrix), 1.0)
    assert np.all(np.diff(latest.eigenvalues) <= 1e-7)


def test_build_spectral_training_matrices_shapes(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    _make_tick_csv(source)

    market = load_constituent_price_panel(source)
    dataset = build_spectral_training_matrices(
        market,
        top_n=8,
        corr_window=30,
        horizon=2,
        seq_len=48,
        robust_quantile_low=20,
        robust_quantile_high=85,
    )

    assert dataset.features.ndim == 2
    assert dataset.features.shape[1] == 16
    assert len(dataset.feature_names) == 16
    assert len(dataset.targets) == len(dataset.features)
    assert len(dataset.labels) == len(dataset.features)
    assert len(dataset.timestamps) == len(dataset.features)
    assert len(dataset.constituents) == 8
    assert np.all(np.isfinite(dataset.features))
    assert set(np.unique(dataset.labels)).issubset({0.0, 1.0})


def test_train_xgb_with_dmatrix_classification_metrics(tmp_path: Path) -> None:
    source = tmp_path / "market.csv"
    _make_tick_csv(source)

    market = load_constituent_price_panel(source)
    dataset = build_spectral_training_matrices(
        market,
        top_n=8,
        corr_window=25,
        horizon=1,
        seq_len=32,
        robust_quantile_low=15,
        robust_quantile_high=85,
    )

    booster, metrics = train_xgb_with_dmatrix(
        dataset,
        train_ratio=0.8,
        num_boost_round=80,
        early_stopping_rounds=12,
    )

    assert booster.num_boosted_rounds() > 0
    assert metrics["train_samples"] > 0
    assert metrics["val_samples"] > 0
    assert np.isfinite(metrics["val_mse"])
    assert np.isfinite(metrics["val_mae"])
    assert np.isfinite(metrics["val_sharpe_ratio"])
    assert 0.0 <= metrics["val_directional_accuracy"] <= 1.0
