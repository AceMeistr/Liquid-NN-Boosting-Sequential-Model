from __future__ import annotations

import numpy as np

from modelmk1.eval.backtest import (
    calmar_ratio,
    directional_accuracy,
    max_consecutive_losses,
    profit_factor,
    run_backtest,
    sharpe_ratio,
    sortino_ratio,
    win_rate,
)


def test_directional_accuracy_perfect() -> None:
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([0.5, -0.5, 0.3, -0.2])
    assert directional_accuracy(y_true, y_pred) == 1.0


def test_directional_accuracy_zero() -> None:
    y_true = np.array([1.0, -1.0, 1.0, -1.0])
    y_pred = np.array([-0.5, 0.5, -0.3, 0.2])
    assert directional_accuracy(y_true, y_pred) == 0.0


def test_sharpe_ratio_positive() -> None:
    returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
    sr = sharpe_ratio(returns)
    assert sr > 0


def test_sortino_ratio_no_downside() -> None:
    returns = np.array([0.01, 0.02, 0.015, 0.005, 0.01])
    sr = sortino_ratio(returns)
    assert sr >= 0


def test_profit_factor_all_wins() -> None:
    pnl = np.array([1.0, 2.0, 3.0])
    assert profit_factor(pnl) == float("inf")


def test_win_rate_basic() -> None:
    pnl = np.array([1.0, -1.0, 2.0, -0.5])
    assert win_rate(pnl) == 0.5


def test_max_consecutive_losses() -> None:
    pnl = np.array([1.0, -1.0, -1.0, -1.0, 1.0, -1.0])
    assert max_consecutive_losses(pnl) == 3


def test_run_backtest_returns_all_metrics() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, 100)
    y_pred = y_true + rng.normal(0, 0.5, 100)
    metrics = run_backtest(y_true, y_pred)
    required = [
        "mse", "mae", "rmse", "directional_accuracy",
        "total_pnl_proxy", "max_drawdown_proxy", "win_rate",
        "profit_factor", "sharpe_ratio", "sortino_ratio",
        "calmar_ratio", "total_trades", "cpcv_directional_accuracy_mean", "cpcv_pbo_proxy",
    ]
    for key in required:
        assert key in metrics, f"Missing metric: {key}"


def test_run_backtest_pbo_proxy_sensitivity() -> None:
    rng = np.random.default_rng(123)
    y_true = rng.normal(0.0, 1.0, 2048)

    # Good signal should generate fewer negative-sharpe CPCV slices than inverse signal.
    y_pred_good = y_true + rng.normal(0.0, 0.05, y_true.shape[0])
    y_pred_bad = -y_true + rng.normal(0.0, 0.05, y_true.shape[0])

    metrics_good = run_backtest(y_true, y_pred_good)
    metrics_bad = run_backtest(y_true, y_pred_bad)

    assert 0.0 <= metrics_good["cpcv_pbo_proxy"] <= 1.0
    assert 0.0 <= metrics_bad["cpcv_pbo_proxy"] <= 1.0
    assert metrics_good["cpcv_pbo_proxy"] <= metrics_bad["cpcv_pbo_proxy"]
