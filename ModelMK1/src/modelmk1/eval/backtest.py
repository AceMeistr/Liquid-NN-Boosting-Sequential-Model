from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from modelmk1.eval.cpcv import CPCVConfig, evaluate_cpcv_distribution
from modelmk1.common.paths import get_app_paths
from modelmk1.common.runtime import write_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def sharpe_ratio(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Annualized Sharpe Ratio (aggregates intra-day to daily if annualization > 252)."""
    if len(returns) < 2:
        return 0.0

    calc_returns = returns
    ann_factor = annualization

    # If using minute-level annualization (e.g. 94500 for NSE), aggregate to daily first.
    if annualization > 252.0:
        bars_per_day = int(annualization / 252.0)
        pad_size = (bars_per_day - (len(returns) % bars_per_day)) % bars_per_day
        if pad_size > 0:
            padded = np.pad(returns, (0, pad_size), mode="constant", constant_values=0.0)
        else:
            padded = returns
        calc_returns = padded.reshape(-1, bars_per_day).sum(axis=1)
        ann_factor = 252.0

    if len(calc_returns) < 2:
        return 0.0

    sigma = float(np.std(calc_returns, ddof=1))
    if sigma <= 1e-12:
        return 0.0
    return float(np.mean(calc_returns) / sigma * np.sqrt(ann_factor))


def sortino_ratio(returns: np.ndarray, annualization: float = 252.0) -> float:
    """Sortino Ratio â€” penalizes only downside deviation."""
    downside = returns[returns < 0]
    if len(downside) < 2:
        return 0.0
    downside_std = downside.std()
    return float(returns.mean() / (downside_std + 1e-12) * np.sqrt(annualization))


def calmar_ratio(returns: np.ndarray, equity: np.ndarray) -> float:
    """Calmar Ratio â€” annualized return / max drawdown."""
    total_return = equity[-1] - equity[0] if len(equity) > 1 else 0.0
    drawdown = equity - np.maximum.accumulate(equity)
    max_dd = abs(float(np.min(drawdown)))
    if max_dd == 0:
        return 0.0
    return float(total_return / max_dd)


def win_rate(pnl: np.ndarray) -> float:
    trades = pnl[pnl != 0]
    if len(trades) == 0:
        return 0.0
    return float((trades > 0).sum() / len(trades))


def profit_factor(pnl: np.ndarray) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def max_consecutive_losses(pnl: np.ndarray) -> int:
    max_streak = 0
    current = 0
    for p in pnl:
        if p < 0:
            current += 1
            max_streak = max(max_streak, current)
        else:
            current = 0
    return max_streak


def expectancy(pnl: np.ndarray) -> float:
    """Average PnL per trade."""
    trades = pnl[pnl != 0]
    if len(trades) == 0:
        return 0.0
    return float(trades.mean())


# ---------------------------------------------------------------------------
# Full backtest runner
# ---------------------------------------------------------------------------

def run_backtest(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    commission_bps: float = 2.0,
    slippage_bps: float = 1.0,
    market_impact_bps: float = 0.5,
) -> dict:
    """Run comprehensive backtest and return metrics dictionary."""
    direction = np.sign(y_pred)
    gross_pnl = direction * y_true
    cost_per_trade = np.abs(direction) * ((2.0 * commission_bps + slippage_bps + market_impact_bps) / 10_000.0)
    pnl = gross_pnl - cost_per_trade
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(equity)
    cpcv = evaluate_cpcv_distribution(y_true, y_pred, config=CPCVConfig())

    metrics = {
        # Error metrics
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        # Directional
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        # PnL metrics
        "total_pnl_proxy": float(np.sum(pnl)),
        "max_drawdown_proxy": float(np.min(drawdown)),
        "win_rate": win_rate(pnl),
        "profit_factor": profit_factor(pnl),
        "expectancy": expectancy(pnl),
        "max_consecutive_losses": max_consecutive_losses(pnl),
        # Risk-adjusted returns
        "sharpe_ratio": sharpe_ratio(pnl),
        "sortino_ratio": sortino_ratio(pnl),
        "calmar_ratio": calmar_ratio(pnl, equity),
        # Distribution
        "total_trades": int(len(pnl)),
        "winning_trades": int((pnl > 0).sum()),
        "losing_trades": int((pnl < 0).sum()),
        # Cost assumptions and CPCV distribution
        "commission_bps": float(commission_bps),
        "slippage_bps": float(slippage_bps),
        "market_impact_bps": float(market_impact_bps),
        "gross_total_pnl_proxy": float(np.sum(gross_pnl)),
        "cpcv": cpcv,
        "cpcv_directional_accuracy_mean": float(cpcv.get("directional_accuracy_mean", 0.0)),
        "cpcv_directional_accuracy_std": float(cpcv.get("directional_accuracy_std", 0.0)),
        "cpcv_pbo_proxy": float(cpcv.get("pbo_proxy", 1.0)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest using generated predictions")
    parser.add_argument("--predictions", type=str, default=None)
    args = parser.parse_args()

    paths = get_app_paths()
    pred_path = paths.outputs / "predictions.csv" if args.predictions is None else args.predictions
    df = pd.read_csv(pred_path)

    y_pred = df["prediction"].to_numpy(dtype=np.float64)
    y_true = df["target"].to_numpy(dtype=np.float64)

    metrics = run_backtest(y_true, y_pred)
    write_json(paths.outputs / "backtest_metrics.json", metrics)

    logger.info("=== Backtest Results ===")
    for k, v in metrics.items():
        logger.info("  %-30s %s", k, f"{v:.6f}" if isinstance(v, float) else v)

    print(metrics)


if __name__ == "__main__":
    main()
