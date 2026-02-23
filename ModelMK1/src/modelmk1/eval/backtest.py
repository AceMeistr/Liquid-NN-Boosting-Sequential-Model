from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from modelmk1.common.paths import get_app_paths
from modelmk1.common.runtime import write_json


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float((np.sign(y_true) == np.sign(y_pred)).mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest using generated predictions")
    parser.add_argument("--predictions", type=str, default=None)
    args = parser.parse_args()

    paths = get_app_paths()
    pred_path = paths.outputs / "predictions.csv" if args.predictions is None else args.predictions
    df = pd.read_csv(pred_path)

    y_pred = df["prediction"].to_numpy(dtype=np.float64)
    y_true = df["target"].to_numpy(dtype=np.float64)
    pnl = np.sign(y_pred) * y_true
    equity = np.cumsum(pnl)
    drawdown = equity - np.maximum.accumulate(equity)

    metrics = {
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "directional_accuracy": directional_accuracy(y_true, y_pred),
        "total_pnl_proxy": float(np.sum(pnl)),
        "max_drawdown_proxy": float(np.min(drawdown)),
    }
    write_json(paths.outputs / "backtest_metrics.json", metrics)
    print(metrics)


if __name__ == "__main__":
    main()
