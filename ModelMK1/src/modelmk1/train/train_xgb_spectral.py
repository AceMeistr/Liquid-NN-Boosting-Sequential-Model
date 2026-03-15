from __future__ import annotations

import argparse
import logging

import pandas as pd

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json
from modelmk1.models.xgb_spectral import (
    build_latest_correlation_spectral,
    build_spectral_training_matrices,
    load_constituent_price_panel,
    train_xgb_with_dmatrix,
)


logger = logging.getLogger(__name__)





def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train classification XGBoost using LNN-generated spectral indicator features")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--corr-window", type=int, default=60)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--num-boost-round", type=int, default=700)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--robust-quantile-low", type=float, default=15.0)
    parser.add_argument("--robust-quantile-high", type=float, default=85.0)
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated indicator feature names to force")
    return parser


def run_xgb_spectral_training(args: argparse.Namespace) -> dict:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    data_path = resolve_data_file(getattr(args, "data_path", None))
    raw_tickers = getattr(args, "tickers", None)
    indicator_names = [t.strip() for t in raw_tickers.split(",")] if raw_tickers else None

    logger.info("Loading market frame from %s", data_path)
    market_frame = load_constituent_price_panel(data_path)

    latest = build_latest_correlation_spectral(
        market_frame,
        top_n=args.top_n,
        corr_window=args.corr_window,
        horizon=args.horizon,
        seq_len=args.seq_len,
        tickers=indicator_names,
        robust_quantile_low=args.robust_quantile_low,
        robust_quantile_high=args.robust_quantile_high,
    )

    dataset = build_spectral_training_matrices(
        market_frame,
        top_n=args.top_n,
        corr_window=args.corr_window,
        horizon=args.horizon,
        seq_len=args.seq_len,
        tickers=indicator_names,
        robust_quantile_low=args.robust_quantile_low,
        robust_quantile_high=args.robust_quantile_high,
    )

    booster, train_metrics = train_xgb_with_dmatrix(
        dataset,
        train_ratio=args.train_ratio,
        num_boost_round=args.num_boost_round,
        early_stopping_rounds=args.early_stopping_rounds,
    )

    corr_df = pd.DataFrame(
        latest.correlation_matrix,
        index=latest.constituents,
        columns=latest.constituents,
    )
    corr_path = model_dir / f"sensex_top{args.top_n}_correlation_matrix.csv"
    corr_df.to_csv(corr_path)

    model_path = model_dir / f"xgb_spectral_top{args.top_n}.json"
    booster.save_model(model_path)

    summary = {
        "dataset": str(data_path),
        "model_type": "xgboost_classifier_spectral_lnn_features",
        "top_n": int(args.top_n),
        "corr_window": int(args.corr_window),
        "horizon": int(args.horizon),
        "seq_len": int(args.seq_len),
        "train_ratio": float(args.train_ratio),
        "num_boost_round": int(args.num_boost_round),
        "early_stopping_rounds": int(args.early_stopping_rounds),
        "robust_quantile_low": float(args.robust_quantile_low),
        "robust_quantile_high": float(args.robust_quantile_high),
        "selected_indicators": latest.constituents,
        "latest_correlation_matrix_csv": str(corr_path),
        "latest_eigenvalues": [float(v) for v in latest.eigenvalues.tolist()],
        "latest_matrix_window_end": str(latest.window_end_timestamp),
        "xgb_model": str(model_path),
        "feature_names": dataset.feature_names,
        "num_samples": int(len(dataset.targets)),
        **train_metrics,
    }

    write_json(model_dir / "xgb_spectral_metrics.json", summary)
    logger.info("Saved reconstructed XGBoost spectral artifacts in %s", model_dir)
    return summary


def main() -> None:
    args = build_parser().parse_args()
    print(run_xgb_spectral_training(args))


if __name__ == "__main__":
    main()
