from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, safe_torch_load, write_json, read_json
from modelmk1.eval.backtest import directional_accuracy
from modelmk1.data.loader import build_supervised_data, load_tick_df, prepare_split, resample_ticks, scale_sequences
from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
from modelmk1.models.lnn_model import MarketLNN, predict_lnn_batched
from modelmk1.models.xgb_model import get_feature_importance, save_xgb, train_xgb

logger = logging.getLogger(__name__)


def _compute_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """Returns the Hurst Exponent to detect structural regimes.
    H > 0.5: Persistent/Trending (LNN should dominate)
    H < 0.5: Anti-persistent/Mean-Reverting (XGBoost should dominate)
    H = 0.5: Random Walk Noise
    """
    if len(series) < max_lag * 2:
        return 0.5 # Default to random walk if insufficient data
        
    lags = range(2, max_lag)
    tau = [np.sqrt(np.std(np.subtract(series[lag:], series[:-lag]))) for lag in lags]
    
    # Fit line to log-log plot to extract fractal dimension / Hurst
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return float(poly[0] * 2.0)


def _bates_granger_optimal_weights(err_a: np.ndarray, err_b: np.ndarray) -> tuple[float, float]:
    """Computes pure float64 restricted least squares variance-covariance weights.
    Avoids float32 underflow issues during matrix multiplication and inversion.
    """
    ea = err_a.astype(np.float64)
    eb = err_b.astype(np.float64)
    
    var_a = np.var(ea)
    var_b = np.var(eb)
    cov = np.cov(ea, eb)[0, 1]
    
    denominator = var_a + var_b - 2 * cov
    if denominator < 1e-12:
        return 0.5, 0.5 # Equal weight on perfect colinearity
        
    w_a = (var_b - cov) / denominator
    # Restrict weights between 0 and 1
    w_a = float(np.clip(w_a, 0.0, 1.0))
    w_b = 1.0 - w_a
    
    return w_a, w_b

_HYBRID_STATE_FILES = (
    "xgb_residual.json",
    "feature_importance.json",
    "hybrid_manifest.json",
    "hybrid_metrics.json",
)


def _read_json(path: Path) -> dict[str, Any]:
    return read_json(path, missing_ok=True)


def _clean_hybrid_state(model_dir: Path) -> list[str]:
    removed: list[str] = []
    for name in _HYBRID_STATE_FILES:
        target = model_dir / name
        if target.exists():
            target.unlink()
            removed.append(str(target))
    return removed


def _load_best_lnn_params(model_dir: Path) -> dict[str, Any]:
    runtime_ckpt = _read_json(model_dir / "optuna_runtime_checkpoint.json")
    best_params = runtime_ckpt.get("best_params", {})
    if isinstance(best_params, dict) and best_params:
        return best_params

    legacy_best = _read_json(model_dir / "optuna_best.json")
    legacy_params = legacy_best.get("best_params", {})
    if isinstance(legacy_params, dict):
        return legacy_params
    return {}


def _load_best_xgb_params(model_dir: Path) -> dict[str, Any]:
    best = _read_json(model_dir / "optuna_xgb_best.json")
    params = best.get("best_params", {})
    if isinstance(params, dict):
        return params
    return {}


def _xgb_regression_params_from_study(best_params: dict[str, Any]) -> dict[str, Any]:
    if not best_params:
        return {}

    mapped: dict[str, Any] = {
        "n_estimators": int(best_params.get("num_boost_round", 800)),
        "max_depth": int(best_params.get("max_depth", 7)),
        "learning_rate": float(best_params.get("eta", 0.025)),
        "subsample": float(best_params.get("subsample", 0.8)),
        "colsample_bytree": float(best_params.get("colsample_bytree", 0.8)),
        "min_child_weight": float(best_params.get("min_child_weight", 3.0)),
        "gamma": float(best_params.get("gamma", 0.1)),
        "reg_lambda": float(best_params.get("lambda", 1.0)),
        "reg_alpha": float(best_params.get("alpha", 1e-3)),
        "early_stopping_rounds": int(best_params.get("early_stopping_rounds", 50)),
        "objective": "reg:squarederror",
    }
    return mapped


def _resolve_hybrid_config(
    args: argparse.Namespace,
    checkpoint: dict[str, Any],
    lnn_best_params: dict[str, Any],
    xgb_best_params: dict[str, Any],
) -> dict[str, Any]:
    # LNN checkpoint takes precedence to guarantee architecture compatibility.
    seq_len = int(checkpoint.get("seq_len", args.seq_len))
    horizon = int(checkpoint.get("horizon", args.horizon))
    resample_freq = str(checkpoint.get("resample_freq", args.resample_freq))

    if args.use_best_studies:
        seq_len = int(lnn_best_params.get("seq_len", seq_len))
        horizon = int(lnn_best_params.get("horizon", horizon))

    ckpt_seq = int(checkpoint.get("seq_len", seq_len))
    ckpt_horizon = int(checkpoint.get("horizon", horizon))
    if (seq_len, horizon) != (ckpt_seq, ckpt_horizon):
        logger.warning(
            "Requested seq_len/horizon (%s/%s) differs from LNN checkpoint (%s/%s). "
            "Using checkpoint-compatible values.",
            seq_len,
            horizon,
            ckpt_seq,
            ckpt_horizon,
        )
        seq_len = ckpt_seq
        horizon = ckpt_horizon

    train_ratio = float(args.train_ratio)
    if args.use_best_studies:
        train_ratio = float(xgb_best_params.get("train_ratio", train_ratio))
    train_ratio = float(np.clip(train_ratio, 0.5, 0.95))

    xgb_params = _xgb_regression_params_from_study(xgb_best_params) if args.use_best_studies else {}

    return {
        "seq_len": seq_len,
        "horizon": horizon,
        "resample_freq": resample_freq,
        "train_ratio": train_ratio,
        "xgb_params": xgb_params,
    }








def run_hybrid_training(args: argparse.Namespace) -> dict:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"

    removed_state_files: list[str] = []
    if args.clean_state:
        removed_state_files = _clean_hybrid_state(model_dir)
        if removed_state_files:
            logger.info("Cleared prior hybrid state artifacts: %s", ", ".join(removed_state_files))

    device = pick_device(force_cpu=args.cpu)
    checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)
    lnn_best_params = _load_best_lnn_params(model_dir)
    xgb_best_params = _load_best_xgb_params(model_dir)

    resolved = _resolve_hybrid_config(args, checkpoint, lnn_best_params, xgb_best_params)

    dataset_path = resolve_data_file(args.data_path)
    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=resolved["resample_freq"])
    bundle = build_supervised_data(
        sampled,
        seq_len=int(resolved["seq_len"]),
        horizon=int(resolved["horizon"]),
        include_stoch_rsi=True,
    )

    scaler = joblib.load(model_dir / "lnn_scaler.joblib")

    split = prepare_split(bundle, train_ratio=float(resolved["train_ratio"]))
    # Re-apply saved scaler from LNN training (overrides the new scaler)
    x_seq_train_scaled = scale_sequences(scaler, bundle.sequences[: len(split.y_train)])
    x_seq_val_scaled = scale_sequences(scaler, bundle.sequences[len(split.y_train) :])
    x_xgb_train_scaled = scaler.transform(bundle.xgb_features[: len(split.y_train)])
    x_xgb_val_scaled = scaler.transform(bundle.xgb_features[len(split.y_train) :])
    y_train = split.y_train
    y_val = split.y_val

    lnn = MarketLNN(
        **checkpoint_model_kwargs(checkpoint),
    ).to(device)
    lnn.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))

    lnn_train_pred, lnn_train_latent = predict_lnn_batched(lnn, x_seq_train_scaled, device, return_latent=True)
    lnn_val_pred, lnn_val_latent = predict_lnn_batched(lnn, x_seq_val_scaled, device, return_latent=True)

    # Dimensionality check for linter
    assert lnn_train_latent is not None and lnn_val_latent is not None
    
    # -----------------------------------------------------------------------
    # PHASE 3.1: Latent Space Feature Transfer
    # Splice LNN's dynamic temporal state directly into XGBoost's feature matrix
    # -----------------------------------------------------------------------
    x_xgb_train_hybrid = np.concatenate([x_xgb_train_scaled, lnn_train_latent], axis=1)
    x_xgb_val_hybrid = np.concatenate([x_xgb_val_scaled, lnn_val_latent], axis=1)
    
    # Dynamically expand feature names to include latent dimensions
    latent_dim = lnn_train_latent.shape[1]
    hybrid_feature_names = bundle.feature_columns + [f"lnn_latent_{i}" for i in range(latent_dim)]

    residual_train = y_train - lnn_train_pred
    residual_val = y_val - lnn_val_pred
    xgb_model = train_xgb(
        x_xgb_train_hybrid,
        residual_train,
        x_xgb_val_hybrid,
        residual_val,
        params=resolved["xgb_params"],
        use_gpu=(device.type == "cuda"),
        feature_names=hybrid_feature_names,
    )

    residual_pred = xgb_model.predict(x_xgb_val_hybrid)
    
    # -----------------------------------------------------------------------
    # PHASE 3.2: Bates-Granger Mathematical Orchestration & Regime Blending
    # -----------------------------------------------------------------------
    
    # XGBoost outputs are residuals. The actual XGBoost complete prediction is LNN + Residual
    xgb_full_pred = lnn_val_pred + residual_pred
    
    # Calculate OOS errors
    lnn_err = y_val - lnn_val_pred
    xgb_err = y_val - xgb_full_pred
    
    # 1. Base Optimal Variance-Covariance weights (Float64 precision)
    w_lnn_optimal, w_xgb_optimal = _bates_granger_optimal_weights(lnn_err, xgb_err)
    
    # 2. Extract Hurst Exponent (Structural Regime Detector) from recent target window
    # E.g. last 100 validation points to gauge current regime
    hurst_lookback = min(len(y_val), 100)
    h_exponent = _compute_hurst_exponent(y_val[-hurst_lookback:])
    
    # 3. Dynamic Regime Shift
    # Normalize structural exponent around 0.5 threshold. 
    # High H (>0.5) strongly biases towards the LNN (trend follower)
    # Low H (<0.5) strongly biases towards XGBoost (mean-reversion / rigid splits)
    regime_shift = (h_exponent - 0.5) * 2.0 # Maps [0.0, 1.0] -> [-1.0, 1.0]
    
    # Shift the optimal weights, clipped securely [0,1]
    w_lnn_final = np.clip(w_lnn_optimal + regime_shift, 0.0, 1.0)
    w_xgb_final = 1.0 - w_lnn_final
    
    hybrid_pred = (lnn_val_pred * w_lnn_final) + (xgb_full_pred * w_xgb_final)

    metrics = {
        "hybrid_val_mse": float(mean_squared_error(y_val, hybrid_pred)),
        "hybrid_val_mae": float(mean_absolute_error(y_val, hybrid_pred)),
        "hybrid_val_directional_accuracy": directional_accuracy(y_val, hybrid_pred),
        "lnn_only_val_mse": float(mean_squared_error(y_val, lnn_val_pred)),
        "lnn_only_val_directional_accuracy": directional_accuracy(y_val, lnn_val_pred),
        "bates_granger_w_lnn": float(w_lnn_optimal),
        "hurst_exponent": float(h_exponent),
        "final_w_lnn": float(w_lnn_final),
        "final_w_xgb": float(w_xgb_final),
        "residual_reduction_pct": float(
            (mean_squared_error(y_val, lnn_val_pred) - mean_squared_error(y_val, hybrid_pred))
            / (mean_squared_error(y_val, lnn_val_pred) + 1e-12)
            * 100
        ),
        "hybrid_train_ratio": float(resolved["train_ratio"]),
        "used_best_studies": bool(args.use_best_studies),
        "clean_state": bool(args.clean_state),
        "xgb_best_params_applied": bool(resolved["xgb_params"]),
        "state_files_removed": removed_state_files,
        "device": device.type,
        "dataset": str(dataset_path),
    }

    save_xgb(xgb_model, model_dir / "xgb_residual.json")

    # Save feature importance
    fi = get_feature_importance(xgb_model, bundle.feature_columns)
    write_json(model_dir / "feature_importance.json", fi)

    write_json(
        model_dir / "hybrid_manifest.json",
        {
            "lnn_checkpoint": str(model_dir / "lnn_best.pt"),
            "lnn_scaler": str(model_dir / "lnn_scaler.joblib"),
            "xgb_model": str(model_dir / "xgb_residual.json"),
            "seq_len": int(resolved["seq_len"]),
            "horizon": int(resolved["horizon"]),
            "resample_freq": str(resolved["resample_freq"]),
            "train_ratio": float(resolved["train_ratio"]),
            "used_best_studies": bool(args.use_best_studies),
            "xgb_best_params_applied": bool(resolved["xgb_params"]),
            "feature_columns": bundle.feature_columns,
        },
    )
    write_json(model_dir / "hybrid_metrics.json", metrics)
    logger.info("Hybrid training complete - MSE reduction: %.1f%%", metrics["residual_reduction_pct"])
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hybrid residual model")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--resample-freq", type=str, default="1min")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--use-best-studies", action="store_true", default=True)
    parser.add_argument("--no-best-studies", dest="use_best_studies", action="store_false")
    parser.add_argument("--clean-state", action="store_true", default=True)
    parser.add_argument("--no-clean-state", dest="clean_state", action="store_false")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(run_hybrid_training(args))


if __name__ == "__main__":
    main()
