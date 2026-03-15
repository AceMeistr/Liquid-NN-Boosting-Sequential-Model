from __future__ import annotations

import json
import math
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, safe_torch_load, read_json
from modelmk1.eval.backtest import sharpe_ratio
from modelmk1.data.loader import (
    build_supervised_data,
    load_tick_df,
    prepare_split,
    resample_ticks,
    scale_sequences,
)
from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
from modelmk1.models.lnn_model import MarketLNN
from modelmk1.models.xgb_model import load_xgb
from modelmk1.models.xgb_spectral import (
    _build_lnn_feature_bundle,
    _scaled_selected_indicators,
    load_constituent_price_panel,
)
from modelmk1.train.train_hybrid import _predict_lnn

ANNUALIZATION_1MIN = 252.0 * 375.0
LOW_VARIANCE_THRESHOLD = 1e-6





def _read_json(path: Path) -> dict:
    return read_json(path, missing_ok=True)


def main() -> None:
    model_dir = get_app_paths().outputs / "model"

    lnn_retrain = _read_json(model_dir / "lnn_trial18_retrain_report.json")
    xgb_retrain = _read_json(model_dir / "xgb_best_retrain_report.json")
    hybrid_metrics = _read_json(model_dir / "hybrid_metrics.json")
    hybrid_manifest = _read_json(model_dir / "hybrid_manifest.json")
    xgb_best = _read_json(model_dir / "optuna_xgb_best.json")

    data_path = resolve_data_file(None)

    # Validate variance of normalized indicators used for spectral correlation.
    frame = load_constituent_price_panel(data_path)
    best_xgb_params = xgb_best["best_params"]
    lnn_bundle = _build_lnn_feature_bundle(
        frame,
        seq_len=int(best_xgb_params["seq_len"]),
        horizon=int(best_xgb_params["horizon"]),
    )
    scaled_indicators, indicator_names = _scaled_selected_indicators(
        lnn_bundle,
        top_n=int(best_xgb_params["top_n"]),
        indicator_names=None,
        robust_quantile_low=float(best_xgb_params["robust_q_low"]),
        robust_quantile_high=float(best_xgb_params["robust_q_high"]),
    )
    indicator_std = np.std(scaled_indicators, axis=0)
    low_var_indices = [idx for idx, value in enumerate(indicator_std) if float(value) <= LOW_VARIANCE_THRESHOLD]

    # Recompute hybrid predictions to derive Sharpe alongside MSE/MAE.
    sampled = resample_ticks(load_tick_df(str(data_path)), freq=str(hybrid_manifest["resample_freq"]))
    bundle = build_supervised_data(
        sampled,
        seq_len=int(hybrid_manifest["seq_len"]),
        horizon=int(hybrid_manifest["horizon"]),
        include_stoch_rsi=True,
    )
    split = prepare_split(bundle, train_ratio=float(hybrid_manifest["train_ratio"]))

    scaler = joblib.load(model_dir / "lnn_scaler.joblib")
    x_seq_val_scaled = scale_sequences(scaler, bundle.sequences[len(split.y_train) :])
    x_xgb_val_scaled = scaler.transform(bundle.xgb_features[len(split.y_train) :])
    y_val = split.y_val.astype(np.float64)

    device = pick_device(force_cpu=False)
    checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)

    lnn_model = MarketLNN(**checkpoint_model_kwargs(checkpoint)).to(device)
    lnn_model.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))
    lnn_val_pred = _predict_lnn(lnn_model, x_seq_val_scaled, device).astype(np.float64)

    xgb_model = load_xgb(model_dir / "xgb_residual.json")
    residual_pred = xgb_model.predict(x_xgb_val_scaled).astype(np.float64)
    hybrid_pred = lnn_val_pred + residual_pred

    lnn_strategy_returns = np.where(lnn_val_pred >= 0.0, 1.0, -1.0) * y_val
    hybrid_strategy_returns = np.where(hybrid_pred >= 0.0, 1.0, -1.0) * y_val

    xgb_metrics = xgb_retrain["metrics"]
    xgb_best_iter = xgb_metrics.get("best_iteration")
    xgb_val_logloss = None
    if isinstance(xgb_best_iter, int):
        logloss_curve = xgb_metrics.get("eval_history", {}).get("val", {}).get("logloss", [])
        if 0 <= xgb_best_iter < len(logloss_curve):
            xgb_val_logloss = float(logloss_curve[xgb_best_iter])

    report = {
        "context": {
            "requested_lnn_trial": 18,
            "xgb_source": "optuna_xgb_best",
            "dataset": str(data_path),
        },
        "variance_check": {
            "normalized_feature_count": int(scaled_indicators.shape[1]),
            "normalized_std_min": float(indicator_std.min()),
            "normalized_std_max": float(indicator_std.max()),
            "low_variance_threshold": LOW_VARIANCE_THRESHOLD,
            "low_variance_count": int(len(low_var_indices)),
            "low_variance_features": [indicator_names[idx] for idx in low_var_indices],
        },
        "lnn_trial18": {
            "loss": float(lnn_retrain["metrics"]["best_val_loss"]),
            "mse": float(lnn_retrain["metrics"]["val_mse"]),
            "mae": float(lnn_retrain["metrics"]["val_mae"]),
            "sharpe": float(lnn_retrain["metrics"].get("cpcv", {}).get("sharpe_mean", 0.0)),
            "directional_accuracy": float(lnn_retrain["metrics"]["val_directional_accuracy"]),
        },
        "xgb_best": {
            "loss_logloss": xgb_val_logloss,
            "mse": float(xgb_metrics["val_mse"]),
            "mae": float(xgb_metrics["val_mae"]),
            "sharpe": float(xgb_metrics["val_sharpe_ratio"]),
            "directional_accuracy": float(xgb_metrics["val_directional_accuracy"]),
            "decision_threshold": float(xgb_metrics["decision_threshold"]),
        },
        "hybrid_best_combo": {
            "loss_mse": float(mean_squared_error(y_val, hybrid_pred)),
            "mse": float(mean_squared_error(y_val, hybrid_pred)),
            "mae": float(mean_absolute_error(y_val, hybrid_pred)),
            "sharpe": float(sharpe_ratio(hybrid_strategy_returns, annualization=ANNUALIZATION_1MIN)),
            "directional_accuracy": float((np.sign(hybrid_pred) == np.sign(y_val)).mean()),
            "lnn_only_sharpe": float(sharpe_ratio(lnn_strategy_returns, annualization=ANNUALIZATION_1MIN)),
            "residual_reduction_pct": float(hybrid_metrics.get("residual_reduction_pct", 0.0)),
        },
    }

    out_path = model_dir / "final_report_trial18_xgbbest_hybrid.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps({
        "report_path": str(out_path),
        "lnn_loss": report["lnn_trial18"]["loss"],
        "lnn_sharpe": report["lnn_trial18"]["sharpe"],
        "xgb_sharpe": report["xgb_best"]["sharpe"],
        "hybrid_mse": report["hybrid_best_combo"]["mse"],
        "hybrid_mae": report["hybrid_best_combo"]["mae"],
        "hybrid_sharpe": report["hybrid_best_combo"]["sharpe"],
        "low_variance_count": report["variance_check"]["low_variance_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
