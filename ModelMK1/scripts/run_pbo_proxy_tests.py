from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch
import xgboost as xgb

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, safe_torch_load
from modelmk1.data.loader import (
    build_supervised_data,
    load_tick_df,
    prepare_split,
    resample_ticks,
    scale_sequences,
)
from modelmk1.eval.cpcv import CPCVConfig, evaluate_cpcv_distribution
from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
from modelmk1.models.lnn_model import MarketLNN, predict_lnn_batched
from modelmk1.models.xgb_model import load_xgb
from modelmk1.models.xgb_spectral import build_spectral_training_matrices, load_constituent_price_panel


def _read_json(path: Path) -> dict:
    return read_json(path, missing_ok=True)



def _sharpe_from_signal(y_true: np.ndarray, y_pred: np.ndarray, annualization: float = 252.0) -> float:
    pnl = np.sign(y_pred) * y_true
    sigma = float(np.std(pnl, ddof=1)) if len(pnl) > 1 else 0.0
    if sigma <= 1e-12:
        return 0.0
    return float(np.mean(pnl) / sigma * math.sqrt(annualization))


def _base_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "samples": int(len(y_true)),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "mae": float(np.mean(np.abs(y_true - y_pred))),
        "directional_accuracy": float((np.sign(y_true) == np.sign(y_pred)).mean()),
        "sharpe_proxy": _sharpe_from_signal(y_true, y_pred),
    }


def _cpcv_config(horizon: int) -> CPCVConfig:
    return CPCVConfig(
        n_groups=16,
        test_group_size=8,
        max_combinations=200,
        purge_gap=max(2 * int(horizon), 2),
        embargo=max(int(horizon), 1),
        seed=42,
    )


def _evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, horizon: int) -> dict:
    cpcv = evaluate_cpcv_distribution(y_true, y_pred, config=_cpcv_config(horizon))
    return {
        **_base_metrics(y_true, y_pred),
        "horizon": int(horizon),
        "cpcv": cpcv,
        "pbo_proxy": float(cpcv.get("pbo_proxy", 1.0)),
    }


def _lnn_eval(model_dir: Path, dataset_path: Path) -> dict:
    device = pick_device(force_cpu=False)
    checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)

    seq_len = int(checkpoint.get("seq_len", 64))
    horizon = int(checkpoint.get("horizon", 15))
    resample_freq = str(checkpoint.get("resample_freq", "1min"))

    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=resample_freq)
    bundle = build_supervised_data(sampled, seq_len=seq_len, horizon=horizon, include_stoch_rsi=True)
    split = prepare_split(bundle, train_ratio=0.8)

    scaler = joblib.load(model_dir / "lnn_scaler.joblib")
    x_seq_val_scaled = scale_sequences(scaler, bundle.sequences[len(split.y_train) :])

    lnn = MarketLNN(**checkpoint_model_kwargs(checkpoint)).to(device)
    lnn.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))

    y_pred = predict_lnn_batched(lnn, x_seq_val_scaled, device)
    y_true = split.y_val.astype(np.float64)
    return _evaluate_model(y_true, y_pred, horizon=horizon)


def _xgb_spectral_eval(model_dir: Path, dataset_path: Path) -> dict:
    report = _read_json(model_dir / "xgb_best_retrain_report.json")
    params = report.get("best_params", {})
    metrics = report.get("metrics", {})

    top_n = int(params["top_n"])
    corr_window = int(params["corr_window"])
    horizon = int(params["horizon"])
    seq_len = int(params["seq_len"])
    train_ratio = float(params["train_ratio"])
    robust_q_low = float(params["robust_q_low"])
    robust_q_high = float(params["robust_q_high"])

    market_frame = load_constituent_price_panel(dataset_path)
    dataset = build_spectral_training_matrices(
        market_frame,
        top_n=top_n,
        corr_window=corr_window,
        horizon=horizon,
        seq_len=seq_len,
        tickers=metrics.get("selected_indicators"),
        robust_quantile_low=robust_q_low,
        robust_quantile_high=robust_q_high,
    )

    split = int(len(dataset.targets) * train_ratio)
    split = max(25, min(split, len(dataset.targets) - 10))

    x_val = dataset.features[split:]
    y_true = dataset.targets[split:].astype(np.float64)

    model_path = Path(metrics["xgb_model"])
    booster = xgb.Booster()
    booster.load_model(str(model_path))

    dval = xgb.DMatrix(x_val, feature_names=dataset.feature_names)
    prob = booster.predict(dval)

    threshold = float(metrics.get("decision_threshold", 0.5))
    y_pred = np.where(prob >= threshold, 1.0, -1.0).astype(np.float64)
    return _evaluate_model(y_true, y_pred, horizon=horizon)


def _hybrid_eval(model_dir: Path, dataset_path: Path) -> dict:
    device = pick_device(force_cpu=False)
    hybrid_manifest = _read_json(model_dir / "hybrid_manifest.json")

    seq_len = int(hybrid_manifest["seq_len"])
    horizon = int(hybrid_manifest["horizon"])
    resample_freq = str(hybrid_manifest["resample_freq"])
    train_ratio = float(hybrid_manifest["train_ratio"])

    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=resample_freq)
    bundle = build_supervised_data(sampled, seq_len=seq_len, horizon=horizon, include_stoch_rsi=True)
    split = prepare_split(bundle, train_ratio=train_ratio)

    scaler = joblib.load(model_dir / "lnn_scaler.joblib")
    x_seq_val_scaled = scale_sequences(scaler, bundle.sequences[len(split.y_train) :])
    x_xgb_val_scaled = scaler.transform(bundle.xgb_features[len(split.y_train) :])

    checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)
    lnn = MarketLNN(**checkpoint_model_kwargs(checkpoint)).to(device)
    lnn.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))

    lnn_pred = predict_lnn_batched(lnn, x_seq_val_scaled, device)
    residual_model = load_xgb(model_dir / "xgb_residual.json")
    residual_pred = residual_model.predict(x_xgb_val_scaled).astype(np.float64)

    y_pred = lnn_pred + residual_pred
    y_true = split.y_val.astype(np.float64)
    return _evaluate_model(y_true, y_pred, horizon=horizon)


def main() -> None:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    dataset_path = Path(resolve_data_file(None))

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(dataset_path),
        "models": {
            "lnn": _lnn_eval(model_dir, dataset_path),
            "xgboost_spectral": _xgb_spectral_eval(model_dir, dataset_path),
            "hybrid": _hybrid_eval(model_dir, dataset_path),
        },
    }

    items = list(report["models"].items())
    best_pbo_model = min(items, key=lambda item: float(item[1].get("pbo_proxy", float("inf"))))
    lowest_name: str = best_pbo_model[0]
    lowest_val: float = float(best_pbo_model[1].get("pbo_proxy", float("inf")))

    report["summary"] = {
        "lowest_pbo_proxy_model": lowest_name,
        "lowest_pbo_proxy": lowest_val,
    }

    out_json = model_dir / "pbo_proxy_report.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    out_md = model_dir / "pbo_proxy_report.md"
    md_lines = [
        "# PBO Proxy Test Report",
        "",
        f"Generated UTC: {report['generated_at_utc']}",
        f"Dataset: {report['dataset']}",
        "",
        "## Models",
        "",
    ]
    for name, metrics in report["models"].items():
        md_lines.extend(
            [
                f"### {name}",
                f"- PBO proxy: {metrics['pbo_proxy']}",
                f"- CPCV Sharpe mean: {metrics['cpcv'].get('sharpe_mean', 0.0)}",
                (
                    "- CPCV Sharpe CI95: "
                    f"[{metrics['cpcv'].get('sharpe_ci95_low', 0.0)}, {metrics['cpcv'].get('sharpe_ci95_high', 0.0)}]"
                ),
                f"- Directional accuracy mean (CPCV): {metrics['cpcv'].get('directional_accuracy_mean', 0.0)}",
                f"- Samples: {metrics['samples']}",
                "",
            ]
        )
    md_lines.extend(
        [
            "## Summary",
            f"- Lowest PBO proxy model: {report['summary']['lowest_pbo_proxy_model']}",
            f"- Lowest PBO proxy value: {report['summary']['lowest_pbo_proxy']}",
            "",
        ]
    )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(json.dumps({
        "checkpoint": str(model_dir / "optuna_lnn_oomsafe_pause_checkpoint.json"),
        "pbo_report_json": str(out_json),
        "pbo_report_md": str(out_md),
        "summary": report["summary"],
    }, indent=2))


if __name__ == "__main__":
    main()
