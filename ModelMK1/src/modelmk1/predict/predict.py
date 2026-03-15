from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import RobustScaler, StandardScaler

from modelmk1.eval.inference_speed import build_pipeline_speed_report
from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, read_json, safe_torch_load, write_json
from modelmk1.data.loader import build_supervised_data, load_tick_df, resample_ticks, scale_sequences
from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
from modelmk1.models.lnn_model import MarketLNN
from modelmk1.models.xgb_model import load_xgb

logger = logging.getLogger(__name__)


@torch.no_grad()
def predict_batch(
    x_seq: np.ndarray,
    x_xgb: np.ndarray,
    model: MarketLNN,
    xgb_model,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Batched hybrid prediction (LNN + XGBoost residual)."""
    model.eval()
    lnn_preds = np.empty(len(x_seq), dtype=np.float64)
    for start in range(0, len(x_seq), batch_size):
        end = min(start + batch_size, len(x_seq))
        chunk = torch.tensor(x_seq[start:end], dtype=torch.float32, device=device)
        out = model(chunk).detach().cpu().numpy().reshape(-1)
        lnn_preds[start:end] = out
        del chunk

    base = lnn_preds
    residual = xgb_model.predict(x_xgb)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return base + residual


@torch.no_grad()
def predict_with_confidence(
    x_seq: np.ndarray,
    x_xgb: np.ndarray,
    model: MarketLNN,
    xgb_model,
    device: torch.device,
    mc_samples: int = 30,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hybrid prediction with Monte Carlo Dropout uncertainty estimation.

    Returns (predictions, confidence, uncertainty_std).
    """
    if mc_samples <= 0:
        raise ValueError("mc_samples must be > 0")

    model.train()  # enable dropout for MC sampling
    residual = xgb_model.predict(x_xgb)

    mean_pred = np.zeros(len(x_seq), dtype=np.float64)
    m2 = np.zeros(len(x_seq), dtype=np.float64)

    for sample_idx in range(1, mc_samples + 1):
        lnn_preds = np.empty(len(x_seq), dtype=np.float64)
        for start in range(0, len(x_seq), batch_size):
            end = min(start + batch_size, len(x_seq))
            chunk = torch.tensor(x_seq[start:end], dtype=torch.float32, device=device)
            out = model(chunk).detach().cpu().numpy().reshape(-1)
            lnn_preds[start:end] = out
            del chunk

        sample_pred = lnn_preds + residual
        delta = sample_pred - mean_pred
        mean_pred = mean_pred + (delta / sample_idx)
        delta2 = sample_pred - mean_pred
        m2 = m2 + (delta * delta2)

    model.eval()
    variance = m2 / max(mc_samples - 1, 1)
    std_pred = np.sqrt(np.clip(variance, 0.0, None))
    confidence = 1.0 / (1.0 + std_pred)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return mean_pred, confidence, std_pred


def _load_hybrid_model(
    model_dir: Path, device: torch.device,
) -> tuple[MarketLNN, object, StandardScaler | RobustScaler, dict]:
    """Load all hybrid model components from manifest."""
    manifest = read_json(model_dir / "hybrid_manifest.json")
    checkpoint = safe_torch_load(Path(manifest["lnn_checkpoint"]), map_location=device)
    scaler = cast(StandardScaler | RobustScaler, joblib.load(manifest["lnn_scaler"]))
    xgb_model = load_xgb(manifest["xgb_model"])

    model = MarketLNN(
        **checkpoint_model_kwargs(checkpoint),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))
    model.eval()

    return model, xgb_model, scaler, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid prediction")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--with-confidence", action="store_true", help="Use MC Dropout for confidence")
    parser.add_argument("--mc-samples", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    paths = get_app_paths()
    model_dir = paths.outputs / "model"

    device = pick_device(force_cpu=args.cpu)
    model, xgb_model, scaler, manifest = _load_hybrid_model(model_dir, device)

    data_file = resolve_data_file(args.data_path)
    df = load_tick_df(str(data_file))
    sampled = resample_ticks(df, freq=manifest.get("resample_freq", "1min"))
    bundle = build_supervised_data(
        sampled, seq_len=int(manifest["seq_len"]),
        horizon=int(manifest["horizon"]), include_stoch_rsi=True,
    )

    x_seq_scaled = scale_sequences(scaler, bundle.sequences)
    x_xgb_scaled = scaler.transform(bundle.xgb_features)

    start_time = time.perf_counter()

    if args.with_confidence:
        preds, confidence, uncertainty = predict_with_confidence(
            x_seq_scaled, x_xgb_scaled, model, xgb_model, device,
            mc_samples=args.mc_samples, batch_size=args.batch_size,
        )
        out = pd.DataFrame({
            "prediction": preds,
            "confidence": confidence,
            "uncertainty_std": uncertainty,
            "target": bundle.targets,
            "error": bundle.targets - preds,
            "abs_error": np.abs(bundle.targets - preds),
        })
    else:
        preds = predict_batch(
            x_seq_scaled, x_xgb_scaled, model, xgb_model, device,
            batch_size=args.batch_size,
        )
        out = pd.DataFrame({
            "prediction": preds,
            "target": bundle.targets,
            "error": bundle.targets - preds,
            "abs_error": np.abs(bundle.targets - preds),
        })

    elapsed = time.perf_counter() - start_time

    out_path = paths.outputs / "predictions.csv"
    out.to_csv(out_path, index=False)

    # Save inference metadata
    avg_latency_ms = elapsed / max(len(preds), 1) * 1000.0
    speed_budget = build_pipeline_speed_report(
        avg_model_latency_ms=float(avg_latency_ms),
        with_confidence=bool(args.with_confidence),
        mc_samples=int(args.mc_samples),
    )

    write_json(paths.outputs / "inference_meta.json", {
        "num_samples": len(preds),
        "total_inference_seconds": round(elapsed, 4),
        "avg_latency_ms": round(avg_latency_ms, 4),
        "device": device.type,
        "with_confidence": args.with_confidence,
    })
    write_json(paths.outputs / "inference_speed_budget.json", speed_budget)
    logger.info(
        "Predictions written to %s (%d samples, %.2fs, %.3fms/sample)",
        out_path, len(preds), elapsed, elapsed / max(len(preds), 1) * 1000,
    )


if __name__ == "__main__":
    main()
