from __future__ import annotations

import argparse

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, safe_torch_load, write_json
from modelmk1.data.loader import build_supervised_data, load_tick_df, resample_ticks
from modelmk1.models.lnn_model import MarketLNN
from modelmk1.models.xgb_model import save_xgb, train_xgb


@torch.no_grad()
def _predict_lnn(model: MarketLNN, x_seq: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    tensor = torch.tensor(x_seq, dtype=torch.float32, device=device)
    return model(tensor).detach().cpu().numpy().reshape(-1)


def run_hybrid_training(args: argparse.Namespace) -> dict:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"

    dataset_path = resolve_data_file(args.data_path)
    df = load_tick_df(str(dataset_path))
    sampled = resample_ticks(df, freq=args.resample_freq)
    bundle = build_supervised_data(sampled, seq_len=args.seq_len, horizon=args.horizon, include_stoch_rsi=True)

    split = int(0.8 * len(bundle.targets))
    x_seq_train, x_seq_val = bundle.sequences[:split], bundle.sequences[split:]
    x_xgb_train, x_xgb_val = bundle.xgb_features[:split], bundle.xgb_features[split:]
    y_train, y_val = bundle.targets[:split], bundle.targets[split:]

    device = pick_device(force_cpu=args.cpu)
    checkpoint = safe_torch_load(model_dir / "lnn_best.pt", map_location=device)
    scaler = joblib.load(model_dir / "lnn_scaler.joblib")

    x_seq_train_scaled = scaler.transform(x_seq_train.reshape(-1, x_seq_train.shape[-1])).reshape(x_seq_train.shape)
    x_seq_val_scaled = scaler.transform(x_seq_val.reshape(-1, x_seq_val.shape[-1])).reshape(x_seq_val.shape)
    x_xgb_train_scaled = scaler.transform(x_xgb_train)
    x_xgb_val_scaled = scaler.transform(x_xgb_val)

    lnn = MarketLNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=1,
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    lnn.load_state_dict(checkpoint["model_state_dict"])

    lnn_train_pred = _predict_lnn(lnn, x_seq_train_scaled, device)
    lnn_val_pred = _predict_lnn(lnn, x_seq_val_scaled, device)

    residual_train = y_train - lnn_train_pred
    residual_val = y_val - lnn_val_pred
    xgb = train_xgb(
        x_xgb_train_scaled,
        residual_train,
        x_xgb_val_scaled,
        residual_val,
        use_gpu=(device.type == "cuda"),
    )

    residual_pred = xgb.predict(x_xgb_val_scaled)
    hybrid_pred = lnn_val_pred + residual_pred
    metrics = {
        "hybrid_val_mse": float(mean_squared_error(y_val, hybrid_pred)),
        "hybrid_val_mae": float(mean_absolute_error(y_val, hybrid_pred)),
        "lnn_only_val_mse": float(mean_squared_error(y_val, lnn_val_pred)),
        "device": device.type,
        "dataset": str(dataset_path),
    }

    save_xgb(xgb, model_dir / "xgb_residual.json")
    write_json(
        model_dir / "hybrid_manifest.json",
        {
            "lnn_checkpoint": str(model_dir / "lnn_best.pt"),
            "lnn_scaler": str(model_dir / "lnn_scaler.joblib"),
            "xgb_model": str(model_dir / "xgb_residual.json"),
            "seq_len": args.seq_len,
            "horizon": args.horizon,
            "resample_freq": args.resample_freq,
            "feature_columns": bundle.feature_columns,
        },
    )
    write_json(model_dir / "hybrid_metrics.json", metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train hybrid residual model")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--resample-freq", type=str, default="1min")
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print(run_hybrid_training(args))


if __name__ == "__main__":
    main()
