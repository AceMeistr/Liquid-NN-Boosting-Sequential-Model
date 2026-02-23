from __future__ import annotations

import argparse
import json

import joblib
import numpy as np
import pandas as pd
import torch

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device
from modelmk1.data.loader import build_supervised_data, load_tick_df, resample_ticks
from modelmk1.models.lnn_model import MarketLNN
from modelmk1.models.xgb_model import load_xgb


@torch.no_grad()
def predict_batch(x_seq: np.ndarray, x_xgb: np.ndarray, model: MarketLNN, xgb, device: torch.device) -> np.ndarray:
    tensor = torch.tensor(x_seq, dtype=torch.float32, device=device)
    base = model(tensor).detach().cpu().numpy().reshape(-1)
    residual = xgb.predict(x_xgb)
    return base + residual


def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid prediction")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    manifest = json.loads((model_dir / "hybrid_manifest.json").read_text(encoding="utf-8"))

    device = pick_device(force_cpu=args.cpu)
    checkpoint = torch.load(manifest["lnn_checkpoint"], map_location=device)
    scaler = joblib.load(manifest["lnn_scaler"])
    xgb = load_xgb(manifest["xgb_model"])

    model = MarketLNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=1,
        dropout=checkpoint.get("dropout", 0.1),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_file = resolve_data_file(args.data_path)
    df = load_tick_df(str(data_file))
    sampled = resample_ticks(df, freq=manifest.get("resample_freq", "1min"))
    bundle = build_supervised_data(sampled, seq_len=int(manifest["seq_len"]), horizon=int(manifest["horizon"]), include_stoch_rsi=True)

    x_seq_scaled = scaler.transform(bundle.sequences.reshape(-1, bundle.sequences.shape[-1])).reshape(bundle.sequences.shape)
    x_xgb_scaled = scaler.transform(bundle.xgb_features)
    preds = predict_batch(x_seq_scaled, x_xgb_scaled, model, xgb, device)

    out = pd.DataFrame({"prediction": preds, "target": bundle.targets, "error": bundle.targets - preds})
    out_path = paths.outputs / "predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
