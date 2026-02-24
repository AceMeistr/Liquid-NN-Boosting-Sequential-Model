from __future__ import annotations

import argparse
import torch

from modelmk1.common.paths import get_app_paths
from modelmk1.common.runtime import write_json
from modelmk1.features.indicators import get_feature_schema
from modelmk1.models.lnn_model import MarketLNN


def run_build(
    input_size: int | None,
    hidden_size: int,
    dropout: float,
    seq_len: int,
    horizon: int,
    resample_freq: str,
) -> dict:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_size = input_size if input_size is not None else len(get_feature_schema(include_stoch_rsi=True))

    model = MarketLNN(input_size=resolved_input_size, hidden_size=hidden_size, output_size=1, dropout=dropout)
    init_path = model_dir / "lnn_init.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": resolved_input_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "seq_len": seq_len,
            "horizon": horizon,
            "resample_freq": resample_freq,
            "status": "initialized_not_trained",
        },
        init_path,
    )

    write_json(
        model_dir / "xgb_template.json",
        {
            "n_estimators": 600,
            "max_depth": 6,
            "learning_rate": 0.03,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1e-3,
            "reg_lambda": 1.0,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "status": "template_not_trained",
        },
    )

    payload = {
        "phase": "models_built_training_on_hold",
        "lnn_init_checkpoint": str(init_path),
        "xgb_template": str(model_dir / "xgb_template.json"),
        "requires_real_data": True,
    }
    write_json(model_dir / "build_manifest.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Build untrained ModelMK1 artifacts")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--horizon", type=int, default=15)
    parser.add_argument("--resample-freq", type=str, default="1min")
    args = parser.parse_args()
    print(
        run_build(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            seq_len=args.seq_len,
            horizon=args.horizon,
            resample_freq=args.resample_freq,
        )
    )


if __name__ == "__main__":
    main()
