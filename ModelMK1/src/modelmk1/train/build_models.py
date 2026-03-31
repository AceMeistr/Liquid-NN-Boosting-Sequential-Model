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
    num_heads: int = 4,
    num_layers: int = 2,
    use_attention: bool = True,
    backbone_type: str = "mamba",
    cnn_frontend: str = "inception",
    mamba_d_state: int = 16,
    mamba_d_conv: int = 4,
    mamba_expand: int = 2,
) -> dict:
    paths = get_app_paths()
    model_dir = paths.outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    resolved_input_size = input_size if input_size is not None else len(get_feature_schema(include_stoch_rsi=True))

    model = MarketLNN(
        input_size=resolved_input_size,
        hidden_size=hidden_size,
        output_size=1,
        dropout=dropout,
        num_heads=num_heads,
        num_layers=num_layers,
        use_attention=use_attention,
        backbone_type=backbone_type,
        cnn_frontend=cnn_frontend,
        mamba_d_state=mamba_d_state,
        mamba_d_conv=mamba_d_conv,
        mamba_expand=mamba_expand,
    )
    init_path = model_dir / "lnn_init.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_size": resolved_input_size,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "use_attention": use_attention,
                    "backbone_type": backbone_type,
                    "cnn_frontend": cnn_frontend,
                    "mamba_d_state": mamba_d_state,
                    "mamba_d_conv": mamba_d_conv,
                    "mamba_expand": mamba_expand,
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
            "n_estimators": 800,
            "max_depth": 7,
            "learning_rate": 0.025,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 1e-3,
            "reg_lambda": 1.0,
            "min_child_weight": 3,
            "gamma": 0.1,
            "tree_method": "hist",
            "device": "cuda",
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
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--use-attention", action="store_true", default=False)
    parser.add_argument("--backbone-type", type=str, default="mamba", choices=["mamba", "attention_gru"])
    parser.add_argument("--cnn-frontend", type=str, default="inception", choices=["inception", "none"])
    parser.add_argument("--mamba-d-state", type=int, default=16)
    parser.add_argument("--mamba-d-conv", type=int, default=4)
    parser.add_argument("--mamba-expand", type=int, default=2)
    args = parser.parse_args()
    print(
        run_build(
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            dropout=args.dropout,
            seq_len=args.seq_len,
            horizon=args.horizon,
            resample_freq=args.resample_freq,
            num_heads=args.num_heads,
            num_layers=args.num_layers,
            use_attention=args.use_attention,
            backbone_type=args.backbone_type,
            cnn_frontend=args.cnn_frontend,
            mamba_d_state=args.mamba_d_state,
            mamba_d_conv=args.mamba_d_conv,
            mamba_expand=args.mamba_expand,
        )
    )


if __name__ == "__main__":
    main()
