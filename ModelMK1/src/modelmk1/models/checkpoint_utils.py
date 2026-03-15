from __future__ import annotations

from typing import Any


def is_legacy_checkpoint(checkpoint: dict[str, Any]) -> bool:
    return "backbone_type" not in checkpoint


def checkpoint_model_kwargs(checkpoint: dict[str, Any]) -> dict[str, Any]:
    legacy = is_legacy_checkpoint(checkpoint)
    kwargs: dict[str, Any] = {
        "input_size": int(checkpoint["input_size"]),
        "hidden_size": int(checkpoint["hidden_size"]),
        "output_size": 1,
        "dropout": float(checkpoint.get("dropout", 0.1)),
        "num_heads": int(checkpoint.get("num_heads", 4)),
        "num_layers": int(checkpoint.get("num_layers", 2)),
        "use_attention": bool(checkpoint.get("use_attention", True)),
    }

    if legacy:
        kwargs.update(
            {
                "backbone_type": "attention_gru",
                "cnn_frontend": "none",
                "mamba_d_state": 16,
                "mamba_d_conv": 4,
                "mamba_expand": 2,
            }
        )
    else:
        kwargs.update(
            {
                "backbone_type": str(checkpoint.get("backbone_type", "mamba")),
                "cnn_frontend": str(checkpoint.get("cnn_frontend", "inception")),
                "mamba_d_state": int(checkpoint.get("mamba_d_state", 16)),
                "mamba_d_conv": int(checkpoint.get("mamba_d_conv", 4)),
                "mamba_expand": int(checkpoint.get("mamba_expand", 2)),
            }
        )

    return kwargs
