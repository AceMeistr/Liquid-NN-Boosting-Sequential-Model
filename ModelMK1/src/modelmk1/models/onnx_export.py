"""ONNX export for production deployment.

Exports the LNN model to ONNX format for optimized inference using
ONNX Runtime, enabling cross-platform deployment without PyTorch.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch

from modelmk1.common.runtime import safe_torch_load
from modelmk1.models.lnn_model import MarketLNN, checkpoint_model_kwargs

logger = logging.getLogger(__name__)

_ORT_AVAILABLE = False
ort: Any = None  # type holder so the name is always bound
try:
    import onnxruntime as _ort

    ort = _ort
    _ORT_AVAILABLE = True
except ImportError:
    pass


def export_to_onnx(
    checkpoint_path: Path,
    output_path: Path,
    seq_len: int,
    device: torch.device | None = None,
) -> Path:
    """Export a trained MarketLNN checkpoint to ONNX format."""
    device = device or torch.device("cpu")
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

    model = MarketLNN(**checkpoint_model_kwargs(checkpoint))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = torch.randn(1, seq_len, checkpoint["input_size"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        (dummy,),
        str(output_path),
        input_names=["sequence"],
        output_names=["prediction"],
        dynamic_axes={
            "sequence": {0: "batch_size"},
            "prediction": {0: "batch_size"},
        },
        opset_version=17,
    )
    logger.info("ONNX model exported to %s", output_path)
    return output_path


def validate_onnx(
    onnx_path: Path,
    checkpoint_path: Path,
    seq_len: int,
    atol: float = 1e-4,
) -> bool:
    """Validate ONNX output matches PyTorch output."""
    if not _ORT_AVAILABLE:
        logger.warning("onnxruntime not available - skipping ONNX validation")
        return True

    checkpoint = safe_torch_load(checkpoint_path, map_location=torch.device("cpu"))
    model = MarketLNN(
        input_size=checkpoint["input_size"],
        hidden_size=checkpoint["hidden_size"],
        output_size=1,
        dropout=checkpoint.get("dropout", 0.1),
        num_heads=checkpoint.get("num_heads", 4),
        num_layers=checkpoint.get("num_layers", 2),
        use_attention=checkpoint.get("use_attention", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    dummy = np.random.randn(2, seq_len, checkpoint["input_size"]).astype(np.float32)

    with torch.no_grad():
        pt_out = model(torch.tensor(dummy)).numpy()

    session = ort.InferenceSession(str(onnx_path))
    ort_out = session.run(None, {"sequence": dummy})[0]

    match = np.allclose(pt_out, ort_out, atol=atol)
    if match:
        logger.info("ONNX validation PASSED (atol=%.1e)", atol)
    else:
        max_diff = float(np.max(np.abs(pt_out - ort_out)))
        logger.warning("ONNX validation FAILED - max diff: %.6f", max_diff)
    return match


class ONNXPredictor:
    """Lightweight ONNX-based inference engine for production."""

    def __init__(self, onnx_path: str | Path) -> None:
        if not _ORT_AVAILABLE:
            raise ImportError("onnxruntime is required for ONNXPredictor")
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )

    def predict(self, sequences: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"sequence": sequences.astype(np.float32)})[0]
