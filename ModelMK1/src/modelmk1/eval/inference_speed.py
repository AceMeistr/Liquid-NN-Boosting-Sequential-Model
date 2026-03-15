from __future__ import annotations

import time
from typing import Any

import torch


def benchmark_model_latency_ms(
    model: torch.nn.Module,
    seq_len: int,
    input_size: int,
    device: torch.device,
    warmup_runs: int = 20,
    timed_runs: int = 120,
) -> float:
    if warmup_runs < 0 or timed_runs <= 0:
        raise ValueError("warmup_runs must be >= 0 and timed_runs must be > 0")

    model.eval()
    x = torch.randn(1, seq_len, input_size, dtype=torch.float32, device=device)

    with torch.inference_mode():
        for _ in range(warmup_runs):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        t0 = time.perf_counter()
        for _ in range(timed_runs):
            _ = model(x)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0

    del x
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return float((elapsed / timed_runs) * 1000.0)


def build_pipeline_speed_report(
    avg_model_latency_ms: float,
    with_confidence: bool,
    mc_samples: int,
    feature_time_ms_range: tuple[float, float] = (10.0, 36.0),
    ingest_time_ms_range: tuple[float, float] = (1.0, 5.0),
    resample_time_ms_range: tuple[float, float] = (2.0, 15.0),
    seq_build_time_ms_range: tuple[float, float] = (0.1, 0.5),
    xgb_time_ms_range: tuple[float, float] = (0.01, 0.05),
    postprocess_time_ms_range: tuple[float, float] = (0.5, 1.0),
) -> dict[str, Any]:
    if avg_model_latency_ms < 0:
        raise ValueError("avg_model_latency_ms must be >= 0")

    mc_factor = max(int(mc_samples), 1) if with_confidence else 1
    model_stage = (avg_model_latency_ms * mc_factor, avg_model_latency_ms * mc_factor)

    def _sum_low_high(parts: list[tuple[float, float]]) -> tuple[float, float]:
        low = sum(p[0] for p in parts)
        high = sum(p[1] for p in parts)
        return low, high

    pipeline_low_high = _sum_low_high([
        ingest_time_ms_range,
        resample_time_ms_range,
        feature_time_ms_range,
        seq_build_time_ms_range,
        model_stage,
        xgb_time_ms_range,
        postprocess_time_ms_range,
    ])

    def _budget_entry(window_ms: float) -> dict[str, float]:
        low_ms, high_ms = pipeline_low_high
        return {
            "budget_ms": float(window_ms),
            "pipeline_ms_low": float(low_ms),
            "pipeline_ms_high": float(high_ms),
            "budget_used_pct_low": float((low_ms / window_ms) * 100.0),
            "budget_used_pct_high": float((high_ms / window_ms) * 100.0),
            "headroom_ms_low": float(window_ms - low_ms),
            "headroom_ms_high": float(window_ms - high_ms),
        }

    return {
        "avg_model_latency_ms": float(avg_model_latency_ms),
        "with_confidence": bool(with_confidence),
        "mc_samples": int(mc_samples if with_confidence else 1),
        "stages_ms": {
            "ingest": {"low": ingest_time_ms_range[0], "high": ingest_time_ms_range[1]},
            "resample": {"low": resample_time_ms_range[0], "high": resample_time_ms_range[1]},
            "feature_compute": {"low": feature_time_ms_range[0], "high": feature_time_ms_range[1]},
            "sequence_build": {"low": seq_build_time_ms_range[0], "high": seq_build_time_ms_range[1]},
            "model": {"low": model_stage[0], "high": model_stage[1]},
            "xgb": {"low": xgb_time_ms_range[0], "high": xgb_time_ms_range[1]},
            "postprocess": {"low": postprocess_time_ms_range[0], "high": postprocess_time_ms_range[1]},
        },
        "windows": {
            "1m": _budget_entry(60_000.0),
            "5m": _budget_entry(300_000.0),
        },
    }
