from __future__ import annotations

import argparse
<<<<<<< HEAD

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json
=======
import logging

import torch

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import pick_device, safe_torch_load, write_json
from modelmk1.eval.inference_speed import benchmark_model_latency_ms, build_pipeline_speed_report
from modelmk1.models.checkpoint_utils import checkpoint_model_kwargs, is_legacy_checkpoint
from modelmk1.models.lnn_model import MarketLNN
>>>>>>> main
from modelmk1.train.build_models import run_build
from modelmk1.train.train_hybrid import build_parser as build_hybrid_parser
from modelmk1.train.train_hybrid import run_hybrid_training
from modelmk1.train.train_lnn import build_parser as build_lnn_parser
from modelmk1.train.train_lnn import run_training
from modelmk1.tuning.optuna_lnn import run_optuna

<<<<<<< HEAD

def run_pipeline(data_path: str | None, trials: int, build_only: bool) -> dict:
    run_build(input_size=None, hidden_size=128, dropout=0.15, seq_len=64, horizon=15, resample_freq="1min")
=======
logger = logging.getLogger(__name__)


def run_pipeline(data_path: str | None, trials: int, build_only: bool) -> dict:
    logger.info("=== ModelMK1 Pipeline: Build -> Optuna -> Train -> Hybrid ===")

    run_build(
        input_size=None,
        hidden_size=128,
        dropout=0.15,
        seq_len=64,
        horizon=15,
        resample_freq="1min",
        use_attention=False,
        backbone_type="mamba",
        cnn_frontend="inception",
    )
    logger.info("Phase 1/4: Build complete")
>>>>>>> main

    if build_only:
        payload = {
            "status": "on_hold",
            "reason": "build_only_requested",
            "action": "Place real dataset in Training Data or pass --data-path to start Optuna + training.",
        }
        write_json(get_app_paths().outputs / "model" / "training_hold.json", payload)
        return payload

    dataset = resolve_data_file(data_path)
<<<<<<< HEAD
    best = run_optuna(trials=trials, timeout=0, data_path=str(dataset), study_name="modelmk1_lnn_optuna")

=======
    logger.info("Phase 2/4: Optuna hyperparameter tuning (%d trials)", trials)
    best = run_optuna(trials=trials, timeout=0, data_path=str(dataset), study_name="modelmk1_lnn_optuna")

    logger.info("Phase 3/4: Training LNN with best hyperparameters")
>>>>>>> main
    lnn_args = build_lnn_parser().parse_args([])
    lnn_args.data_path = str(dataset)
    lnn_args.epochs = int(best["best_params"].get("epochs", 10))
    lnn_args.batch_size = int(best["best_params"].get("batch_size", 128))
    lnn_args.hidden_size = int(best["best_params"].get("hidden_size", 128))
    lnn_args.dropout = float(best["best_params"].get("dropout", 0.15))
    lnn_args.lr = float(best["best_params"].get("lr", 1e-3))
    lnn_args.weight_decay = float(best["best_params"].get("weight_decay", 1e-4))
    lnn_args.seq_len = int(best["best_params"].get("seq_len", 64))
    lnn_args.horizon = int(best["best_params"].get("horizon", 15))
<<<<<<< HEAD
    lnn_metrics = run_training(lnn_args)

=======
    # New architecture params from Optuna
    lnn_args.num_heads = int(best["best_params"].get("num_heads", 4))
    lnn_args.num_layers = int(best["best_params"].get("num_layers", 2))
    lnn_args.use_attention = bool(best["best_params"].get("use_attention", True))
    lnn_args.backbone_type = best["best_params"].get("backbone_type", "mamba")
    lnn_args.cnn_frontend = best["best_params"].get("cnn_frontend", "inception")
    lnn_args.mamba_d_state = int(best["best_params"].get("mamba_d_state", 16))
    lnn_args.mamba_d_conv = int(best["best_params"].get("mamba_d_conv", 4))
    lnn_args.mamba_expand = int(best["best_params"].get("mamba_expand", 2))
    lnn_args.scheduler = best["best_params"].get("scheduler", "cosine")
    lnn_args.warmup_epochs = int(best["best_params"].get("warmup_epochs", 2))
    lnn_args.use_ema = bool(best["best_params"].get("use_ema", True))
    lnn_args.ema_decay = float(best["best_params"].get("ema_decay", 0.999))
    lnn_metrics = run_training(lnn_args)

    logger.info("Phase 4/4: Training hybrid residual model")
>>>>>>> main
    hybrid_args = build_hybrid_parser().parse_args([])
    hybrid_args.data_path = str(dataset)
    hybrid_args.seq_len = lnn_args.seq_len
    hybrid_args.horizon = lnn_args.horizon
    hybrid_metrics = run_hybrid_training(hybrid_args)
<<<<<<< HEAD
    return {"optuna": best, "lnn": lnn_metrics, "hybrid": hybrid_metrics}
=======

    logger.info("Phase 5/5: Benchmarking inference speed for 1m and 5m windows")
    paths = get_app_paths()
    bench_device = pick_device()
    checkpoint = safe_torch_load(paths.outputs / "model" / "lnn_best.pt", map_location=bench_device)
    speed_model = MarketLNN(
        **checkpoint_model_kwargs(checkpoint),
    ).to(bench_device)
    speed_model.load_state_dict(checkpoint["model_state_dict"], strict=not is_legacy_checkpoint(checkpoint))
    avg_model_latency_ms = benchmark_model_latency_ms(
        speed_model,
        seq_len=int(checkpoint.get("seq_len", lnn_args.seq_len)),
        input_size=int(checkpoint["input_size"]),
        device=bench_device,
        warmup_runs=10,
        timed_runs=60,
    )
    speed_report = build_pipeline_speed_report(
        avg_model_latency_ms=avg_model_latency_ms,
        with_confidence=True,
        mc_samples=30,
    )
    write_json(paths.outputs / "model" / "inference_speed_budget.json", speed_report)
    del speed_model
    if bench_device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info("=== Pipeline Complete ===")
    return {
        "optuna": best,
        "lnn": lnn_metrics,
        "hybrid": hybrid_metrics,
        "inference_speed": speed_report,
    }
>>>>>>> main


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelMK1 pipeline: build -> optuna -> train -> hybrid")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    print(run_pipeline(args.data_path, args.trials, args.build_only))


if __name__ == "__main__":
    main()
