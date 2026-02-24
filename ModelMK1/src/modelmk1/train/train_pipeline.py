from __future__ import annotations

import argparse

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json
from modelmk1.train.build_models import run_build
from modelmk1.train.train_hybrid import build_parser as build_hybrid_parser
from modelmk1.train.train_hybrid import run_hybrid_training
from modelmk1.train.train_lnn import build_parser as build_lnn_parser
from modelmk1.train.train_lnn import run_training
from modelmk1.tuning.optuna_lnn import run_optuna


def run_pipeline(data_path: str | None, trials: int, build_only: bool) -> dict:
    run_build(input_size=None, hidden_size=128, dropout=0.15, seq_len=64, horizon=15, resample_freq="1min")

    if build_only:
        payload = {
            "status": "on_hold",
            "reason": "build_only_requested",
            "action": "Place real dataset in Training Data or pass --data-path to start Optuna + training.",
        }
        write_json(get_app_paths().outputs / "model" / "training_hold.json", payload)
        return payload

    dataset = resolve_data_file(data_path)
    best = run_optuna(trials=trials, timeout=0, data_path=str(dataset), study_name="modelmk1_lnn_optuna")

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
    lnn_metrics = run_training(lnn_args)

    hybrid_args = build_hybrid_parser().parse_args([])
    hybrid_args.data_path = str(dataset)
    hybrid_args.seq_len = lnn_args.seq_len
    hybrid_args.horizon = lnn_args.horizon
    hybrid_metrics = run_hybrid_training(hybrid_args)
    return {"optuna": best, "lnn": lnn_metrics, "hybrid": hybrid_metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description="ModelMK1 pipeline: build -> optuna -> train -> hybrid")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--build-only", action="store_true")
    args = parser.parse_args()
    print(run_pipeline(args.data_path, args.trials, args.build_only))


if __name__ == "__main__":
    main()
