from __future__ import annotations

import argparse

import optuna

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json
from modelmk1.train.train_lnn import build_parser as build_lnn_parser
from modelmk1.train.train_lnn import run_training


def run_optuna(trials: int, timeout: int, data_path: str | None, study_name: str) -> dict:
    dataset = resolve_data_file(data_path)

    def objective(trial: optuna.Trial) -> float:
        parser = build_lnn_parser()
        args = parser.parse_args([])
        args.data_path = str(dataset)
        args.epochs = trial.suggest_int("epochs", 8, 20)
        args.batch_size = trial.suggest_categorical("batch_size", [64, 128, 256])
        args.hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
        args.dropout = trial.suggest_float("dropout", 0.05, 0.4)
        args.lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        args.weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        args.seq_len = trial.suggest_categorical("seq_len", [32, 64, 128])
        args.horizon = trial.suggest_categorical("horizon", [10, 15, 30])
        args.early_stopping_patience = 4
        args.seed = 42
        args.cpu = False
        metrics = run_training(args)
        trial.report(metrics["best_val_loss"], step=trial.number)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return metrics["best_val_loss"]

    study = optuna.create_study(direction="minimize", study_name=study_name, pruner=optuna.pruners.MedianPruner(n_warmup_steps=3))
    study.optimize(objective, n_trials=trials, timeout=None if timeout <= 0 else timeout)

    payload = {"best_value": study.best_value, "best_params": study.best_params, "study_name": study.study_name}
    write_json(get_app_paths().outputs / "model" / "optuna_best.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for LNN")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--study-name", type=str, default="modelmk1_lnn_optuna")
    parser.add_argument("--data-path", type=str, default=None)
    args = parser.parse_args()
    print(run_optuna(args.trials, args.timeout, args.data_path, args.study_name))


if __name__ == "__main__":
    main()
