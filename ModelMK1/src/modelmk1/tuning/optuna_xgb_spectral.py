from __future__ import annotations

import argparse
import gc
import logging
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState
import pandas as pd

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json, build_sqlite_storage_url, is_cuda_oom
from modelmk1.models.xgb_spectral import (
    build_latest_correlation_spectral,
    build_spectral_training_matrices,
    load_constituent_price_panel,
    train_xgb_with_dmatrix,
)


logger = logging.getLogger(__name__)




def _study_state_counts(study: optuna.Study) -> dict[str, int]:
    counts = {
        "complete": 0,
        "pruned": 0,
        "failed": 0,
        "running": 0,
        "waiting": 0,
    }
    for trial in study.trials:
        if trial.state == TrialState.COMPLETE:
            counts["complete"] += 1
        elif trial.state == TrialState.PRUNED:
            counts["pruned"] += 1
        elif trial.state == TrialState.FAIL:
            counts["failed"] += 1
        elif trial.state == TrialState.RUNNING:
            counts["running"] += 1
        elif trial.state == TrialState.WAITING:
            counts["waiting"] += 1
    return counts


def _write_runtime_checkpoint(study: optuna.Study, db_path: str) -> None:
    model_dir = get_app_paths().outputs / "model"
    state_counts = _study_state_counts(study)

    payload: dict[str, Any] = {
        "study_name": study.study_name,
        "storage": db_path,
        "n_trials_total": len(study.trials),
        "n_trials_complete": state_counts["complete"],
        "n_trials_pruned": state_counts["pruned"],
        "n_trials_failed": state_counts["failed"],
        "n_trials_running": state_counts["running"],
        "n_trials_waiting": state_counts["waiting"],
    }

    if state_counts["complete"] > 0:
        best = study.best_trial
        payload["best_trial_number"] = best.number
        payload["best_score"] = float(study.best_value)
        payload["best_params"] = best.params
        payload["best_sharpe_ratio"] = float(best.user_attrs.get("val_sharpe_ratio", 0.0))
        payload["best_directional_accuracy"] = float(best.user_attrs.get("val_directional_accuracy", 0.0))
        payload["best_meets_target"] = bool(best.user_attrs.get("meets_target", False))

    write_json(model_dir / "optuna_xgb_runtime_checkpoint.json", payload)


def _trial_score(metrics: dict[str, Any]) -> float:
    sharpe = float(metrics.get("val_sharpe_ratio", 0.0))
    direction = float(metrics.get("val_directional_accuracy", 0.0))

    # Prioritize Sharpe while enforcing directional edge.
    score = sharpe + (direction - 0.5) * 8.0
    if direction < 0.52:
        score -= (0.52 - direction) * 30.0
    if sharpe < 2.0:
        score -= (2.0 - sharpe) * 3.0
    return float(score)


def _build_xgb_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": trial.suggest_float("eta", 0.005, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 12.0, log=True),
        "lambda": trial.suggest_float("lambda", 1e-3, 20.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-5, 2.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 6.0),
        "max_bin": trial.suggest_categorical("max_bin", [256, 512, 1024]),
        "decision_threshold": trial.suggest_float("decision_threshold", 0.45, 0.60),
        "tree_method": "gpu_hist",
        "device": "cuda",
        "seed": 42,
    }


def _extract_xgb_params(params: dict[str, Any]) -> dict[str, Any]:
    return {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": float(params["eta"]),
        "max_depth": int(params["max_depth"]),
        "subsample": float(params["subsample"]),
        "colsample_bytree": float(params["colsample_bytree"]),
        "min_child_weight": float(params["min_child_weight"]),
        "lambda": float(params["lambda"]),
        "alpha": float(params["alpha"]),
        "gamma": float(params["gamma"]),
        "max_bin": int(params["max_bin"]),
        "decision_threshold": float(params.get("decision_threshold", 0.5)),
        "tree_method": "gpu_hist",
        "device": "cuda",
        "seed": 42,
    }


def _train_and_export_best(
    best_params: dict[str, Any],
    market_frame: pd.DataFrame,
    indicator_names: list[str] | None,
    model_dir: Path,
    dataset_path: Path,
) -> dict[str, Any]:
    top_n = int(best_params["top_n"])
    corr_window = int(best_params["corr_window"])
    horizon = int(best_params["horizon"])
    seq_len = int(best_params["seq_len"])
    train_ratio = float(best_params["train_ratio"])
    num_boost_round = int(best_params["num_boost_round"])
    early_stopping_rounds = int(best_params["early_stopping_rounds"])
    robust_q_low = float(best_params["robust_q_low"])
    robust_q_high = float(best_params["robust_q_high"])

    latest = build_latest_correlation_spectral(
        market_frame,
        top_n=top_n,
        corr_window=corr_window,
        horizon=horizon,
        seq_len=seq_len,
        tickers=indicator_names,
        robust_quantile_low=robust_q_low,
        robust_quantile_high=robust_q_high,
    )
    dataset = build_spectral_training_matrices(
        market_frame,
        top_n=top_n,
        corr_window=corr_window,
        horizon=horizon,
        seq_len=seq_len,
        tickers=indicator_names,
        robust_quantile_low=robust_q_low,
        robust_quantile_high=robust_q_high,
    )

    booster, train_metrics = train_xgb_with_dmatrix(
        dataset,
        train_ratio=train_ratio,
        num_boost_round=num_boost_round,
        early_stopping_rounds=early_stopping_rounds,
        params=_extract_xgb_params(best_params),
    )

    corr_df = pd.DataFrame(
        latest.correlation_matrix,
        index=latest.constituents,
        columns=latest.constituents,
    )
    corr_path = model_dir / f"sensex_top{top_n}_correlation_matrix_optuna.csv"
    corr_df.to_csv(corr_path)

    model_path = model_dir / f"xgb_spectral_optuna_top{top_n}.json"
    booster.save_model(model_path)

    summary = {
        "dataset": str(dataset_path),
        "model_type": "xgboost_classifier_spectral_lnn_features_optuna",
        "top_n": top_n,
        "corr_window": corr_window,
        "horizon": horizon,
        "seq_len": seq_len,
        "train_ratio": train_ratio,
        "num_boost_round": num_boost_round,
        "early_stopping_rounds": early_stopping_rounds,
        "robust_q_low": robust_q_low,
        "robust_q_high": robust_q_high,
        "selected_indicators": latest.constituents,
        "latest_correlation_matrix_csv": str(corr_path),
        "latest_eigenvalues": [float(v) for v in latest.eigenvalues.tolist()],
        "latest_matrix_window_end": str(latest.window_end_timestamp),
        "xgb_model": str(model_path),
        "feature_names": dataset.feature_names,
        "num_samples": int(len(dataset.targets)),
        **train_metrics,
    }
    write_json(model_dir / "xgb_spectral_optuna_metrics.json", summary)
    return summary


def run_optuna_xgb_spectral(
    trials: int,
    timeout: int,
    data_path: str | None,
    study_name: str,
    jobs: int,
    tickers_raw: str | None,
) -> dict[str, Any]:
    dataset_path = resolve_data_file(data_path)
    model_dir = get_app_paths().outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_path = str((model_dir / "optuna_xgb_study.db").resolve())
    indicator_names = None
    if tickers_raw:
        indicator_names = [t.strip() for t in tickers_raw.split(",") if t.strip()] or None

    logger.info("Loading market frame for classification XGBoost Optuna from %s", dataset_path)
    market_frame = load_constituent_price_panel(dataset_path)

    def objective(trial: optuna.Trial) -> float:
        top_n = trial.suggest_categorical("top_n", [8, 10, 12, 16])
        corr_window = trial.suggest_categorical("corr_window", [30, 45, 60, 90])
        horizon = trial.suggest_categorical("horizon", [1, 2, 3, 5])
        seq_len = trial.suggest_categorical("seq_len", [32, 48, 64, 96])
        train_ratio = trial.suggest_float("train_ratio", 0.75, 0.9)
        num_boost_round = trial.suggest_int("num_boost_round", 300, 1300, step=100)
        early_stopping_rounds = trial.suggest_int("early_stopping_rounds", 20, 120, step=10)
        robust_q_low = trial.suggest_categorical("robust_q_low", [10, 15, 20, 25])
        robust_q_high = trial.suggest_categorical("robust_q_high", [75, 80, 85, 90])

        if robust_q_high <= robust_q_low + 10:
            raise optuna.TrialPruned("Invalid robust quantile configuration")

        try:
            dataset = build_spectral_training_matrices(
                market_frame,
                top_n=int(top_n),
                corr_window=int(corr_window),
                horizon=int(horizon),
                seq_len=int(seq_len),
                tickers=indicator_names,
                robust_quantile_low=float(robust_q_low),
                robust_quantile_high=float(robust_q_high),
            )
            _, metrics = train_xgb_with_dmatrix(
                dataset,
                train_ratio=float(train_ratio),
                num_boost_round=int(num_boost_round),
                early_stopping_rounds=int(early_stopping_rounds),
                params=_build_xgb_params(trial),
            )
        except Exception as exc:  # noqa: BLE001
            if is_cuda_oom(exc):
                logger.warning("Trial %d pruned due to GPU memory pressure", trial.number)
                gc.collect()
                raise optuna.TrialPruned("Pruned after CUDA OOM") from exc
            raise

        sharpe = float(metrics.get("val_sharpe_ratio", 0.0))
        direction = float(metrics.get("val_directional_accuracy", 0.0))
        meets_target = bool(sharpe >= 2.0 and direction >= 0.52)
        score = _trial_score(metrics)

        trial.set_user_attr("val_sharpe_ratio", sharpe)
        trial.set_user_attr("val_directional_accuracy", direction)
        trial.set_user_attr("val_mae", float(metrics.get("val_mae", 0.0)))
        trial.set_user_attr("val_mse", float(metrics.get("val_mse", 0.0)))
        trial.set_user_attr("meets_target", meets_target)
        trial.set_user_attr("objective_score", score)
        gc.collect()
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1),
        storage=build_sqlite_storage_url(db_path),
        load_if_exists=True,
    )

    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    existing_finished_trials = len([t for t in study.trials if t.state in finished_states])
    remaining_trials = max(trials - existing_finished_trials, 0)

    _write_runtime_checkpoint(study, db_path)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=max(1, int(jobs)),
            timeout=None if timeout <= 0 else timeout,
            callbacks=[lambda s, _t: _write_runtime_checkpoint(s, db_path)],
        )
    else:
        logger.info(
            "Requested %d total Optuna trials; study already has %d finished trials",
            trials,
            existing_finished_trials,
        )

    _write_runtime_checkpoint(study, db_path)

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    payload: dict[str, Any] = {
        "best_score": float(study.best_value) if complete_trials else None,
        "best_params": study.best_params if complete_trials else {},
        "best_sharpe_ratio": (
            float(study.best_trial.user_attrs.get("val_sharpe_ratio", 0.0)) if complete_trials else None
        ),
        "best_directional_accuracy": (
            float(study.best_trial.user_attrs.get("val_directional_accuracy", 0.0)) if complete_trials else None
        ),
        "best_meets_target": (
            bool(study.best_trial.user_attrs.get("meets_target", False)) if complete_trials else False
        ),
        "study_name": study.study_name,
        "n_trials_completed": len(complete_trials),
        "n_trials_total": len(study.trials),
        "storage": db_path,
    }

    if complete_trials:
        best_model_metrics = _train_and_export_best(
            study.best_params,
            market_frame=market_frame,
            indicator_names=indicator_names,
            model_dir=model_dir,
            dataset_path=dataset_path,
        )
        payload["best_model_metrics"] = best_model_metrics
        logger.info(
            "Optuna best score=%.4f | sharpe=%.3f | directional_acc=%.3f | meets_target=%s",
            float(study.best_value),
            float(study.best_trial.user_attrs.get("val_sharpe_ratio", 0.0)),
            float(study.best_trial.user_attrs.get("val_directional_accuracy", 0.0)),
            bool(study.best_trial.user_attrs.get("meets_target", False)),
        )
    else:
        logger.warning("XGBoost Optuna finished without completed trials")

    write_json(model_dir / "optuna_xgb_best.json", payload)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for classification spectral XGBoost")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--study-name", type=str, default="modelmk1_xgb_optuna")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--tickers", type=str, default=None, help="Comma-separated explicit indicator names")
    args = parser.parse_args()
    print(
        run_optuna_xgb_spectral(
            trials=args.trials,
            timeout=args.timeout,
            data_path=args.data_path,
            study_name=args.study_name,
            jobs=args.jobs,
            tickers_raw=args.tickers,
        )
    )


if __name__ == "__main__":
    main()
