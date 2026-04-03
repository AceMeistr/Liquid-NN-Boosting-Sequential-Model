from __future__ import annotations

import argparse
import gc
import logging
import math
from pathlib import Path
from typing import Any

import optuna
from optuna.trial import TrialState
import pandas as pd

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import write_json, build_optuna_rdb_storage, build_sqlite_storage_url, is_cuda_oom
from modelmk1.models.xgb_spectral import (
    build_latest_correlation_spectral,
    build_spectral_training_matrices,
    load_constituent_price_panel,
    train_xgb_with_dmatrix,
    train_xgb_walk_forward,
)
from modelmk1.tuning.plateau import compute_plateau_penalty


logger = logging.getLogger(__name__)
_DEFAULT_PRUNER = "median"
_SUPPORTED_PRUNERS = ("median", "hyperband")
_PLATEAU_MAX_NEIGHBORS = 8
_PLATEAU_MIN_LOCAL_NEIGHBORS = 4
_PLATEAU_RADIUS = 0.30
_PLATEAU_GRACE_TRIALS = 12
_PLATEAU_CLIFF_WEIGHT = 0.35
_PLATEAU_SPREAD_WEIGHT = 0.10
_TARGET_SHARPE_MIN = 2.0
_TARGET_DIRECTIONAL_ACC = 0.52
_TARGET_PBO_MAX = 0.20
_PBO_SHARPE_TANDEM_WEIGHT = 2.00
_MAX_DRAWDOWN_GUARD = 0.25


def _build_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    name = str(pruner_name).strip().lower()
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0, interval_steps=1)
    if name == "hyperband":
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=1300, reduction_factor=3)
    raise ValueError(f"Unsupported pruner '{pruner_name}'. Supported values: {', '.join(_SUPPORTED_PRUNERS)}")


def _build_sqlite_storage_url(db_path: str) -> str:
    """Compatibility wrapper used by tests and external callers."""
    return build_sqlite_storage_url(db_path)


def _parse_tickers(tickers_raw: str | None) -> list[str] | None:
    if not tickers_raw:
        return None
    tickers = [t.strip() for t in tickers_raw.split(",") if t.strip()]
    return tickers or None


def _effective_sharpe_with_pbo(raw_sharpe: float, pbo_proxy: float) -> tuple[float, float, float]:
    if not isinstance(raw_sharpe, (int, float)):
        raw_sharpe = 0.0
    if not isinstance(pbo_proxy, (int, float)):
        pbo_proxy = 1.0

    raw = float(raw_sharpe)
    pbo = float(min(max(float(pbo_proxy), 0.0), 1.0))
    pbo_excess = max(pbo - _TARGET_PBO_MAX, 0.0)
    pbo_sharpe_penalty = pbo_excess * _PBO_SHARPE_TANDEM_WEIGHT
    effective_sharpe = raw - pbo_sharpe_penalty
    return float(effective_sharpe), float(pbo_sharpe_penalty), float(pbo)




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
        payload["best_effective_sharpe"] = float(best.user_attrs.get("effective_sharpe_for_objective", 0.0))
        payload["best_pbo_proxy"] = float(best.user_attrs.get("val_pbo_proxy", 1.0))
        payload["best_directional_accuracy"] = float(best.user_attrs.get("val_directional_accuracy", 0.0))
        payload["best_max_drawdown_ratio"] = float(best.user_attrs.get("max_drawdown_ratio", 0.0))
        payload["best_meets_target"] = bool(best.user_attrs.get("meets_target", False))

    write_json(model_dir / "optuna_xgb_runtime_checkpoint.json", payload)


def _trial_score(metrics: dict[str, Any]) -> float:
    raw_sharpe = float(metrics.get("val_sharpe_ratio", 0.0))
    effective_sharpe, _, _ = _effective_sharpe_with_pbo(raw_sharpe, float(metrics.get("val_pbo_proxy", 1.0)))
    direction = float(metrics.get("val_directional_accuracy", 0.0))

    # PBO is coupled to Sharpe only (tandem), not added as a separate global term.
    score = effective_sharpe + (direction - 0.5) * 8.0
    if direction < _TARGET_DIRECTIONAL_ACC:
        score -= (_TARGET_DIRECTIONAL_ACC - direction) * 30.0
    if effective_sharpe < _TARGET_SHARPE_MIN:
        score -= (_TARGET_SHARPE_MIN - effective_sharpe) * 3.0
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
    pruner: str = _DEFAULT_PRUNER,
    storage_url: str | None = None,
    storage_db_path: str | None = None,
) -> dict[str, Any]:
    dataset_path = resolve_data_file(data_path)
    model_dir = get_app_paths().outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_file = Path(storage_db_path).expanduser() if storage_db_path else (model_dir / "optuna_xgb_study.db")
    if not db_file.is_absolute():
        db_file = (Path.cwd() / db_file).resolve()
    db_path = str(db_file.resolve())
    storage_reference = str(storage_url).strip() if storage_url else db_path
    storage_backend = build_optuna_rdb_storage(db_path=db_path, storage_url=storage_url)
    indicator_names = _parse_tickers(tickers_raw)

    logger.info("Loading market frame for classification XGBoost Optuna from %s", dataset_path)
    market_frame = load_constituent_price_panel(dataset_path)

    def objective(trial: optuna.Trial) -> float:
        top_n = trial.suggest_categorical("top_n", [8, 10, 12, 16])
        corr_window = trial.suggest_categorical("corr_window", [30, 45, 60, 90])
        horizon = trial.suggest_categorical("horizon", [1, 2, 3, 5])
        seq_len = trial.suggest_categorical("seq_len", [32, 48, 64, 96])
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
            # Walk-forward CV: anchored expanding folds, scaler re-fit per fold
            # Prune at fold-level via trial.report(step=fold_idx)
            wf_metrics = train_xgb_walk_forward(
                dataset,
                n_splits=5,
                min_train_ratio=0.50,
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

        sharpe = float(wf_metrics["wf_sharpe_p5"])  # 5th-percentile = worst-case regime
        direction = float(wf_metrics["wf_dir_acc_mean"])
        max_drawdown_ratio = float(wf_metrics.get("wf_max_drawdown_ratio", 1.0))

        if (not math.isfinite(max_drawdown_ratio)) or max_drawdown_ratio > _MAX_DRAWDOWN_GUARD:
            raise optuna.TrialPruned(
                f"Max drawdown guard triggered: {max_drawdown_ratio:.3f} > {_MAX_DRAWDOWN_GUARD:.2f}"
            )
        
        # Deflated Sharpe Multiple Testing Haircut
        completed_trials = len([t for t in trial.study.trials if t.state == TrialState.COMPLETE])
        gamma = 0.5772
        e_max = math.sqrt(2 * math.log(max(completed_trials, 1))) if completed_trials > 0 else 0
        haircut_factor = math.sqrt(max(1.0 - gamma * (e_max / max(1.0, sharpe)), 0.01))
        deflated_sharpe = sharpe * haircut_factor

        meets_target = bool(deflated_sharpe >= _TARGET_SHARPE_MIN and direction > _TARGET_DIRECTIONAL_ACC)
        
        # Calculate Base Score (Modified from _trial_score)
        score = deflated_sharpe + (direction - 0.5) * 8.0
        if direction < _TARGET_DIRECTIONAL_ACC:
            score -= (_TARGET_DIRECTIONAL_ACC - direction) * 30.0
        if deflated_sharpe < _TARGET_SHARPE_MIN:
            score -= (_TARGET_SHARPE_MIN - deflated_sharpe) * 3.0
            
        base_score = float(score)

        plateau = compute_plateau_penalty(
            study=trial.study,
            current_params=trial.params,
            current_trial_number=trial.number,
            base_score=base_score,
            direction="maximize",
            distributions=trial.distributions,
            max_neighbors=_PLATEAU_MAX_NEIGHBORS,
            min_local_neighbors=_PLATEAU_MIN_LOCAL_NEIGHBORS,
            radius=_PLATEAU_RADIUS,
            grace_completed_trials=_PLATEAU_GRACE_TRIALS,
            cliff_weight=_PLATEAU_CLIFF_WEIGHT,
            spread_weight=_PLATEAU_SPREAD_WEIGHT,
        )
        
        # Hard Plateau Rejection
        if int(plateau["neighbor_count"]) >= _PLATEAU_MAX_NEIGHBORS and float(plateau["local_iqr"]) < 0.01:
            raise optuna.TrialPruned("Trial pruned due to strict plateau trapping - forcing exploration.")
            
        final_score = base_score - float(plateau["penalty"])

        trial.set_user_attr("val_sharpe_ratio", sharpe)
        trial.set_user_attr("val_pbo_proxy", 0.0)  # Replaced by Deflated Sharpe
        trial.set_user_attr("effective_sharpe_for_objective", deflated_sharpe)
        trial.set_user_attr("pbo_sharpe_penalty", 0.0)
        trial.set_user_attr("val_directional_accuracy", direction)
        trial.set_user_attr("val_mae", float(wf_metrics["wf_loss_mean"]))
        trial.set_user_attr("val_mse", float(wf_metrics["wf_loss_mean"]))
        trial.set_user_attr("wf_sharpe_mean", float(wf_metrics["wf_sharpe_mean"]))
        trial.set_user_attr("wf_sharpe_std", float(wf_metrics["wf_sharpe_std"]))
        trial.set_user_attr("wf_n_folds", int(wf_metrics["wf_n_folds"]))
        trial.set_user_attr("max_drawdown_ratio", max_drawdown_ratio)
        trial.set_user_attr("meets_target", meets_target)
        trial.set_user_attr("base_objective_score", float(base_score))
        trial.set_user_attr("plateau_penalty", float(plateau["penalty"]))
        trial.set_user_attr("plateau_neighbor_count", int(plateau["neighbor_count"]))
        trial.set_user_attr("plateau_neighbor_median", float(plateau["local_median"]))
        trial.set_user_attr("plateau_neighbor_iqr", float(plateau["local_iqr"]))
        trial.set_user_attr("plateau_cliff_gap", float(plateau["cliff_gap"]))
        trial.set_user_attr("plateau_reason", str(plateau["reason"]))
        trial.set_user_attr("objective_score", float(final_score))
        gc.collect()
        return float(final_score)

    selected_pruner = str(pruner).strip().lower()
    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=_build_pruner(selected_pruner),
        storage=storage_backend,
        load_if_exists=True,
    )

    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    existing_finished_trials = len([t for t in study.trials if t.state in finished_states])
    remaining_trials = max(trials - existing_finished_trials, 0)

    _write_runtime_checkpoint(study, storage_reference)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=max(1, int(jobs)),
            timeout=None if timeout <= 0 else timeout,
            callbacks=[lambda s, _t: _write_runtime_checkpoint(s, storage_reference)],
        )
    else:
        logger.info(
            "Requested %d total Optuna trials; study already has %d finished trials",
            trials,
            existing_finished_trials,
        )

    _write_runtime_checkpoint(study, storage_reference)

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]
    payload: dict[str, Any] = {
        "best_score": float(study.best_value) if complete_trials else None,
        "best_params": study.best_params if complete_trials else {},
        "best_sharpe_ratio": (
            float(study.best_trial.user_attrs.get("val_sharpe_ratio", 0.0)) if complete_trials else None
        ),
        "best_effective_sharpe": (
            float(study.best_trial.user_attrs.get("effective_sharpe_for_objective", 0.0)) if complete_trials else None
        ),
        "best_pbo_proxy": (
            float(study.best_trial.user_attrs.get("val_pbo_proxy", 1.0)) if complete_trials else None
        ),
        "best_directional_accuracy": (
            float(study.best_trial.user_attrs.get("val_directional_accuracy", 0.0)) if complete_trials else None
        ),
        "best_max_drawdown_ratio": (
            float(study.best_trial.user_attrs.get("max_drawdown_ratio", 0.0)) if complete_trials else None
        ),
        "best_meets_target": (
            bool(study.best_trial.user_attrs.get("meets_target", False)) if complete_trials else False
        ),
        "study_name": study.study_name,
        "n_trials_completed": len(complete_trials),
        "n_trials_total": len(study.trials),
        "pruner": selected_pruner,
        "storage": storage_reference,
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
    parser.add_argument("--pruner", type=str, default=_DEFAULT_PRUNER, choices=list(_SUPPORTED_PRUNERS))
    parser.add_argument("--storage-url", type=str, default=None)
    parser.add_argument("--storage-db-path", type=str, default=None)
    args = parser.parse_args()
    print(
        run_optuna_xgb_spectral(
            trials=args.trials,
            timeout=args.timeout,
            data_path=args.data_path,
            study_name=args.study_name,
            jobs=args.jobs,
            tickers_raw=args.tickers,
            pruner=args.pruner,
            storage_url=args.storage_url,
            storage_db_path=args.storage_db_path,
        )
    )


if __name__ == "__main__":
    main()
