from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import threading
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

import optuna
import torch
from optuna.trial import TrialState

from modelmk1.common.paths import get_app_paths, resolve_data_file
from modelmk1.common.runtime import (
    write_json,
    enable_performance_mode,
    build_optuna_rdb_storage,
    is_cuda_oom,
)
from modelmk1.tuning.plateau import compute_plateau_penalty
from modelmk1.train.train_lnn import build_parser as build_lnn_parser
from modelmk1.train.train_lnn import run_training
from modelmk1.data.loader import (
    load_tick_df,
    resample_ticks,
    build_supervised_data,
    anchored_walk_forward_splits,
)

logger = logging.getLogger(__name__)
_CHECKPOINT_WRITE_LOCK = threading.Lock()

_TARGET_SHARPE_MIN = 2.0
_TARGET_SHARPE_STRETCH = 3.0
_TARGET_DIRECTIONAL_ACC = 0.52
_TARGET_DIRECTIONAL_ACC_STRICT_EPS = 1e-6
_TARGET_PBO_MAX = 0.20
_PBO_SHARPE_TANDEM_WEIGHT = 2.00
_DEFAULT_BATCH_SIZE_CHOICES = [128, 256, 512]
_DEFAULT_PRUNER = "median"
_SUPPORTED_PRUNERS = ("median", "hyperband")
_PLATEAU_MAX_NEIGHBORS = 8
_PLATEAU_MIN_LOCAL_NEIGHBORS = 4
_PLATEAU_RADIUS = 0.30
_PLATEAU_GRACE_TRIALS = 12
_PLATEAU_CLIFF_WEIGHT = 0.35
_PLATEAU_SPREAD_WEIGHT = 0.10
_MAX_DRAWDOWN_GUARD = 0.25


def _normalize_batch_size_choices(batch_size_choices: list[int] | tuple[int, ...] | None) -> list[int]:
    if not batch_size_choices:
        return list(_DEFAULT_BATCH_SIZE_CHOICES)
    normalized = sorted({max(1, int(v)) for v in batch_size_choices})
    return normalized or list(_DEFAULT_BATCH_SIZE_CHOICES)


def _objective_distributions(batch_size_choices: list[int]) -> dict[str, optuna.distributions.BaseDistribution]:
    return {
        "epochs": optuna.distributions.IntDistribution(3, 5),
        "batch_size": optuna.distributions.CategoricalDistribution(batch_size_choices),
        "hidden_size": optuna.distributions.CategoricalDistribution([64, 128, 256]),
        "dropout": optuna.distributions.FloatDistribution(0.05, 0.4),
        "lr": optuna.distributions.FloatDistribution(1e-4, 5e-3, log=True),
        "weight_decay": optuna.distributions.FloatDistribution(1e-6, 1e-2, log=True),
        "seq_len": optuna.distributions.CategoricalDistribution([32, 64]),
        "horizon": optuna.distributions.CategoricalDistribution([10, 15, 30]),
        "num_heads": optuna.distributions.CategoricalDistribution([2, 4]),
        "num_layers": optuna.distributions.IntDistribution(1, 2),
        "use_attention": optuna.distributions.CategoricalDistribution([True, False]),
        "backbone_type": optuna.distributions.CategoricalDistribution(["mamba"]),
        "cnn_frontend": optuna.distributions.CategoricalDistribution(["inception", "none"]),
        "mamba_d_state": optuna.distributions.CategoricalDistribution([8, 16, 32]),
        "mamba_d_conv": optuna.distributions.CategoricalDistribution([2, 4]),
        "mamba_expand": optuna.distributions.CategoricalDistribution([2, 3]),
        "scheduler": optuna.distributions.CategoricalDistribution(["cosine", "cosine_warm", "plateau"]),
        "warmup_epochs": optuna.distributions.IntDistribution(0, 1),
        "use_ema": optuna.distributions.CategoricalDistribution([True, False]),
        "ema_decay": optuna.distributions.FloatDistribution(0.99, 0.9999, log=True),
    }


def _build_pruner(pruner_name: str) -> optuna.pruners.BasePruner:
    name = str(pruner_name).strip().lower()
    if name == "median":
        return optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=0, interval_steps=1)
    if name == "hyperband":
        # Epochs are short (<= 15), so keep resource range tight and rung checks frequent.
        return optuna.pruners.HyperbandPruner(min_resource=1, max_resource=15, reduction_factor=3)
    raise ValueError(f"Unsupported pruner '{pruner_name}'. Supported values: {', '.join(_SUPPORTED_PRUNERS)}")


def _effective_sharpe_with_pbo(raw_sharpe: float, pbo_proxy: float) -> tuple[float, float, float]:
    if not math.isfinite(raw_sharpe):
        raw_sharpe = -10.0
    if not math.isfinite(pbo_proxy):
        pbo_proxy = 1.0

    normalized_pbo = float(min(max(float(pbo_proxy), 0.0), 1.0))
    pbo_excess = max(normalized_pbo - _TARGET_PBO_MAX, 0.0)
    pbo_sharpe_penalty = pbo_excess * _PBO_SHARPE_TANDEM_WEIGHT
    effective_sharpe = float(raw_sharpe) - float(pbo_sharpe_penalty)
    return float(effective_sharpe), float(pbo_sharpe_penalty), float(normalized_pbo)


def _is_valid_param_value(value: Any, distribution: optuna.distributions.BaseDistribution) -> bool:
    try:
        if isinstance(distribution, optuna.distributions.CategoricalDistribution):
            return value in distribution.choices
        if isinstance(distribution, optuna.distributions.IntDistribution):
            int_value = int(value)
            if int_value < distribution.low or int_value > distribution.high:
                return False
            return (int_value - distribution.low) % distribution.step == 0
        if isinstance(distribution, optuna.distributions.FloatDistribution):
            float_value = float(value)
            if float_value < distribution.low or float_value > distribution.high:
                return False
            return True
    except Exception:  # noqa: BLE001
        return False
    return False


def _seed_study_with_completed_trials(
    study: optuna.Study,
    distributions: dict[str, optuna.distributions.BaseDistribution],
    seed_trials: list[dict[str, Any]] | None,
) -> int:
    if seed_trials is None or len(study.trials) > 0:
        return 0

    added: int = 0
    for idx, seed in enumerate(seed_trials):
        params: dict[str, Any] = seed.get("params", {})
        value: Any = seed.get("value")
        if not isinstance(params, dict) or value is None:
            continue

        filtered_params: dict[str, Any] = {}
        filtered_distributions: dict[str, optuna.distributions.BaseDistribution] = {}
        for key, distribution in distributions.items():
            if key not in params:
                continue
            candidate = params[key]
            if not _is_valid_param_value(candidate, distribution):
                continue
            if isinstance(distribution, optuna.distributions.IntDistribution):
                filtered_params[key] = int(candidate)
            elif isinstance(distribution, optuna.distributions.FloatDistribution):
                filtered_params[key] = float(candidate)
            else:
                filtered_params[key] = candidate
            filtered_distributions[key] = distribution

        if not filtered_params:
            continue

        u_attrs_raw = seed.get("user_attrs")
        user_attrs: dict[str, Any] = {}
        if isinstance(u_attrs_raw, dict):
            user_attrs.update(u_attrs_raw)
        user_attrs["seeded_trial"] = True
        user_attrs["seed_index"] = idx

        try:
            study.add_trial(
                optuna.trial.create_trial(
                    params=filtered_params,
                    distributions=filtered_distributions,
                    value=float(value),
                    user_attrs=user_attrs,
                )
            )
            added += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not seed study with trial %d: %s", idx, exc)

    if added > 0:
        logger.info("Seeded study %s with %d completed trials", study.study_name, added)
    return added


def _timestamp_utc_compact() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _study_payload(study: optuna.Study, db_path: str) -> tuple[dict, dict[str, int]]:
    state_counts: dict[str, int] = {
        "complete": 0,
        "pruned": 0,
        "fail": 0,
        "running": 0,
        "waiting": 0,
    }
    for trial in study.trials:
        key = trial.state.name.lower()
        if key in state_counts:
            state_counts[key] += 1

    payload: dict = {
        "study_name": study.study_name,
        "storage": db_path,
        "n_trials_total": len(study.trials),
        "n_trials_complete": state_counts["complete"],
        "n_trials_pruned": state_counts["pruned"],
        "n_trials_failed": state_counts["fail"],
        "n_trials_running": state_counts["running"],
        "n_trials_waiting": state_counts["waiting"],
    }

    if state_counts["complete"] > 0:
        best = study.best_trial
        payload["best_trial_number"] = best.number
        payload["best_value"] = float(study.best_value)
        payload["best_params"] = best.params
        payload["best_val_loss"] = float(best.user_attrs.get("best_val_loss", 0.0))
        payload["best_sharpe_ratio"] = float(best.user_attrs.get("cpcv_sharpe_mean", 0.0))
        payload["best_effective_sharpe"] = float(best.user_attrs.get("effective_sharpe_for_objective", 0.0))
        payload["best_pbo_proxy"] = float(best.user_attrs.get("cpcv_pbo_proxy", 1.0))
        payload["best_directional_accuracy"] = float(best.user_attrs.get("val_directional_accuracy", 0.0))
        payload["best_meets_target"] = bool(best.user_attrs.get("meets_target", False))

    return payload, state_counts



def _write_runtime_checkpoint(study: optuna.Study, db_path: str) -> None:
    model_dir = get_app_paths().outputs / "model"
    payload, state_counts = _study_payload(study, db_path)
    payload["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    # Multiple Optuna workers can finish concurrently; serialize checkpoint writes.
    with _CHECKPOINT_WRITE_LOCK:
        write_json(model_dir / "optuna_runtime_checkpoint.json", payload)

        # Keep a legacy-compatible checkpoint in sync so bootstrapping always has fresh state.
        manual_checkpoint = {
            "study_name": study.study_name,
            "completed_trials": int(state_counts["complete"]),
            "best_value": payload.get("best_value"),
            "best_params": payload.get("best_params", {}),
            "best_sharpe_ratio": payload.get("best_sharpe_ratio"),
            "best_directional_accuracy": payload.get("best_directional_accuracy"),
            "updated_at_utc": payload["updated_at_utc"],
        }
        write_json(model_dir / "optuna_checkpoint.json", manual_checkpoint)


def _archive_current_best_study(study: optuna.Study, db_path: str, reason: str) -> str:
    model_dir = get_app_paths().outputs / "model"
    payload, _ = _study_payload(study, db_path)
    payload["archive_reason"] = reason
    payload["archived_at_utc"] = datetime.now(timezone.utc).isoformat()
    archive_name = f"optuna_lnn_best_{reason}_{_timestamp_utc_compact()}.json"
    archive_path = model_dir / archive_name
    write_json(archive_path, payload)
    return str(archive_path)


def _bootstrap_from_manual_checkpoint(
    study: optuna.Study,
    distributions: dict[str, optuna.distributions.BaseDistribution],
) -> None:
    """Seed a fresh persistent study with the last manual checkpoint if present."""
    if len(study.trials) > 0:
        return

    checkpoint_path = get_app_paths().outputs / "model" / "optuna_checkpoint.json"
    if not checkpoint_path.exists():
        return

    try:
        with checkpoint_path.open("r", encoding="utf-8") as fp:
            checkpoint = json.load(fp)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to load manual checkpoint %s: %s", checkpoint_path, exc)
        return

    best_params = checkpoint.get("best_params")
    best_value = checkpoint.get("best_value")
    completed_trials = int(checkpoint.get("completed_trials", 0))

    if not isinstance(best_params, dict) or best_value is None or completed_trials <= 0:
        return

    filtered_params = {k: v for k, v in best_params.items() if k in distributions}
    filtered_distributions = {k: distributions[k] for k in filtered_params}

    if not filtered_params:
        return

    try:
        bootstrap_trial = optuna.trial.create_trial(
            params=filtered_params,
            distributions=filtered_distributions,
            value=float(best_value),
            user_attrs={"bootstrapped_from_manual_checkpoint": True},
        )
        study.add_trial(bootstrap_trial)
        logger.info("Bootstrapped study with manual checkpoint trial value %.6f", float(best_value))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not bootstrap study from manual checkpoint: %s", exc)


def run_optuna(
    trials: int,
    timeout: int,
    data_path: str | None,
    study_name: str,
    workers: int = 3,
    fixed_batch_size: int | None = None,
    batch_size_choices: list[int] | None = None,
    narrow_from_params: list[dict[str, Any]] | None = None,
    seed_trials: list[dict[str, Any]] | None = None,
    bootstrap_manual_checkpoint: bool = True,
    pruner: str = _DEFAULT_PRUNER,
    storage_url: str | None = None,
    storage_db_path: str | None = None,
) -> dict:
    dataset = resolve_data_file(data_path)
    model_dir = get_app_paths().outputs / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    db_file = Path(storage_db_path).expanduser() if storage_db_path else (model_dir / "optuna_study.db")
    if not db_file.is_absolute():
        db_file = (Path.cwd() / db_file).resolve()
    db_path = str(db_file.resolve())
    storage_reference = str(storage_url).strip() if storage_url else db_path
    storage_backend = build_optuna_rdb_storage(db_path=db_path, storage_url=storage_url)
    
    # Pre-load requested dataset globally for all trials in this worker
    logger.info("Pre-loading and resampling dataset for Optuna worker: %s", dataset)
    tmp_parser = build_lnn_parser()
    tmp_args = tmp_parser.parse_args([])
    df_optuna = load_tick_df(str(dataset))
    sampled_optuna = resample_ticks(df_optuna, freq=tmp_args.resample_freq)
    batch_choices = [max(1, int(fixed_batch_size))] if fixed_batch_size is not None else _normalize_batch_size_choices(batch_size_choices)
    search_distributions = _objective_distributions(batch_choices)
    narrowed_params = [payload for payload in (narrow_from_params or []) if isinstance(payload, dict) and payload]

    def _seed_values(name: str) -> list[Any]:
        values: list[Any] = []
        for payload in narrowed_params:
            if name in payload:
                values.append(payload[name])
        return values

    def _suggest_int(trial: optuna.Trial, name: str, low: int, high: int, pad: int = 1) -> int:
        seed_values = [
            int(v)
            for v in _seed_values(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if seed_values:
            narrowed_low = max(low, min(seed_values) - pad)
            narrowed_high = min(high, max(seed_values) + pad)
            if narrowed_low >= narrowed_high:
                return int(narrowed_low)
            return int(trial.suggest_int(name, narrowed_low, narrowed_high))
        return int(trial.suggest_int(name, low, high))

    def _suggest_float(
        trial: optuna.Trial,
        name: str,
        low: float,
        high: float,
        *,
        log: bool = False,
        abs_pad: float = 0.05,
        log_low_mult: float = 0.6,
        log_high_mult: float = 1.6,
    ) -> float:
        seed_values = [
            float(v)
            for v in _seed_values(name)
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ]
        if seed_values:
            if log:
                narrowed_low = max(low, min(seed_values) * log_low_mult)
                narrowed_high = min(high, max(seed_values) * log_high_mult)
            else:
                narrowed_low = max(low, min(seed_values) - abs_pad)
                narrowed_high = min(high, max(seed_values) + abs_pad)
            if math.isfinite(narrowed_low) and math.isfinite(narrowed_high) and narrowed_low < narrowed_high:
                return float(trial.suggest_float(name, narrowed_low, narrowed_high, log=log))
        return float(trial.suggest_float(name, low, high, log=log))

    def _suggest_categorical(trial: optuna.Trial, name: str, choices: list[Any]) -> Any:
        normalized_choices = list(choices)
        if not normalized_choices:
            raise ValueError(f"No choices available for parameter {name}")
        seeded = [value for value in _seed_values(name) if value in normalized_choices]
        deduped_seeded: list[Any] = []
        for value in seeded:
            if value not in deduped_seeded:
                deduped_seeded.append(value)
        if len(deduped_seeded) == 1 and len(normalized_choices) > 1:
            for fallback in normalized_choices:
                if fallback != deduped_seeded[0]:
                    deduped_seeded.append(fallback)
                    break
        choices_to_use = deduped_seeded if deduped_seeded else normalized_choices
        return trial.suggest_categorical(name, choices_to_use)


    def objective(trial: optuna.Trial) -> float:
        parser = build_lnn_parser()
        args = parser.parse_args([])
        args.data_path = str(dataset)

        # Core hyperparameters
        args.epochs = _suggest_int(trial, "epochs", 8, 15, pad=2)
        args.batch_size = int(_suggest_categorical(trial, "batch_size", batch_choices))
        args.hidden_size = int(_suggest_categorical(trial, "hidden_size", [64, 128, 256]))
        args.dropout = _suggest_float(trial, "dropout", 0.05, 0.4, abs_pad=0.08)
        args.lr = _suggest_float(trial, "lr", 1e-4, 5e-3, log=True, log_low_mult=0.5, log_high_mult=1.8)
        args.weight_decay = _suggest_float(
            trial,
            "weight_decay",
            1e-6,
            1e-2,
            log=True,
            log_low_mult=0.4,
            log_high_mult=2.2,
        )
        args.seq_len = int(_suggest_categorical(trial, "seq_len", [32, 64]))
        args.horizon = int(_suggest_categorical(trial, "horizon", [10, 15, 30]))

        # Architecture hyperparameters
        args.num_heads = int(_suggest_categorical(trial, "num_heads", [2, 4]))
        args.num_layers = _suggest_int(trial, "num_layers", 1, 2, pad=1)
        args.use_attention = bool(_suggest_categorical(trial, "use_attention", [True, False]))
        args.backbone_type = str(_suggest_categorical(trial, "backbone_type", ["mamba"]))
        args.cnn_frontend = str(_suggest_categorical(trial, "cnn_frontend", ["inception", "none"]))
        args.mamba_d_state = int(_suggest_categorical(trial, "mamba_d_state", [8, 16, 32]))
        args.mamba_d_conv = int(_suggest_categorical(trial, "mamba_d_conv", [2, 4]))
        args.mamba_expand = int(_suggest_categorical(trial, "mamba_expand", [2, 3]))

        # Training strategy
        args.scheduler = str(_suggest_categorical(trial, "scheduler", ["cosine", "cosine_warm", "plateau"]))
        args.warmup_epochs = _suggest_int(trial, "warmup_epochs", 0, 1, pad=1)
        args.use_ema = bool(_suggest_categorical(trial, "use_ema", [True, False]))
        args.ema_decay = (
            _suggest_float(
                trial,
                "ema_decay",
                0.99,
                0.9999,
                log=True,
                log_low_mult=0.995,
                log_high_mult=1.005,
            )
            if args.use_ema
            else 0.999
        )

        # Keep concurrent 3-worker runs inside 6GB-class GPU limits without changing study schema.
        requested_batch_size = int(args.batch_size)
        effective_batch_size = requested_batch_size
        if workers >= 3 and requested_batch_size > 256:
            effective_batch_size = 256
            args.batch_size = effective_batch_size

        trial.set_user_attr("requested_batch_size", requested_batch_size)
        trial.set_user_attr("effective_batch_size", effective_batch_size)
        trial.set_user_attr("batch_size_choices", [int(v) for v in batch_choices])

        # Hard stop policy: trials are short by design (3-5 epochs max).
        args.early_stopping_patience = 3
        args.seed = 42
        args.cpu = False
        args.grad_clip_norm = 1.0

        bundle = build_supervised_data(sampled_optuna, seq_len=args.seq_len, horizon=args.horizon, include_stoch_rsi=True)
        splits = list(anchored_walk_forward_splits(bundle, n_splits=5))
        
        all_split_sharpes = []
        all_split_losses = []
        all_split_dir_accs = []
        all_split_drawdowns = []
        
        enable_performance_mode()
        
        for split_idx, split in enumerate(splits):
            split_sharpes = []
            
            # Seed rotation for robustness (3 seeds)
            for seed in [42, 137, 256]:
                args.seed = seed
                best_epoch_val = float("inf")
                no_improve_epochs = 0
                
                def _on_epoch_end(
                    epoch: int,
                    _train: float,
                    val: float,
                    _dir_acc: float,
                    _epoch_sharpe: float,
                    _lr: float,
                ) -> None:
                    nonlocal best_epoch_val, no_improve_epochs
                    if val + 1e-9 < best_epoch_val:
                        best_epoch_val = val
                        no_improve_epochs = 0
                    else:
                        no_improve_epochs += 1

                    if not math.isfinite(val):
                        raise optuna.TrialPruned("Validation loss is non-finite")

                    # Hard catastrophic epoch-level gates
                    if epoch >= 2 and float(_dir_acc) < (_TARGET_DIRECTIONAL_ACC - 0.04):
                        raise optuna.TrialPruned(f"Epoch-{epoch} directional accuracy too weak ({float(_dir_acc):.3f})")
                    if epoch >= 2 and float(_epoch_sharpe) < (_TARGET_SHARPE_MIN - 1.5):
                        raise optuna.TrialPruned(f"Epoch-{epoch} Sharpe too weak ({float(_epoch_sharpe):.3f})")
                        
                    # Let Optuna pruner inspect at split-level, so we only raise Hard Kills here.
                
                try:
                    metrics = run_training(args, on_epoch_end=_on_epoch_end, split_bundle=split)
                    
                    cpcv = metrics.get("cpcv", {}) if isinstance(metrics.get("cpcv", {}), dict) else {}
                    sharpe = float(cpcv.get("sharpe_mean", 0.0))
                    val_loss = float(metrics["best_val_loss"])
                    dir_acc = float(metrics.get("val_directional_accuracy", 0.0))
                    max_drawdown_ratio = float(metrics.get("max_drawdown_ratio", 1.0))

                    if (not math.isfinite(max_drawdown_ratio)) or max_drawdown_ratio > _MAX_DRAWDOWN_GUARD:
                        raise optuna.TrialPruned(
                            f"Max drawdown guard triggered: {max_drawdown_ratio:.3f} > {_MAX_DRAWDOWN_GUARD:.2f}"
                        )
                    
                    split_sharpes.append(sharpe)
                    all_split_losses.append(val_loss)
                    all_split_dir_accs.append(dir_acc)
                    all_split_drawdowns.append(max_drawdown_ratio)
                    
                except Exception as exc:
                    if is_cuda_oom(exc):
                        logger.warning("Trial %d pruned due to CUDA OOM", trial.number)
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        gc.collect()
                        raise optuna.TrialPruned("Pruned after CUDA OOM") from exc
                    raise

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            # Evaluate split robustly (average across seeds)
            avg_split_sharpe = float(sum(split_sharpes) / len(split_sharpes))
            all_split_sharpes.append(avg_split_sharpe)
            
            # Pruning score is minimized
            accuracy_penalty = max(_TARGET_DIRECTIONAL_ACC - float(all_split_dir_accs[-1]), 0.0)
            sharpe_penalty = max(_TARGET_SHARPE_MIN - avg_split_sharpe, 0.0)
            prune_score = float(all_split_losses[-1]) + accuracy_penalty * 0.75 + sharpe_penalty * 0.15
            
            # Prune at the SPLIT level, allowing Hyperband to allocate splits dynamically
            trial.report(prune_score, step=split_idx)
            if trial.should_prune():
                raise optuna.TrialPruned(f"Optuna pruner cut trial at split {split_idx}")

        import numpy as np
        # Primary Objective is 5th Percentile Sharpe (Robustness across Walk-Forward Regimes)
        percentile_5_sharpe = float(np.percentile(all_split_sharpes, 5))
        avg_val_loss = float(np.mean(all_split_losses))
        avg_dir_acc = float(np.mean(all_split_dir_accs))
        worst_drawdown_ratio = float(np.max(all_split_drawdowns)) if all_split_drawdowns else 0.0

        # Deflated Sharpe Multiple Testing Haircut
        completed_trials = len([t for t in trial.study.trials if t.state == TrialState.COMPLETE])
        gamma = 0.5772 # Euler-Mascheroni constant
        # Expected max of N standard normals approximation
        e_max = math.sqrt(2 * math.log(max(completed_trials, 1))) if completed_trials > 0 else 0
        haircut_factor = math.sqrt(max(1.0 - gamma * (e_max / max(1.0, percentile_5_sharpe)), 0.01))
        deflated_sharpe = percentile_5_sharpe * haircut_factor

        penalty = 0.0
        if avg_dir_acc < _TARGET_DIRECTIONAL_ACC:
            penalty += (_TARGET_DIRECTIONAL_ACC - avg_dir_acc) * 12.0 # Reduced from 24.0 due to strict CV

        if deflated_sharpe < _TARGET_SHARPE_MIN:
            penalty += (_TARGET_SHARPE_MIN - deflated_sharpe) * 0.30
        else:
            sharpe_bonus = min(deflated_sharpe, _TARGET_SHARPE_STRETCH) - _TARGET_SHARPE_MIN
            penalty -= sharpe_bonus * 0.10

        meets_target = bool(
            deflated_sharpe >= _TARGET_SHARPE_MIN
            and avg_dir_acc > (_TARGET_DIRECTIONAL_ACC + _TARGET_DIRECTIONAL_ACC_STRICT_EPS)
        )
        base_objective_score = avg_val_loss + penalty
        
        plateau = compute_plateau_penalty(
            study=trial.study,
            current_params=trial.params,
            current_trial_number=trial.number,
            base_score=base_objective_score,
            direction="minimize",
            distributions=search_distributions,
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
            
        objective_score = base_objective_score + float(plateau["penalty"])

        trial.set_user_attr("best_val_loss", avg_val_loss)
        trial.set_user_attr("val_directional_accuracy", avg_dir_acc)
        trial.set_user_attr("cpcv_sharpe_mean", percentile_5_sharpe)
        trial.set_user_attr("cpcv_pbo_proxy", 0.0) # Disabled proxy in favor of Deflated Sharpe
        trial.set_user_attr("effective_sharpe_for_objective", deflated_sharpe)
        trial.set_user_attr("pbo_sharpe_penalty", 0.0)
        trial.set_user_attr("meets_target", meets_target)
        trial.set_user_attr("max_drawdown_ratio", worst_drawdown_ratio)
        trial.set_user_attr("best_epoch", -1)
        trial.set_user_attr("base_objective_score", float(base_objective_score))
        trial.set_user_attr("plateau_penalty", float(plateau["penalty"]))
        trial.set_user_attr("plateau_neighbor_count", int(plateau["neighbor_count"]))
        trial.set_user_attr("plateau_neighbor_median", float(plateau["local_median"]))
        trial.set_user_attr("plateau_neighbor_iqr", float(plateau["local_iqr"]))
        trial.set_user_attr("plateau_cliff_gap", float(plateau["cliff_gap"]))
        trial.set_user_attr("plateau_reason", str(plateau["reason"]))
        trial.set_user_attr("objective_score", float(objective_score))
        return float(objective_score)

    selected_pruner = str(pruner).strip().lower()
    study = optuna.create_study(
        direction="minimize",
        study_name=study_name,
        pruner=_build_pruner(selected_pruner),
        sampler=optuna.samplers.TPESampler(seed=42),
        storage=storage_backend,
        load_if_exists=True,
    )
    if bootstrap_manual_checkpoint:
        _bootstrap_from_manual_checkpoint(study, search_distributions)
    seeded_trials_added = _seed_study_with_completed_trials(study, search_distributions, seed_trials)
    finished_states = {TrialState.COMPLETE, TrialState.PRUNED, TrialState.FAIL}
    existing_finished_trials = len([t for t in study.trials if t.state in finished_states])
    remaining_trials = max(trials - existing_finished_trials, 0)

    pre_run_archive = _archive_current_best_study(study, storage_reference, reason="pre_run")
    _write_runtime_checkpoint(study, storage_reference)

    if remaining_trials > 0:
        study.optimize(
            objective,
            n_trials=remaining_trials,
            n_jobs=max(1, int(workers)),
            timeout=None if timeout <= 0 else timeout,
            catch=(Exception,),
            callbacks=[lambda s, _t: _write_runtime_checkpoint(s, storage_reference)],
        )
    else:
        logger.info(
            "Requested %d total Optuna trials; study already has %d finished trials",
            trials,
            existing_finished_trials,
        )

    _write_runtime_checkpoint(study, storage_reference)
    post_run_archive = _archive_current_best_study(study, storage_reference, reason="post_run")

    complete_trials = [t for t in study.trials if t.state == TrialState.COMPLETE]

    payload = {
        "best_value": float(study.best_value) if complete_trials else None,
        "best_params": study.best_params if complete_trials else {},
        "best_val_loss": (
            float(study.best_trial.user_attrs.get("best_val_loss", 0.0)) if complete_trials else None
        ),
        "best_sharpe_ratio": (
            float(study.best_trial.user_attrs.get("cpcv_sharpe_mean", 0.0)) if complete_trials else None
        ),
        "best_effective_sharpe": (
            float(study.best_trial.user_attrs.get("effective_sharpe_for_objective", 0.0)) if complete_trials else None
        ),
        "best_pbo_proxy": (
            float(study.best_trial.user_attrs.get("cpcv_pbo_proxy", 1.0)) if complete_trials else None
        ),
        "best_directional_accuracy": (
            float(study.best_trial.user_attrs.get("val_directional_accuracy", 0.0)) if complete_trials else None
        ),
        "best_max_drawdown_ratio": (
            float(study.best_trial.user_attrs.get("max_drawdown_ratio", 0.0)) if complete_trials else None
        ),
        "best_meets_target": bool(study.best_trial.user_attrs.get("meets_target", False)) if complete_trials else False,
        "study_name": study.study_name,
        "n_trials_completed": len(complete_trials),
        "n_trials_total": len(study.trials),
        "workers": max(1, int(workers)),
        "fixed_batch_size": int(fixed_batch_size) if fixed_batch_size is not None else None,
        "batch_size_choices": [int(v) for v in batch_choices],
        "pruner": selected_pruner,
        "narrow_from_params_count": len(narrowed_params),
        "seeded_trials_added": int(seeded_trials_added),
        "storage": storage_reference,
        "pre_run_archive": pre_run_archive,
        "post_run_archive": post_run_archive,
    }
    write_json(model_dir / "optuna_best.json", payload)
    if complete_trials:
        logger.info("Optuna best value: %.6f with params: %s", study.best_value, study.best_params)
    else:
        logger.warning("Optuna finished without completed trials")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for LNN")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--study-name", type=str, default="modelmk1_lnn_optuna")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--pruner", type=str, default=_DEFAULT_PRUNER, choices=list(_SUPPORTED_PRUNERS))
    parser.add_argument("--storage-url", type=str, default=None)
    parser.add_argument("--storage-db-path", type=str, default=None)
    args = parser.parse_args()
    print(
        run_optuna(
            args.trials,
            args.timeout,
            args.data_path,
            args.study_name,
            workers=args.workers,
            pruner=args.pruner,
            storage_url=args.storage_url,
            storage_db_path=args.storage_db_path,
        )
    )


if __name__ == "__main__":
    main()
