from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from modelmk1.common.paths import get_app_paths
from modelmk1.common.runtime import configure_logging, write_json
from modelmk1.tuning.optuna_lnn import run_optuna

logger = logging.getLogger(__name__)


def _parse_batch_sizes(raw: str) -> tuple[int, int]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if len(values) != 2:
        raise ValueError("--batch-sizes must contain exactly two comma-separated integers, e.g. 64,96")
    first = max(1, int(values[0]))
    second = max(1, int(values[1]))
    if first == second:
        raise ValueError("Batch sizes must be different for staged OOM-safe search")
    return first, second


def _summary_view(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "study_name": payload.get("study_name"),
        "best_value": payload.get("best_value"),
        "best_sharpe_ratio": payload.get("best_sharpe_ratio"),
        "best_directional_accuracy": payload.get("best_directional_accuracy"),
        "best_meets_target": payload.get("best_meets_target"),
        "n_trials_completed": payload.get("n_trials_completed"),
        "n_trials_total": payload.get("n_trials_total"),
        "fixed_batch_size": payload.get("fixed_batch_size"),
        "batch_size_choices": payload.get("batch_size_choices"),
        "seeded_trials_added": payload.get("seeded_trials_added"),
    }


def main() -> None:
    configure_logging()

    parser = argparse.ArgumentParser(description="Run staged OOM-safe Optuna search for LNN")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--stage1-workers", type=int, default=None)
    parser.add_argument("--stage23-workers", type=int, default=None)
    parser.add_argument("--stage-trials", type=int, default=30)
    parser.add_argument("--timeout", type=int, default=0)
    parser.add_argument("--batch-sizes", type=str, default="64,96")
    parser.add_argument("--study-prefix", type=str, default="modelmk1_lnn_oomsafe")
    args = parser.parse_args()

    batch_size_a, batch_size_b = _parse_batch_sizes(args.batch_sizes)
    base_workers = max(1, int(args.workers))
    stage1_workers = max(1, int(args.stage1_workers)) if args.stage1_workers is not None else base_workers
    stage23_workers = max(1, int(args.stage23_workers)) if args.stage23_workers is not None else base_workers
    stage_trials = max(1, int(args.stage_trials))

    study_a = f"{args.study_prefix}_bs{batch_size_a}"
    study_b = f"{args.study_prefix}_bs{batch_size_b}"
    study_focus = f"{args.study_prefix}_focus"

    logger.info(
        "Stage 1/3: study=%s fixed batch=%s trials=%s workers=%s",
        study_a,
        batch_size_a,
        stage_trials,
        stage1_workers,
    )
    stage_a = run_optuna(
        trials=stage_trials,
        timeout=args.timeout,
        data_path=args.data_path,
        study_name=study_a,
        workers=stage1_workers,
        fixed_batch_size=batch_size_a,
        batch_size_choices=[batch_size_a],
        narrow_from_params=None,
        seed_trials=None,
        bootstrap_manual_checkpoint=False,
    )

    logger.info(
        "Stage 2/3: study=%s fixed batch=%s trials=%s workers=%s",
        study_b,
        batch_size_b,
        stage_trials,
        stage23_workers,
    )
    stage_b = run_optuna(
        trials=stage_trials,
        timeout=args.timeout,
        data_path=args.data_path,
        study_name=study_b,
        workers=stage23_workers,
        fixed_batch_size=batch_size_b,
        batch_size_choices=[batch_size_b],
        narrow_from_params=None,
        seed_trials=None,
        bootstrap_manual_checkpoint=False,
    )

    candidates: list[dict[str, Any]] = []
    for stage_name, payload in (("stage_a", stage_a), ("stage_b", stage_b)):
        params = payload.get("best_params") if isinstance(payload.get("best_params"), dict) else {}
        value = payload.get("best_value")
        if params and value is not None:
            candidates.append(
                {
                    "stage": stage_name,
                    "study_name": payload.get("study_name"),
                    "best_value": float(value),
                    "best_params": params,
                }
            )

    if not candidates:
        raise RuntimeError("Neither fixed-batch stage produced a valid complete trial")

    candidates.sort(key=lambda item: float(item["best_value"]))
    top_candidates = candidates[:2]

    seed_trials: list[dict[str, Any]] = []
    focus_params: list[dict[str, Any]] = []
    focus_batch_choices: set[int] = set()
    for candidate in top_candidates:
        params = dict(candidate["best_params"])
        if "batch_size" in params:
            focus_batch_choices.add(int(params["batch_size"]))
        focus_params.append(params)
        seed_trials.append(
            {
                "params": params,
                "value": float(candidate["best_value"]),
                "user_attrs": {
                    "source_stage": str(candidate["stage"]),
                    "source_study": str(candidate["study_name"]),
                },
            }
        )

    if not focus_batch_choices:
        focus_batch_choices = {batch_size_a, batch_size_b}

    focus_total_trials = stage_trials + len(seed_trials)
    logger.info(
        "Stage 3/3: study=%s focused trials=%s (includes %s seeded best trials) workers=%s batch_choices=%s",
        study_focus,
        focus_total_trials,
        len(seed_trials),
        stage23_workers,
        sorted(focus_batch_choices),
    )
    stage_focus = run_optuna(
        trials=focus_total_trials,
        timeout=args.timeout,
        data_path=args.data_path,
        study_name=study_focus,
        workers=stage23_workers,
        fixed_batch_size=None,
        batch_size_choices=sorted(focus_batch_choices),
        narrow_from_params=focus_params,
        seed_trials=seed_trials,
        bootstrap_manual_checkpoint=False,
    )

    all_stages = [stage_a, stage_b, stage_focus]
    valid_stage_bests = [payload for payload in all_stages if payload.get("best_value") is not None]
    if valid_stage_bests:
        final_best = min(valid_stage_bests, key=lambda payload: float(payload["best_value"]))
    else:
        final_best = {}

    summary: dict[str, Any] = {
        "request": {
            "stage_trials": stage_trials,
            "workers_stage1": stage1_workers,
            "workers_stage23": stage23_workers,
            "batch_sizes": [batch_size_a, batch_size_b],
            "focus_batch_choices": sorted(focus_batch_choices),
            "timeout": int(args.timeout),
            "study_prefix": args.study_prefix,
            "seed_trials_count": len(seed_trials),
        },
        "stages": {
            "stage_a": _summary_view(stage_a),
            "stage_b": _summary_view(stage_b),
            "stage_focus": _summary_view(stage_focus),
        },
        "final_best": {
            "study_name": final_best.get("study_name"),
            "best_value": final_best.get("best_value"),
            "best_params": final_best.get("best_params", {}),
            "best_sharpe_ratio": final_best.get("best_sharpe_ratio"),
            "best_directional_accuracy": final_best.get("best_directional_accuracy"),
            "best_meets_target": final_best.get("best_meets_target"),
        },
    }

    out_path = get_app_paths().outputs / "model" / "optuna_lnn_staged_oomsafe_summary.json"
    write_json(out_path, summary)
    print(summary)


if __name__ == "__main__":
    main()
