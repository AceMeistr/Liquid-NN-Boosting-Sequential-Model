from __future__ import annotations

import math
from statistics import median
from typing import Any, Sequence


def _interpolated_percentile(sorted_values: Sequence[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    q_clamped = min(max(float(q), 0.0), 1.0)
    pos = (len(sorted_values) - 1) * q_clamped
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_values[lo])
    frac = pos - lo
    return float(sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac)


def _normalized_param_value(
    params: dict[str, Any],
    name: str,
    distribution: Any,
) -> float | None:
    if name not in params:
        return None

    raw_value = params[name]
    dist_type = type(distribution).__name__

    try:
        if dist_type == "CategoricalDistribution":
            choices = list(getattr(distribution, "choices", []))
            if not choices or raw_value not in choices:
                return None
            if len(choices) == 1:
                return 0.0
            return float(choices.index(raw_value)) / float(len(choices) - 1)

        if dist_type == "IntDistribution":
            low = float(getattr(distribution, "low"))
            high = float(getattr(distribution, "high"))
            value = float(int(raw_value))
            if value < low or value > high:
                return None
            span = max(high - low, 1.0)
            return (value - low) / span

        if dist_type == "FloatDistribution":
            low = float(getattr(distribution, "low"))
            high = float(getattr(distribution, "high"))
            value = float(raw_value)
            if value < low or value > high:
                return None
            is_log = bool(getattr(distribution, "log", False))
            if is_log and low > 0.0 and high > 0.0 and value > 0.0:
                low_log = math.log(low)
                high_log = math.log(high)
                span = max(high_log - low_log, 1e-12)
                return (math.log(value) - low_log) / span
            span = max(high - low, 1e-12)
            return (value - low) / span
    except Exception:  # noqa: BLE001
        return None

    return None


def _params_distance(
    params_a: dict[str, Any],
    params_b: dict[str, Any],
    distributions: dict[str, Any],
) -> float:
    sq_diffs: list[float] = []
    for name, distribution in distributions.items():
        a_norm = _normalized_param_value(params_a, name, distribution)
        b_norm = _normalized_param_value(params_b, name, distribution)
        if a_norm is None or b_norm is None:
            continue
        diff = a_norm - b_norm
        sq_diffs.append(diff * diff)

    if not sq_diffs:
        return float("nan")
    return float(math.sqrt(sum(sq_diffs) / float(len(sq_diffs))))


def _extract_trial_score(trial: Any, score_attr_names: tuple[str, ...]) -> float | None:
    for attr_name in score_attr_names:
        if attr_name in trial.user_attrs:
            try:
                score = float(trial.user_attrs[attr_name])
                if math.isfinite(score):
                    return score
            except Exception:  # noqa: BLE001
                continue
    if trial.value is None:
        return None
    try:
        score = float(trial.value)
        if math.isfinite(score):
            return score
    except Exception:  # noqa: BLE001
        return None
    return None


def compute_plateau_penalty(
    *,
    study: Any,
    current_params: dict[str, Any],
    current_trial_number: int,
    base_score: float,
    direction: str,
    distributions: dict[str, Any],
    max_neighbors: int = 8,
    min_local_neighbors: int = 4,
    radius: float = 0.30,
    grace_completed_trials: int = 12,
    cliff_weight: float = 0.35,
    spread_weight: float = 0.10,
    score_attr_names: tuple[str, ...] = ("base_objective_score", "objective_score"),
) -> dict[str, float | int | str]:
    direction_normalized = str(direction).strip().lower()
    if direction_normalized not in {"minimize", "maximize"}:
        raise ValueError("direction must be either 'minimize' or 'maximize'")

    completed = [
        trial
        for trial in study.trials
        if str(getattr(getattr(trial, "state", None), "name", "")).upper() == "COMPLETE"
    ]
    if len(completed) < max(int(grace_completed_trials), 3):
        return {
            "penalty": 0.0,
            "neighbor_count": 0,
            "local_median": 0.0,
            "local_iqr": 0.0,
            "local_best": 0.0,
            "coverage": 0.0,
            "cliff_gap": 0.0,
            "reason": "insufficient_completed_trials",
        }

    local_candidates: list[tuple[float, float]] = []
    for trial in completed:
        if trial.number == current_trial_number:
            continue
        score = _extract_trial_score(trial, score_attr_names)
        if score is None:
            continue
        dist = _params_distance(current_params, trial.params, distributions)
        if not math.isfinite(dist):
            continue
        local_candidates.append((dist, score))

    if len(local_candidates) < max(int(min_local_neighbors), 3):
        return {
            "penalty": 0.0,
            "neighbor_count": 0,
            "local_median": 0.0,
            "local_iqr": 0.0,
            "local_best": 0.0,
            "coverage": 0.0,
            "cliff_gap": 0.0,
            "reason": "insufficient_neighbors",
        }

    local_candidates.sort(key=lambda item: item[0])
    scores_in_radius = [score for dist, score in local_candidates if dist <= float(radius)]
    if len(scores_in_radius) < max(int(min_local_neighbors), 3):
        scores_in_radius = [score for _, score in local_candidates[: max(int(max_neighbors), int(min_local_neighbors), 3)]]

    if len(scores_in_radius) < max(int(min_local_neighbors), 3):
        return {
            "penalty": 0.0,
            "neighbor_count": len(scores_in_radius),
            "local_median": 0.0,
            "local_iqr": 0.0,
            "local_best": 0.0,
            "coverage": 0.0,
            "cliff_gap": 0.0,
            "reason": "insufficient_local_scores",
        }

    scores_sorted = sorted(float(score) for score in scores_in_radius)
    local_median = float(median(scores_sorted))
    q1 = _interpolated_percentile(scores_sorted, 0.25)
    q3 = _interpolated_percentile(scores_sorted, 0.75)
    local_iqr = max(float(q3 - q1), 0.0)
    local_best = min(scores_sorted) if direction_normalized == "minimize" else max(scores_sorted)

    if direction_normalized == "minimize":
        cliff_gap = max(local_median - float(base_score), 0.0)
    else:
        cliff_gap = max(float(base_score) - local_median, 0.0)

    coverage = min(len(scores_sorted) / float(max(int(max_neighbors), 1)), 1.0)
    penalty = (cliff_gap * float(cliff_weight) + local_iqr * float(spread_weight)) * coverage

    return {
        "penalty": float(max(penalty, 0.0)),
        "neighbor_count": int(len(scores_sorted)),
        "local_median": float(local_median),
        "local_iqr": float(local_iqr),
        "local_best": float(local_best),
        "coverage": float(coverage),
        "cliff_gap": float(cliff_gap),
        "reason": "ok",
    }
