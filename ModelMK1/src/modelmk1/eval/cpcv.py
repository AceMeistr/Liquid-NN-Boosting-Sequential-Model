from __future__ import annotations

import itertools
from dataclasses import dataclass

import math
import numpy as np


@dataclass(frozen=True)
class CPCVConfig:
    n_groups: int = 16
    test_group_size: int = 8
    max_combinations: int = 200
    purge_gap: int = 30
    embargo: int = 15
    seed: int = 42


def _group_edges(n_samples: int, n_groups: int) -> np.ndarray:
    if n_groups < 2:
        raise ValueError("n_groups must be >= 2")
    if n_samples < n_groups:
        raise ValueError("n_samples must be >= n_groups")
    return np.linspace(0, n_samples, n_groups + 1, dtype=np.int64)


def _sample_combinations(n_groups: int, test_group_size: int, max_combinations: int, seed: int) -> list[tuple[int, ...]]:
    if test_group_size <= 0 or test_group_size >= n_groups:
        raise ValueError("test_group_size must be in [1, n_groups-1]")

    all_combos = list(itertools.combinations(range(n_groups), test_group_size))
    if max_combinations <= 0 or len(all_combos) <= max_combinations:
        return all_combos

    rng = np.random.default_rng(seed)
    pick_idx = rng.choice(len(all_combos), size=max_combinations, replace=False)
    return [all_combos[int(i)] for i in pick_idx]


def _test_mask_with_boundary_trim(
    n_samples: int,
    group_edges: np.ndarray,
    test_groups: tuple[int, ...],
    trim: int,
) -> np.ndarray:
    mask = np.zeros(n_samples, dtype=bool)
    test_set = set(test_groups)

    for g in test_groups:
        start = int(group_edges[g])
        end = int(group_edges[g + 1])
        left = start
        right = end

        if g - 1 not in test_set:
            left = min(left + trim, right)
        if g + 1 not in test_set:
            right = max(right - trim, left)

        if right > left:
            mask[left:right] = True
    return mask


def evaluate_cpcv_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    config: CPCVConfig | None = None,
) -> dict:
    cfg = config or CPCVConfig()

    y_true = np.asarray(y_true, dtype=np.float64).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)

    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must have identical shapes")
    if y_true.size < cfg.n_groups:
        return {
            "n_combinations_evaluated": 0,
            "directional_accuracy_mean": 0.0,
            "directional_accuracy_std": 0.0,
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "pbo_proxy": 1.0,
            "note": "insufficient_samples_for_cpcv",
        }

    combos = _sample_combinations(cfg.n_groups, cfg.test_group_size, cfg.max_combinations, cfg.seed)
    edges = _group_edges(y_true.size, cfg.n_groups)
    boundary_trim = max(cfg.purge_gap + cfg.embargo, 0)

    dir_acc_values: list[float] = []
    sharpe_values: list[float] = []

    for combo in combos:
        mask = _test_mask_with_boundary_trim(y_true.size, edges, combo, boundary_trim)
        if int(mask.sum()) < max(cfg.n_groups, 16):
            continue

        yt = y_true[mask]
        yp = y_pred[mask]
        pnl = np.sign(yp) * yt

        dir_acc = float((np.sign(yt) == np.sign(yp)).mean())
        pnl_std = float(pnl.std())
        sharpe = 0.0 if pnl_std <= 1e-12 else float(pnl.mean() / (pnl_std + 1e-12) * np.sqrt(252.0))

        dir_acc_values.append(dir_acc)
        sharpe_values.append(sharpe)

    if not dir_acc_values:
        return {
            "n_combinations_evaluated": 0,
            "directional_accuracy_mean": 0.0,
            "directional_accuracy_std": 0.0,
            "sharpe_mean": 0.0,
            "sharpe_std": 0.0,
            "pbo_proxy": 1.0,
            "note": "all_combinations_filtered_after_purge_embargo",
        }

    dir_arr = np.asarray(dir_acc_values, dtype=np.float64)
    sharpe_arr = np.asarray(sharpe_values, dtype=np.float64)

    return {
        "n_combinations_evaluated": int(dir_arr.size),
        "n_groups": cfg.n_groups,
        "test_group_size": cfg.test_group_size,
        "max_combinations": cfg.max_combinations,
        "purge_gap": cfg.purge_gap,
        "embargo": cfg.embargo,
        "directional_accuracy_mean": float(dir_arr.mean()),
        "directional_accuracy_std": float(dir_arr.std()),
        "directional_accuracy_ci95_low": float(np.percentile(dir_arr, 2.5)),
        "directional_accuracy_ci95_high": float(np.percentile(dir_arr, 97.5)),
        "sharpe_mean": float(sharpe_arr.mean()),
        "sharpe_std": float(sharpe_arr.std()),
        "sharpe_ci95_low": float(np.percentile(sharpe_arr, 2.5)),
        "sharpe_ci95_high": float(np.percentile(sharpe_arr, 97.5)),
        # Bailey-Marcos-López Theoretical PBO (CDF normal fit at 0)
        "pbo_proxy": float(0.5 * (1 + math.erf((0.0 - float(sharpe_arr.mean())) / (float(sharpe_arr.std()) * math.sqrt(2))))) if float(sharpe_arr.std()) > 1e-12 else (1.0 if float(sharpe_arr.mean()) < 0.0 else 0.0),
    }
