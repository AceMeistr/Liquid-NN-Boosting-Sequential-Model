from __future__ import annotations

import numpy as np

from modelmk1.eval.cpcv import CPCVConfig, evaluate_cpcv_distribution


def test_cpcv_distribution_basic() -> None:
    rng = np.random.default_rng(7)
    y_true = rng.normal(0.0, 1.0, 1024)
    y_pred = y_true + rng.normal(0.0, 0.25, 1024)

    summary = evaluate_cpcv_distribution(
        y_true,
        y_pred,
        config=CPCVConfig(
            n_groups=8,
            test_group_size=4,
            max_combinations=30,
            purge_gap=4,
            embargo=2,
            seed=42,
        ),
    )

    assert summary["n_combinations_evaluated"] > 0
    assert 0.0 <= summary["directional_accuracy_mean"] <= 1.0
    assert 0.0 <= summary["pbo_proxy"] <= 1.0


def test_cpcv_distribution_insufficient_samples() -> None:
    y_true = np.array([1.0, -1.0, 1.0])
    y_pred = np.array([0.8, -0.5, 0.2])

    summary = evaluate_cpcv_distribution(
        y_true,
        y_pred,
        config=CPCVConfig(n_groups=4, test_group_size=2, max_combinations=5),
    )

    assert summary["n_combinations_evaluated"] == 0
