from __future__ import annotations

<<<<<<< HEAD
from pathlib import Path
=======
import logging
from pathlib import Path
from typing import Any
>>>>>>> main

import numpy as np
import xgboost as xgb

<<<<<<< HEAD
=======
logger = logging.getLogger(__name__)

>>>>>>> main

def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
    use_gpu: bool = False,
<<<<<<< HEAD
) -> xgb.XGBRegressor:
    defaults = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
=======
    feature_names: list[str] | None = None,
) -> xgb.XGBRegressor:
    defaults: dict[str, Any] = {
        "n_estimators": 800,
        "max_depth": 7,
        "learning_rate": 0.025,
>>>>>>> main
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-3,
        "reg_lambda": 1.0,
<<<<<<< HEAD
=======
        "min_child_weight": 3,
        "gamma": 0.1,
>>>>>>> main
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
<<<<<<< HEAD
=======
        "early_stopping_rounds": 50,
>>>>>>> main
    }
    if params:
        defaults.update(params)

    if use_gpu:
        defaults.update({"tree_method": "hist", "device": "cuda"})

<<<<<<< HEAD
    model = xgb.XGBRegressor(**defaults)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


=======
    early_stopping = defaults.pop("early_stopping_rounds", 50)

    defaults["early_stopping_rounds"] = early_stopping
    model = xgb.XGBRegressor(**defaults)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        verbose=False,
    )

    if feature_names is not None:
        _log_feature_importance(model, feature_names)

    return model


def _log_feature_importance(model: xgb.XGBRegressor, names: list[str], top_k: int = 10) -> None:
    importance = model.feature_importances_
    if len(importance) != len(names):
        return
    ranked = sorted(zip(names, importance), key=lambda x: x[1], reverse=True)
    logger.info("Top-%d XGBoost feature importances:", top_k)
    for name, score in ranked[:top_k]:
        logger.info("  %-25s %.4f", name, score)


def get_feature_importance(model: xgb.XGBRegressor, names: list[str]) -> dict[str, float]:
    importance = model.feature_importances_
    if len(importance) != len(names):
        return {}
    return dict(sorted(zip(names, importance.tolist()), key=lambda x: x[1], reverse=True))


>>>>>>> main
def save_xgb(model: xgb.XGBRegressor, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(target)


def load_xgb(path: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(Path(path))
    return model
