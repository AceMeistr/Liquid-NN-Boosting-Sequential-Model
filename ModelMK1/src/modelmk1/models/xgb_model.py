from __future__ import annotations

from pathlib import Path

import numpy as np
import xgboost as xgb


def train_xgb(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    params: dict | None = None,
    use_gpu: bool = False,
) -> xgb.XGBRegressor:
    defaults = {
        "n_estimators": 600,
        "max_depth": 6,
        "learning_rate": 0.03,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 1e-3,
        "reg_lambda": 1.0,
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "n_jobs": -1,
        "random_state": 42,
    }
    if params:
        defaults.update(params)

    if use_gpu:
        defaults.update({"tree_method": "hist", "device": "cuda"})

    model = xgb.XGBRegressor(**defaults)
    model.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
    return model


def save_xgb(model: xgb.XGBRegressor, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(target)


def load_xgb(path: str | Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(Path(path))
    return model
