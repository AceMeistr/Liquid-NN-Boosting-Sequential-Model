from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import RobustScaler
import xgboost as xgb

from modelmk1.data.loader import DataBundle, build_supervised_data, load_tick_df, resample_ticks
from modelmk1.eval.backtest import sharpe_ratio


logger = logging.getLogger(__name__)

_ANNUALIZATION_1MIN = 252.0 * 375.0
_BASE_OHLCV_COLUMNS = {"price", "high", "low", "volume"}


@dataclass(frozen=True)
class CorrelationSpectralResult:
    constituents: list[str]
    correlation_matrix: np.ndarray
    eigenvalues: np.ndarray
    window_end_timestamp: pd.Timestamp


@dataclass(frozen=True)
class SpectralDataset:
    features: np.ndarray
    targets: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    constituents: list[str]
    feature_names: list[str]



def _sanitize_quantiles(low: float, high: float) -> tuple[float, float]:
    q_low = float(np.clip(low, 5.0, 40.0))
    q_high = float(np.clip(high, 60.0, 95.0))
    if q_high <= q_low + 10.0:
        q_high = min(95.0, q_low + 10.0)
    return q_low, q_high


def _stable_corr(window_values: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(window_values.T)
    if not np.all(np.isfinite(corr)):
        corr = np.eye(window_values.shape[1], dtype=np.float64)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return corr


def _sorted_eigenvalues(corr: np.ndarray) -> np.ndarray:
    eigenvalues = np.linalg.eigvalsh(corr).astype(np.float32)
    return np.sort(eigenvalues)[::-1]


def _build_lnn_feature_bundle(market_frame: pd.DataFrame, seq_len: int, horizon: int) -> DataBundle:
    return build_supervised_data(
        market_frame,
        seq_len=int(seq_len),
        horizon=int(horizon),
        include_stoch_rsi=True,
    )


def _candidate_indicator_indices(feature_names: list[str]) -> np.ndarray:
    indices = [idx for idx, name in enumerate(feature_names) if name not in _BASE_OHLCV_COLUMNS]
    if len(indices) < 2:
        indices = list(range(len(feature_names)))
    return np.asarray(indices, dtype=np.int64)


def _indicator_scores(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    x = values.astype(np.float64)
    y = targets.astype(np.float64)

    x_centered = x - x.mean(axis=0, keepdims=True)
    y_centered = y - y.mean()

    x_std = x_centered.std(axis=0)
    y_std = float(y_centered.std())

    denom = x_std * (y_std + 1e-12)
    cov = (x_centered * y_centered[:, None]).mean(axis=0)
    corr = cov / (denom + 1e-12)
    corr = np.where(np.isfinite(corr), corr, 0.0)
    return np.abs(corr)


def _select_indicator_subset(
    bundle: DataBundle,
    top_n: int,
    explicit_names: list[str] | None,
) -> tuple[np.ndarray, list[str]]:
    if top_n < 2:
        raise ValueError("top_n must be >= 2")

    all_names = bundle.feature_columns
    if explicit_names:
        cleaned = [name.strip() for name in explicit_names if name.strip()]
        matched = [name for name in cleaned if name in all_names]
        if len(matched) < 2:
            raise ValueError("Explicit indicator list does not contain at least two valid feature names")
        selected_names = matched[: min(top_n, len(matched))]
        selected_indices = np.asarray([all_names.index(name) for name in selected_names], dtype=np.int64)
        return selected_indices, selected_names

    candidate_indices = _candidate_indicator_indices(all_names)
    candidate_values = bundle.xgb_features[:, candidate_indices]
    scores = _indicator_scores(candidate_values, bundle.targets)

    rank = np.argsort(scores)[::-1]
    count = min(int(top_n), len(rank))
    chosen_in_candidate = rank[:count]
    selected_indices = candidate_indices[chosen_in_candidate]
    selected_names = [all_names[int(idx)] for idx in selected_indices.tolist()]

    if len(selected_indices) < 2:
        raise ValueError("Unable to select at least two indicators for spectral correlation features")
    return selected_indices.astype(np.int64), selected_names


def _scaled_selected_indicators(
    bundle: DataBundle,
    top_n: int,
    indicator_names: list[str] | None,
    robust_quantile_low: float,
    robust_quantile_high: float,
) -> tuple[np.ndarray, list[str]]:
    selected_indices, selected_names = _select_indicator_subset(
        bundle,
        top_n=top_n,
        explicit_names=indicator_names,
    )

    q_low, q_high = _sanitize_quantiles(robust_quantile_low, robust_quantile_high)
    scaler = RobustScaler(
        with_centering=True,
        with_scaling=True,
        quantile_range=(q_low, q_high),
        unit_variance=False,
    )

    selected = bundle.xgb_features[:, selected_indices]
    scaled = scaler.fit_transform(selected)
    scaled = np.nan_to_num(scaled, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return scaled, selected_names


def load_constituent_price_panel(data_path: str | Path) -> pd.DataFrame:
    """Load market bars and return a 1-minute frame consumed by LNN feature generation."""
    source = Path(data_path)
    tick_df = load_tick_df(str(source))
    sampled = resample_ticks(tick_df, freq="1min")
    if len(sampled) < 500:
        raise ValueError("Not enough bars to build robust spectral-classification features")
    return sampled


def build_latest_correlation_spectral(
    price_panel: pd.DataFrame,
    top_n: int = 10,
    corr_window: int = 60,
    tickers: list[str] | None = None,
    seq_len: int = 64,
    horizon: int = 1,
    robust_quantile_low: float = 15.0,
    robust_quantile_high: float = 85.0,
) -> CorrelationSpectralResult:
    """Build latest indicator-correlation matrix and eigenvalues from LNN-generated features."""
    if corr_window < 5:
        raise ValueError("corr_window must be >= 5")

    bundle = _build_lnn_feature_bundle(price_panel, seq_len=seq_len, horizon=horizon)
    scaled, selected_names = _scaled_selected_indicators(
        bundle,
        top_n=top_n,
        indicator_names=tickers,
        robust_quantile_low=robust_quantile_low,
        robust_quantile_high=robust_quantile_high,
    )

    if len(scaled) < corr_window:
        raise ValueError("Not enough samples for requested corr_window")

    window = scaled[-corr_window:]
    corr = _stable_corr(window)
    eig = _sorted_eigenvalues(corr)

    if "timestamp" in price_panel.columns and len(price_panel) > 0:
        window_end = pd.Timestamp(price_panel["timestamp"].iloc[-1])
    else:
        window_end = pd.Timestamp.utcnow()

    return CorrelationSpectralResult(
        constituents=selected_names,
        correlation_matrix=corr.astype(np.float32),
        eigenvalues=eig,
        window_end_timestamp=window_end,
    )


def build_spectral_training_matrices(
    price_panel: pd.DataFrame,
    top_n: int = 10,
    corr_window: int = 60,
    horizon: int = 1,
    tickers: list[str] | None = None,
    seq_len: int = 64,
    robust_quantile_low: float = 15.0,
    robust_quantile_high: float = 85.0,
) -> SpectralDataset:
    """Build classification-ready spectral features from LNN feature vectors."""
    if corr_window < 5:
        raise ValueError("corr_window must be >= 5")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    bundle = _build_lnn_feature_bundle(price_panel, seq_len=seq_len, horizon=horizon)
    scaled, selected_names = _scaled_selected_indicators(
        bundle,
        top_n=top_n,
        indicator_names=tickers,
        robust_quantile_low=robust_quantile_low,
        robust_quantile_high=robust_quantile_high,
    )

    if len(scaled) <= corr_window:
        raise ValueError("Insufficient feature rows for requested correlation window")

    top_k = int(scaled.shape[1])
    start_idx = corr_window - 1
    n_rows = len(scaled) - start_idx

    features = np.empty((n_rows, top_k * 2), dtype=np.float32)
    targets = np.empty(n_rows, dtype=np.float32)
    labels = np.empty(n_rows, dtype=np.float32)
    timestamps = np.empty(n_rows, dtype=np.int64)

    for row_idx, end_idx in enumerate(range(start_idx, len(scaled))):
        window = scaled[end_idx - corr_window + 1 : end_idx + 1]
        corr = _stable_corr(window)
        eig = _sorted_eigenvalues(corr)

        features[row_idx, :top_k] = eig
        features[row_idx, top_k:] = scaled[end_idx]

        realized_return = float(bundle.targets[end_idx])
        targets[row_idx] = realized_return
        labels[row_idx] = 1.0 if realized_return > 0.0 else 0.0
        timestamps[row_idx] = int(end_idx)

    feature_names = [f"corr_eigenvalue_{i + 1:02d}" for i in range(top_k)]
    feature_names.extend([f"indicator_{name}" for name in selected_names])

    return SpectralDataset(
        features=features,
        targets=targets,
        labels=labels,
        timestamps=timestamps,
        constituents=selected_names,
        feature_names=feature_names,
    )


def train_xgb_with_dmatrix(
    dataset: SpectralDataset,
    train_ratio: float = 0.8,
    num_boost_round: int = 700,
    early_stopping_rounds: int = 50,
    params: dict[str, Any] | None = None,
) -> tuple[xgb.Booster, dict[str, Any]]:
    """Train binary XGBoost to optimize directional quality and Sharpe-like behavior."""
    if not (0.0 < train_ratio < 1.0):
        raise ValueError(f"train_ratio must be in (0,1), got {train_ratio}")
    if len(dataset.targets) < 50:
        raise ValueError("Need at least 50 samples for stable classification split")

    split = int(len(dataset.targets) * train_ratio)
    split = max(25, min(split, len(dataset.targets) - 10))

    x_train, x_val = dataset.features[:split], dataset.features[split:]
    y_train_cls, y_val_cls = dataset.labels[:split], dataset.labels[split:]
    y_val_returns = dataset.targets[split:]

    dtrain = xgb.DMatrix(x_train, label=y_train_cls, feature_names=dataset.feature_names)
    dval = xgb.DMatrix(x_val, label=y_val_cls, feature_names=dataset.feature_names)

    pos = float(np.sum(y_train_cls == 1.0))
    neg = float(np.sum(y_train_cls == 0.0))
    scale_pos_weight = neg / max(pos, 1.0)

    defaults: dict[str, Any] = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "eta": 0.03,
        "max_depth": 5,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "min_child_weight": 2.0,
        "lambda": 1.0,
        "alpha": 1e-3,
        "gamma": 0.05,
        "max_bin": 512,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "gpu_hist",
        "device": "cuda",
        "seed": 42,
    }
    if params:
        defaults.update(params)

    decision_threshold = float(defaults.pop("decision_threshold", 0.5))
    decision_threshold = float(np.clip(decision_threshold, 0.35, 0.65))

    evals_result: dict[str, dict[str, list[float]]] = {}
    try:
        booster = xgb.train(
            params=defaults,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )
    except xgb.core.XGBoostError as exc:
        message = str(exc).lower()
        gpu_hist_unsupported = "gpu_hist" in message and "invalid input" in message
        if not gpu_hist_unsupported:
            raise

        fallback = dict(defaults)
        fallback["tree_method"] = "hist"
        fallback["device"] = "cuda"
        evals_result = {}
        logger.warning("gpu_hist unsupported by current XGBoost, falling back to hist+cuda")
        booster = xgb.train(
            params=fallback,
            dtrain=dtrain,
            num_boost_round=num_boost_round,
            evals=[(dtrain, "train"), (dval, "val")],
            early_stopping_rounds=early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=False,
        )

    prob = booster.predict(dval)
    pred_label = (prob >= decision_threshold).astype(np.float32)

    mse = float(mean_squared_error(y_val_cls, prob))
    mae = float(mean_absolute_error(y_val_cls, prob))
    rmse = float(np.sqrt(mse))

    directional_accuracy = float((pred_label == y_val_cls).mean())
    strategy_direction = np.where(pred_label >= 0.5, 1.0, -1.0)
    strategy_returns = strategy_direction * y_val_returns

    sharpe_raw = sharpe_ratio(strategy_returns, annualization=1.0)
    sharpe_annualized = sharpe_ratio(strategy_returns, annualization=_ANNUALIZATION_1MIN)

    importance_raw = booster.get_score(importance_type="gain")
    importance: dict[str, float] = {}
    for fname in dataset.feature_names:
        raw_score = importance_raw.get(fname, 0.0)
        if isinstance(raw_score, list):
            importance[fname] = float(np.mean(raw_score)) if raw_score else 0.0
        else:
            importance[fname] = float(raw_score)

    metrics: dict[str, Any] = {
        "target_type": "classification",
        "train_samples": int(len(y_train_cls)),
        "val_samples": int(len(y_val_cls)),
        "val_mse": mse,
        "val_mae": mae,
        "val_rmse": rmse,
        "val_directional_accuracy": directional_accuracy,
        "val_sharpe_ratio": sharpe_annualized,
        "val_sharpe_ratio_raw": sharpe_raw,
        "val_strategy_mean_return": float(np.mean(strategy_returns)),
        "val_strategy_std_return": float(np.std(strategy_returns, ddof=1)) if len(strategy_returns) > 1 else 0.0,
        "val_positive_rate": float(np.mean(y_val_cls)),
        "val_pred_positive_rate": float(np.mean(pred_label)),
        "decision_threshold": decision_threshold,
        "best_iteration": int(booster.best_iteration) if booster.best_iteration is not None else None,
        "feature_importance_gain": importance,
        "eval_history": evals_result,
    }

    logger.info(
        "XGBoost classification complete | sharpe=%.3f | directional_acc=%.3f | mse=%.6f | mae=%.6f",
        sharpe_annualized,
        directional_accuracy,
        mse,
        mae,
    )
    return booster, metrics
