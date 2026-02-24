from __future__ import annotations

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()


def ema(series: pd.Series, window: int) -> pd.Series:
    return series.ewm(span=window, adjust=False, min_periods=window).mean()


def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    fast_ema = ema(series, fast)
    slow_ema = ema(series, slow)
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False, min_periods=signal).mean()
    hist = macd_line - signal_line
    return pd.DataFrame({"macd": macd_line, "macd_signal": signal_line, "macd_hist": hist})


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"]
    low = df["low"]
    close = df["price"]
    prev_close = close.shift(1)
    tr = pd.concat([(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()


def bollinger_bands(series: pd.Series, window: int = 20, k: float = 2.0) -> pd.DataFrame:
    mid = sma(series, window)
    std = series.rolling(window=window, min_periods=window).std()
    upper = mid + k * std
    lower = mid - k * std
    width = (upper - lower) / (mid + 1e-12)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width})


def stoch_rsi(series: pd.Series, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    rsi_series = rsi(series, window=window)
    min_rsi = rsi_series.rolling(window=window, min_periods=window).min()
    max_rsi = rsi_series.rolling(window=window, min_periods=window).max()
    stoch = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)
    stoch_k = stoch.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({"stoch_rsi": stoch, "stoch_rsi_k": stoch_k, "stoch_rsi_d": stoch_d})


def get_feature_schema(include_stoch_rsi: bool = True) -> list[str]:
    base = [
        "price",
        "high",
        "low",
        "volume",
        "sma_14",
        "sma_50",
        "ema_12",
        "ema_26",
        "rsi_14",
        "macd",
        "macd_signal",
        "macd_hist",
        "atr_14",
        "bb_mid",
        "bb_upper",
        "bb_lower",
        "bb_width",
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_z_20",
    ]
    if include_stoch_rsi:
        return [*base, "stoch_rsi", "stoch_rsi_k", "stoch_rsi_d"]
    return base


def add_all_indicators(df: pd.DataFrame, include_stoch_rsi: bool = True) -> pd.DataFrame:
    out = df.copy()
    start_rows = len(out)
    out["sma_14"] = sma(out["price"], 14)
    out["sma_50"] = sma(out["price"], 50)
    out["ema_12"] = ema(out["price"], 12)
    out["ema_26"] = ema(out["price"], 26)
    out["rsi_14"] = rsi(out["price"], 14)

    macd_df = macd(out["price"], 12, 26, 9)
    for col in macd_df.columns:
        out[col] = macd_df[col]

    out["atr_14"] = atr(out, 14)
    bb_df = bollinger_bands(out["price"], 20, 2.0)
    for col in bb_df.columns:
        out[col] = bb_df[col]

    if include_stoch_rsi:
        srsi_df = stoch_rsi(out["price"], 14, 3, 3)
        for col in srsi_df.columns:
            out[col] = srsi_df[col]

    out["ret_1"] = out["price"].pct_change(1)
    out["ret_5"] = out["price"].pct_change(5)
    out["ret_20"] = out["price"].pct_change(20)
    out["vol_z_20"] = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-12)

    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    dropped = start_rows - len(out)
    if dropped > 0:
        logger.info("Dropped %s rows after indicator construction due to NaN/Inf", dropped)
    return out
