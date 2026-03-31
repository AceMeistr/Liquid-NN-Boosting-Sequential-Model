from __future__ import annotations

import logging

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


# ========================  BASIC INDICATORS  ========================

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
    pct_b = (series - lower) / (upper - lower + 1e-12)
    return pd.DataFrame({"bb_mid": mid, "bb_upper": upper, "bb_lower": lower, "bb_width": width, "bb_pct_b": pct_b})


# ========================  MATHEMATICAL FORWARD-LOOKING  ========================

def kalman_smoothing(series: pd.Series, measurement_noise: float = 0.01) -> pd.Series:
    """O(N) 1D Kalman Filter to denoise price while immediately responding to structural shifts.
    Outperforms SMA/EMA significantly by minimizing lag on true variance spikes.
    """
    n = len(series)
    estimates = np.zeros(n)
    errors = np.zeros(n)
    
    estimates[0] = series.iloc[0]
    errors[0] = 1.0
    process_noise = 1e-4

    for i in range(1, n):
        # Predict
        pred_est = estimates[i-1]
        pred_err = errors[i-1] + process_noise
        
        # Update
        kalman_gain = pred_err / (pred_err + measurement_noise)
        estimates[i] = pred_est + kalman_gain * (series.iloc[i] - pred_est)
        errors[i] = (1 - kalman_gain) * pred_err
        
    return pd.Series(estimates, index=series.index)


def shannon_entropy(series: pd.Series, window: int = 20, num_bins: int = 10) -> pd.Series:
    """Rolling Shannon Information Entropy.
    Quantifies the complexity/randomness of the return series. Sharp entropy drops often 
    precede major directional trends (order emerging from chaos), acting as a forward-looking momentum regime indicator.
    """
    def _compute_entropy(x: np.ndarray) -> float:
        hist, _ = np.histogram(x, bins=num_bins, density=True)
        # Filter 0 to avoid log2(0)
        hist = hist[hist > 0.0]
        return -np.sum(hist * np.log2(hist))
        
    return series.rolling(window=window, min_periods=window).apply(_compute_entropy, raw=True)



def stoch_rsi(series: pd.Series, window: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    rsi_series = rsi(series, window=window)
    min_rsi = rsi_series.rolling(window=window, min_periods=window).min()
    max_rsi = rsi_series.rolling(window=window, min_periods=window).max()
    stoch = (rsi_series - min_rsi) / (max_rsi - min_rsi + 1e-12)
    stoch_k = stoch.rolling(window=smooth_k, min_periods=smooth_k).mean()
    stoch_d = stoch_k.rolling(window=smooth_d, min_periods=smooth_d).mean()
    return pd.DataFrame({"stoch_rsi": stoch, "stoch_rsi_k": stoch_k, "stoch_rsi_d": stoch_d})


# ========================  ADVANCED INDICATORS  ========================

def vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Volume Weighted Average Price."""
    typical = (df["high"] + df["low"] + df["price"]) / 3.0
    cum_tp_vol = (typical * df["volume"]).rolling(window, min_periods=window).sum()
    cum_vol = df["volume"].rolling(window, min_periods=window).sum()
    return cum_tp_vol / (cum_vol + 1e-12)


def obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df["price"].diff())
    return (direction * df["volume"]).cumsum()


def cmf(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Chaikin Money Flow."""
    mfm = ((df["price"] - df["low"]) - (df["high"] - df["price"])) / (df["high"] - df["low"] + 1e-12)
    mfv = mfm * df["volume"]
    return mfv.rolling(window, min_periods=window).sum() / (df["volume"].rolling(window, min_periods=window).sum() + 1e-12)


def adx(df: pd.DataFrame, window: int = 14) -> pd.DataFrame:
    """Average Directional Index with +DI / -DI."""
    high = df["high"]
    low = df["low"]
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    plus_dm = np.where((high - prev_high) > (prev_low - low), np.maximum(high - prev_high, 0), 0)
    minus_dm = np.where((prev_low - low) > (high - prev_high), np.maximum(prev_low - low, 0), 0)

    atr_vals = atr(df, window)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window).mean() / (atr_vals + 1e-12)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / window, min_periods=window).mean() / (atr_vals + 1e-12)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)
    adx_val = dx.ewm(alpha=1 / window, min_periods=window).mean()

    return pd.DataFrame({"adx": adx_val, "plus_di": plus_di, "minus_di": minus_di})


def keltner_channels(df: pd.DataFrame, ema_window: int = 20, atr_window: int = 14, multiplier: float = 1.5) -> pd.DataFrame:
    """Keltner Channels."""
    mid = ema(df["price"], ema_window)
    atr_vals = atr(df, atr_window)
    upper = mid + multiplier * atr_vals
    lower = mid - multiplier * atr_vals
    return pd.DataFrame({"kc_upper": upper, "kc_mid": mid, "kc_lower": lower})


def parkinson_volatility(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Parkinson high-low volatility estimator (more efficient than close-close)."""
    log_hl = np.log(df["high"] / (df["low"] + 1e-12))
    return np.sqrt((log_hl ** 2).rolling(window, min_periods=window).mean() / (4 * np.log(2)))


def price_momentum(series: pd.Series, window: int) -> pd.Series:
    """Rate of change as percentage."""
    shifted = series.shift(window)
    return (series - shifted) / (shifted + 1e-12) * 100


def tick_intensity(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """Rolling count proxy for tick intensity (volume acceleration)."""
    return df["volume"].diff().rolling(window, min_periods=window).mean()


# ========================  FEATURE SCHEMA  ========================

def get_feature_schema(include_stoch_rsi: bool = True) -> list[str]:
    base = [
        "price", "high", "low", "volume",
        # Moving averages
        "sma_14", "sma_50", "ema_12", "ema_26",
        # RSI
        "rsi_14",
        # MACD
        "macd", "macd_signal", "macd_hist",
        # Volatility
        "atr_14",
        "bb_mid", "bb_upper", "bb_lower", "bb_width", "bb_pct_b",
        "parkinson_vol",
        # Keltner
        "kc_upper", "kc_mid", "kc_lower",
        # Volume indicators
        "vwap_20", "obv", "cmf_20", "tick_intensity",
        # Trend / Directional
        "adx", "plus_di", "minus_di",
        # Returns & momentum
        "ret_1", "ret_5", "ret_20",
        "momentum_10", "momentum_30",
        "price_sma_ratio",
        # Volume z-score
        "vol_z_20",
        # Advanced Math / Forward-Looking
        "kalman_price", "price_kalman_ratio",
        "entropy_20",
    ]
    if include_stoch_rsi:
        return [*base, "stoch_rsi", "stoch_rsi_k", "stoch_rsi_d"]
    return base


# ========================  FEATURE BUILDER  ========================

def add_all_indicators(df: pd.DataFrame, include_stoch_rsi: bool = True) -> pd.DataFrame:
    out = df.copy()
    start_rows = len(out)

    # --- Moving averages ---
    out["sma_14"] = sma(out["price"], 14)
    out["sma_50"] = sma(out["price"], 50)
    out["ema_12"] = ema(out["price"], 12)
    out["ema_26"] = ema(out["price"], 26)

    # --- RSI ---
    out["rsi_14"] = rsi(out["price"], 14)

    # --- MACD ---
    macd_df = macd(out["price"], 12, 26, 9)
    out = pd.concat([out, macd_df], axis=1)

    # --- Volatility ---
    out["atr_14"] = atr(out, 14)
    bb_df = bollinger_bands(out["price"], 20, 2.0)
    out = pd.concat([out, bb_df], axis=1)
    out["parkinson_vol"] = parkinson_volatility(out, 20)

    # --- Keltner Channels ---
    kc_df = keltner_channels(out)
    out = pd.concat([out, kc_df], axis=1)

    # --- Volume indicators ---
    out["vwap_20"] = vwap(out, 20)
    # Use 20-bar OBV momentum (stationary) instead of cumulative OBV
    # Cumulative OBV encodes calendar time rather than market signal.
    out["obv"] = obv(out).diff(20)
    out["cmf_20"] = cmf(out, 20)
    out["tick_intensity"] = tick_intensity(out, 20)

    # --- Trend / Directional ---
    adx_df = adx(out, 14)
    out = pd.concat([out, adx_df], axis=1)

    # --- Stochastic RSI ---
    if include_stoch_rsi:
        srsi_df = stoch_rsi(out["price"], 14, 3, 3)
        out = pd.concat([out, srsi_df], axis=1)

    # --- Returns & momentum ---
    out["ret_1"] = out["price"].pct_change(1)
    out["ret_5"] = out["price"].pct_change(5)
    out["ret_20"] = out["price"].pct_change(20)
    out["momentum_10"] = price_momentum(out["price"], 10)
    out["momentum_30"] = price_momentum(out["price"], 30)
    out["price_sma_ratio"] = out["price"] / (out["sma_50"] + 1e-12)

    # --- Volume z-score ---
    out["vol_z_20"] = (out["volume"] - out["volume"].rolling(20).mean()) / (out["volume"].rolling(20).std() + 1e-12)

    # --- Advanced Mathematical / Forward Looking ---
    # Replace lagging SMA/EMA position ratios with Kalman Filter ratio
    out["kalman_price"] = kalman_smoothing(out["price"])
    out["price_kalman_ratio"] = out["price"] / (out["kalman_price"] + 1e-12)
    
    # Entropy applied to log returns
    log_returns = np.log(out["price"] / out["price"].shift(1))
    out["entropy_20"] = shannon_entropy(log_returns, window=20)

    # --- Final sanitation ---
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out.dropna().reset_index(drop=True)
    dropped = start_rows - len(out)
    if dropped > 0:
        logger.info("Dropped %s rows after indicator construction due to NaN/Inf", dropped)
    return out
