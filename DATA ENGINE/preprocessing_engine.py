from __future__ import annotations

import gc
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from loguru import logger

from config import PipelineConfig
from logging_utils import configure_logging


class PreprocessingEngine:
    """Clean and normalize daily raw ingestion output."""

    def __init__(self, config: PipelineConfig):
        """Initialize preprocessing engine.

        Args:
            config: Pipeline configuration instance.
        """
        self.cfg = config
        configure_logging("preprocessing_engine", self.cfg.LOG_ROOT)

    @staticmethod
    def _raw_path(raw_root: Path, trading_date: date) -> Path:
        return raw_root / f"{trading_date.year}" / f"{trading_date.month:02d}" / f"master_{trading_date.isoformat()}.parquet"

    @staticmethod
    def _clean_path(clean_root: Path, trading_date: date) -> Path:
        return clean_root / f"{trading_date.year}" / f"{trading_date.month:02d}" / f"clean_{trading_date.isoformat()}.parquet"

    @staticmethod
    def _symbols_from_columns(columns: pd.Index) -> list[str]:
        symbols = sorted({c[:-5] for c in columns if c.endswith("_open")})
        return symbols

    def apply_corporate_adjustments(
        self,
        df: pd.DataFrame,
        symbol: str,
        ca_table: pd.DataFrame,
    ) -> pd.DataFrame:
        """Apply split and dividend adjustments for historical bars.

        Args:
            df: Input daily frame.
            symbol: Symbol prefix used in frame columns.
            ca_table: Corporate actions table.

        Returns:
            Adjusted dataframe.
        """
        if ca_table.empty:
            return df

        open_col = f"{symbol}_open"
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        close_col = f"{symbol}_close"
        vol_col = f"{symbol}_volume"

        if close_col not in df.columns:
            return df

        max_ts = pd.to_datetime(df.index.max()).tz_localize(None)
        events = ca_table.loc[
            (ca_table["symbol"].astype(str).str.lower() == symbol.lower())
            & (ca_table["ex_date"] <= max_ts)
        ].sort_values("ex_date")

        if events.empty:
            return df

        for _, evt in events.iterrows():
            ex_dt = pd.Timestamp(evt["ex_date"]).tz_localize("Asia/Kolkata")
            mask = df.index < ex_dt

            split_ratio = float(evt.get("split_ratio", np.nan)) if pd.notna(evt.get("split_ratio")) else np.nan
            dividend = float(evt.get("dividend_amount", np.nan)) if pd.notna(evt.get("dividend_amount")) else np.nan

            if pd.notna(split_ratio) and split_ratio > 0 and split_ratio != 1.0:
                px_factor = np.float32(1.0 / split_ratio)
                vol_factor = np.float32(split_ratio)
                for col in (open_col, high_col, low_col, close_col):
                    if col in df.columns:
                        df.loc[mask, col] = (df.loc[mask, col].astype("float32") * px_factor).astype("float32")
                if vol_col in df.columns:
                    df.loc[mask, vol_col] = (df.loc[mask, vol_col].astype("float32") * vol_factor).astype("int32")
                logger.info(
                    "Applied split adjustment symbol={} ex_date={} split_ratio={} factor={}",
                    symbol,
                    ex_dt.date(),
                    split_ratio,
                    px_factor,
                )

            if pd.notna(dividend) and dividend > 0:
                prior_close = df.loc[mask, close_col].dropna()
                if not prior_close.empty and float(prior_close.iloc[-1]) > 0:
                    close_pre = float(prior_close.iloc[-1])
                    div_factor = np.float32((close_pre - dividend) / close_pre)
                    for col in (open_col, high_col, low_col, close_col):
                        if col in df.columns:
                            df.loc[mask, col] = (df.loc[mask, col].astype("float32") * div_factor).astype("float32")
                    logger.info(
                        "Applied dividend adjustment symbol={} ex_date={} dividend={} factor={}",
                        symbol,
                        ex_dt.date(),
                        dividend,
                        div_factor,
                    )

        return df

    def validate_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Validate OHLCV integrity and quarantine invalid bars.

        Args:
            df: Input frame.
            symbol: Symbol prefix.

        Returns:
            Updated frame with invalid bars flagged and OHLC set to NaN.
        """
        open_col = f"{symbol}_open"
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        close_col = f"{symbol}_close"
        vol_col = f"{symbol}_volume"
        invalid_col = f"{symbol}_is_invalid_bar"

        required = [open_col, high_col, low_col, close_col, vol_col]
        if not all(col in df.columns for col in required):
            return df

        o = df[open_col].astype("float32")
        h = df[high_col].astype("float32")
        l = df[low_col].astype("float32")
        c = df[close_col].astype("float32")
        v = df[vol_col].astype("float32")

        cond_h = h < np.maximum(o, c)
        cond_l = l > np.minimum(o, c)
        cond_px = (o <= 0) | (h <= 0) | (l <= 0) | (c <= 0)
        cond_v = v < 0

        ratio = (c + self.cfg.EPSILON) / (c.shift(1) + self.cfg.EPSILON)
        log_ret = pd.Series(np.log(ratio.to_numpy(dtype=np.float64)), index=ratio.index)
        cond_jump = log_ret.abs().fillna(0.0) >= 0.20

        invalid = cond_h | cond_l | cond_px | cond_v | cond_jump
        df[invalid_col] = invalid.astype("bool")

        for col in (open_col, high_col, low_col, close_col):
            df.loc[invalid, col] = np.nan

        if invalid.any():
            logger.warning(
                "Quarantined {} bars for symbol={} on {}",
                int(invalid.sum()),
                symbol,
                pd.Timestamp(df.index.min()).date(),
            )

        return df

    def detect_stale_feeds(self, df: pd.DataFrame, symbol: str, window: int = 5) -> pd.Series:
        """Detect stale quote sequences by identical OHLC runs.

        Args:
            df: Input frame.
            symbol: Symbol prefix.
            window: Minimum consecutive bars to consider stale.

        Returns:
            Boolean stale indicator series.
        """
        open_col = f"{symbol}_open"
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        close_col = f"{symbol}_close"

        required = [open_col, high_col, low_col, close_col]
        if not all(col in df.columns for col in required):
            return pd.Series(False, index=df.index)

        same_ohlc = (
            df[open_col].eq(df[open_col].shift(1))
            & df[high_col].eq(df[high_col].shift(1))
            & df[low_col].eq(df[low_col].shift(1))
            & df[close_col].eq(df[close_col].shift(1))
        )

        groups = (~same_ohlc).cumsum()
        run_len = same_ohlc.groupby(groups).cumcount() + 1
        stale = same_ohlc & run_len.ge(window)
        logger.info("Detected stale bars symbol={} count={}", symbol, int(stale.sum()))
        return stale.astype("bool")

    def impute_gaps(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Impute intraday missing values without crossing session boundaries.

        Args:
            df: Input frame.
            symbol: Symbol prefix.

        Returns:
            Gap-imputed dataframe.
        """
        open_col = f"{symbol}_open"
        high_col = f"{symbol}_high"
        low_col = f"{symbol}_low"
        close_col = f"{symbol}_close"
        vol_col = f"{symbol}_volume"

        if close_col not in df.columns or "session_date" not in df.columns:
            return df

        for _, idx in df.groupby("session_date", sort=True).groups.items():
            session_idx = pd.Index(idx)
            close = df.loc[session_idx, close_col].astype("float32")
            close = close.interpolate(method="linear", limit_direction="both")
            df.loc[session_idx, close_col] = close.astype("float32")
            for col in (open_col, high_col, low_col):
                if col in df.columns:
                    df.loc[session_idx, col] = df.loc[session_idx, col].fillna(close).astype("float32")
            if vol_col in df.columns:
                df.loc[session_idx, vol_col] = df.loc[session_idx, vol_col].fillna(0).astype("int32")

        return df

    def compute_log_returns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute session-aware log returns and overnight return for index.

        Args:
            df: Input frame.
            symbol: Symbol prefix.

        Returns:
            Frame with return columns appended.
        """
        close_col = f"{symbol}_close"
        ret_col = f"{symbol}_log_ret"

        if close_col not in df.columns or "session_date" not in df.columns:
            return df

        ret = df.groupby("session_date", sort=True)[close_col].transform(
            lambda s: np.log((s + self.cfg.EPSILON) / (s.shift(1) + self.cfg.EPSILON))
        )
        if "bar_index_in_session" in df.columns:
            ret[df["bar_index_in_session"].eq(0)] = np.nan
        df[ret_col] = ret.astype("float32")

        if symbol.lower() == "idx":
            overnight = pd.Series(np.nan, index=df.index, dtype="float32")
            session_dates = pd.to_datetime(df["session_date"], errors="coerce").dt.date
            dates = sorted([d for d in session_dates.dropna().unique()])
            prev_last_close: float | None = None
            for d in dates:
                mask = session_dates.eq(d)
                session_close = df.loc[mask, close_col].astype("float32")
                if session_close.empty:
                    continue
                first_close = float(session_close.iloc[0])
                if prev_last_close is not None and prev_last_close > 0 and first_close > 0:
                    value = float(np.log((first_close + self.cfg.EPSILON) / (prev_last_close + self.cfg.EPSILON)))
                    overnight.loc[mask] = value
                prev_last_close = float(session_close.iloc[-1])
            df["idx_overnight_ret"] = overnight

        return df

    def normalize_expiry_regime(self, df: pd.DataFrame) -> pd.DataFrame:
        """Attach expiry-regime features based on regime table and BSE calendar.

        Args:
            df: Input frame.

        Returns:
            Frame with expiry regime features appended.
        """
        if "session_date" not in df.columns:
            return df

        regime = pd.read_csv(self.cfg.REGIME_TABLE_PATH, parse_dates=["date_from", "date_to"])
        cal = mcal.get_calendar("BSE")

        days_to_expiry = pd.Series(np.nan, index=df.index, dtype="float32")
        dte_normalized = pd.Series(np.nan, index=df.index, dtype="float32")
        is_expiry_day = pd.Series(False, index=df.index, dtype="bool")
        regime_flag = pd.Series(0, index=df.index, dtype="int8")

        session_dates = sorted([d for d in pd.to_datetime(df["session_date"], errors="coerce").dt.date.dropna().unique()])

        for sess_date in session_dates:
            sess_ts = pd.Timestamp(sess_date)
            row = regime.loc[(regime["date_from"] <= sess_ts) & (regime["date_to"] >= sess_ts)]

            expiry_weekday = int(row.iloc[0]["expiry_weekday"]) if not row.empty else 3
            flag = int(row.iloc[0]["regime_flag"]) if not row.empty and "regime_flag" in row.columns else 0

            schedule = cal.schedule(start_date=sess_date, end_date=sess_date + timedelta(days=20))
            sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
            candidates = [d.date() for d in sessions if d.weekday() == expiry_weekday and d.date() >= sess_date]
            if not candidates:
                continue
            expiry_date = candidates[0]

            mask = pd.to_datetime(df["session_date"], errors="coerce").dt.date.eq(sess_date)
            dte = float((expiry_date - sess_date).days)

            days_to_expiry.loc[mask] = dte
            dte_normalized.loc[mask] = dte / 7.0
            is_expiry_day.loc[mask] = bool(sess_date == expiry_date)
            regime_flag.loc[mask] = int(flag)

        df["days_to_expiry"] = days_to_expiry
        df["dte_normalized"] = dte_normalized
        df["is_expiry_day"] = is_expiry_day
        df["regime_flag"] = regime_flag
        return df

    def _compute_idx_volume_zscore(self, df: pd.DataFrame) -> pd.DataFrame:
        if "idx_volume" not in df.columns or "session_date" not in df.columns:
            return df

        grp = df.groupby("session_date", sort=True)["idx_volume"]
        mu = grp.transform(lambda s: s.rolling(self.cfg.ROLLING_WINDOW_Z, min_periods=5).mean())
        sigma = grp.transform(lambda s: s.rolling(self.cfg.ROLLING_WINDOW_Z, min_periods=5).std())
        z = (df["idx_volume"] - mu) / sigma.clip(lower=self.cfg.EPSILON)
        df["idx_vol_zscore"] = z.clip(-self.cfg.ZSCORE_CLIP, self.cfg.ZSCORE_CLIP).astype("float32")
        return df

    def run_day(self, trading_date: date) -> None:
        """Run preprocessing pipeline for one day.

        Args:
            trading_date: Trading date to process.
        """
        start_ts = perf_counter()
        raw_path = self._raw_path(self.cfg.raw_root, trading_date)
        if not raw_path.exists():
            raise FileNotFoundError(f"Raw parquet missing for {trading_date}: {raw_path}")

        df = pd.read_parquet(raw_path)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
            df = df.set_index("timestamp")

        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, errors="coerce", utc=True)

        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index).tz_localize("Asia/Kolkata")
        else:
            df.index = pd.DatetimeIndex(df.index).tz_convert("Asia/Kolkata")

        if "session_date" not in df.columns:
            df["session_date"] = pd.DatetimeIndex(pd.DatetimeIndex(df.index).date)
        if "bar_index_in_session" not in df.columns:
            df["bar_index_in_session"] = np.arange(len(df), dtype="int16")

        ca_table = pd.read_csv(self.cfg.CA_TABLE_PATH)
        if not ca_table.empty and "ex_date" in ca_table.columns:
            ca_table["ex_date"] = pd.to_datetime(ca_table["ex_date"], errors="coerce")

        symbols = self._symbols_from_columns(df.columns)
        for symbol in symbols:
            df = self.apply_corporate_adjustments(df, symbol, ca_table)
            df = self.validate_ohlcv(df, symbol)
            stale_col = f"{symbol}_is_stale"
            stale = self.detect_stale_feeds(df, symbol, window=5)
            if stale_col in df.columns:
                df[stale_col] = (df[stale_col].fillna(False) | stale).astype("bool")
            else:
                df[stale_col] = stale.astype("bool")
            df = self.impute_gaps(df, symbol)
            df = self.compute_log_returns(df, symbol)

            for col in (f"{symbol}_open", f"{symbol}_high", f"{symbol}_low", f"{symbol}_close"):
                if col in df.columns:
                    df[col] = df[col].astype("float32")
            vol_col = f"{symbol}_volume"
            if vol_col in df.columns:
                df[vol_col] = pd.to_numeric(df[vol_col], errors="coerce").fillna(0).astype("int32")

        df = self.normalize_expiry_regime(df)
        df = self._compute_idx_volume_zscore(df)

        out_path = self._clean_path(self.cfg.clean_root, trading_date)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.reset_index().to_parquet(out_path, index=False, compression="snappy")

        logger.info(
            "Preprocessing complete for {} rows={} cols={} elapsed={:.2f}s",
            trading_date,
            len(df),
            len(df.columns),
            perf_counter() - start_ts,
        )

        del df
        gc.collect()
