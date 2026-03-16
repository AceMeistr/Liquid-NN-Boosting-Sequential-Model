from __future__ import annotations

import gc
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from loguru import logger
from scipy.interpolate import PchipInterpolator
from scipy.stats import norm

from config import PipelineConfig
from logging_utils import configure_logging


class FeatureEngine:
    """Feature engineering module for the Sensex MFT pipeline."""

    def __init__(self, config: PipelineConfig):
        """Initialize feature engine.

        Args:
            config: Pipeline configuration instance.
        """
        self.cfg = config
        configure_logging("feature_engine", self.cfg.LOG_ROOT)
        self.weights = pd.read_csv(self.cfg.WEIGHTS_PATH)

    def compute_synthetic_iv_surface(
        self,
        bar: pd.Series,
        strikes_df: pd.DataFrame,
        rfr: float,
        dte_years: float,
    ) -> dict[str, float]:
        """Compute synthetic ATM IV and skew metrics from a bar-level option slice.

        Args:
            bar: Current bar with index spot close.
            strikes_df: Option strikes dataframe.
            rfr: Risk-free rate.
            dte_years: Time-to-expiry in years.

        Returns:
            Dict containing synth_atm_iv, iv_skew_25d, iv_butterfly.
        """
        nan_result = {
            "synth_atm_iv": np.nan,
            "iv_skew_25d": np.nan,
            "iv_butterfly": np.nan,
        }

        if strikes_df.empty or "strike" not in strikes_df.columns and strikes_df.index.name != "strike":
            return nan_result

        S = float(bar.get("idx_close", np.nan))
        if not np.isfinite(S) or S <= 0:
            return nan_result

        T = max(float(dte_years), 5.0 / (375.0 * 365.0))
        if strikes_df.index.name == "strike":
            work = strikes_df.reset_index().copy()
        else:
            work = strikes_df.copy()

        if "iv" not in work.columns or "strike" not in work.columns:
            return nan_result

        work["strike"] = pd.to_numeric(work["strike"], errors="coerce")
        work["iv"] = pd.to_numeric(work["iv"], errors="coerce")
        work = work.dropna(subset=["strike", "iv"])
        work = work[(work["iv"] >= 0.01) & (work["iv"] <= 2.0)]
        if work.empty:
            return nan_result

        F = S * np.exp(rfr * T)
        work["m"] = np.log(work["strike"] / F)
        work = work.sort_values("m")

        iv_jump = work["iv"].diff().abs().fillna(0)
        work = work[iv_jump <= 0.15]
        if len(work) < self.cfg.IV_MIN_VALID_STRIKES:
            return nan_result

        try:
            interpolator = PchipInterpolator(work["m"].to_numpy(), work["iv"].to_numpy())
            synth_atm_iv = float(interpolator(0.0))

            sigma = max(synth_atm_iv, 0.01)
            d1_call = norm.ppf(0.25)
            d1_put = norm.ppf(0.75)

            m_25call = 0.5 * sigma * sigma * T - d1_call * sigma * np.sqrt(T)
            m_25put = 0.5 * sigma * sigma * T - d1_put * sigma * np.sqrt(T)

            iv_call = float(interpolator(m_25call))
            iv_put = float(interpolator(m_25put))
            iv_skew = iv_put - iv_call
            iv_bfly = ((iv_put + iv_call) / 2.0) - synth_atm_iv

            return {
                "synth_atm_iv": float(np.float32(synth_atm_iv)),
                "iv_skew_25d": float(np.float32(iv_skew)),
                "iv_butterfly": float(np.float32(iv_bfly)),
            }
        except Exception:
            return nan_result

    def compute_greeks_net(self, strikes_df: pd.DataFrame, S: float, r: float, T: float) -> dict[str, float]:
        """Aggregate net Greek exposures and gamma exposure from selected strikes.

        Args:
            strikes_df: Option rows with delta/gamma/vega/oi and option_type.
            S: Spot price.
            r: Risk-free rate.
            T: Time-to-expiry in years.

        Returns:
            Dict containing net delta/gamma/vega and gex.
        """
        _ = (S, r)
        T = max(float(T), self.cfg.GAMMA_MIN_T_SECONDS / (375.0 * 365.0))
        _ = T

        if strikes_df.empty:
            return {
                "opt_delta_net": np.nan,
                "opt_gamma_net": np.nan,
                "opt_vega_net": np.nan,
                "gex": np.nan,
            }

        work = strikes_df.copy()
        required = ["delta", "gamma", "vega", "oi", "option_type"]
        for col in required:
            if col not in work.columns:
                return {
                    "opt_delta_net": np.nan,
                    "opt_gamma_net": np.nan,
                    "opt_vega_net": np.nan,
                    "gex": np.nan,
                }

        sign = np.where(work["option_type"].astype(str).str.upper().eq("C"), 1.0, -1.0)
        delta = pd.to_numeric(work["delta"], errors="coerce").astype("float32")
        gamma = pd.to_numeric(work["gamma"], errors="coerce").astype("float32")
        vega = pd.to_numeric(work["vega"], errors="coerce").astype("float32")
        oi = pd.to_numeric(work["oi"], errors="coerce").fillna(0).astype("float32")

        opt_delta_net = float(np.nansum(delta * sign))
        opt_gamma_net = float(np.nansum(gamma))
        opt_vega_net = float(np.nansum(vega))
        gex = float(np.nansum(gamma * oi) * self.cfg.SENSEX_CONTRACT_SIZE)

        return {
            "opt_delta_net": float(np.float32(opt_delta_net)),
            "opt_gamma_net": float(np.float32(opt_gamma_net)),
            "opt_vega_net": float(np.float32(opt_vega_net)),
            "gex": float(np.float32(gex)),
        }

    def compute_vwap_deviation(self, session_df: pd.DataFrame) -> pd.Series:
        """Compute ATR-normalized VWAP deviation for one session.

        Args:
            session_df: Dataframe containing idx_high/idx_low/idx_close/idx_volume.

        Returns:
            Session-length VWAP deviation series.
        """
        required = ["idx_high", "idx_low", "idx_close", "idx_volume"]
        if not all(c in session_df.columns for c in required):
            return pd.Series(np.nan, index=session_df.index, dtype="float32")

        high = session_df["idx_high"].astype("float32")
        low = session_df["idx_low"].astype("float32")
        close = session_df["idx_close"].astype("float32")
        volume = session_df["idx_volume"].astype("float32")

        tp = (high + low + close) / 3.0
        cum_tp = (tp * volume).cumsum()
        cum_vol = volume.cumsum().clip(lower=1.0)
        vwap = cum_tp / cum_vol

        tr = pd.concat(
            [
                (high - low).abs(),
                (high - close.shift(1)).abs(),
                (low - close.shift(1)).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1.0 / self.cfg.ROLLING_WINDOW_ATR, adjust=False).mean()
        dev = (close - vwap) / atr.clip(lower=self.cfg.EPSILON)
        return dev.astype("float32")

    def compute_constituent_features(self, day_df: pd.DataFrame) -> pd.DataFrame:
        """Compute cross-sectional constituent features for one day.

        Args:
            day_df: Day-level frame with constituent log returns and stale flags.

        Returns:
            Feature dataframe indexed by timestamps.
        """
        out = pd.DataFrame(index=day_df.index)
        if "idx_log_ret" not in day_df.columns:
            return out

        returns: dict[str, pd.Series] = {}
        w_map: dict[str, float] = {}
        sector_map: dict[str, str] = {}

        for _, row in self.weights.iterrows():
            sym = str(row["symbol"]).lower()
            col = f"{sym}_log_ret"
            if col in day_df.columns:
                returns[sym] = day_df[col].astype("float32").fillna(0.0)
                w_map[sym] = float(row.get("weight", 0.0))
                sector_map[sym] = str(row.get("sector", "other")).lower()

        if not returns:
            return out

        ret_df = pd.DataFrame(returns, index=day_df.index)
        weights = pd.Series(w_map)
        if weights.sum() != 0:
            weights = weights / weights.sum()

        weighted = ret_df.mul(weights, axis=1)
        wrs = weighted.sum(axis=1)
        idx_ret = day_df["idx_log_ret"].astype("float32").fillna(0.0)

        out["wrs_residual"] = (wrs - idx_ret).astype("float32")

        centered = ret_df.sub(wrs, axis=0)
        cssd = np.sqrt((centered.pow(2).mul(weights, axis=1)).sum(axis=1))
        out["cssd"] = cssd.astype("float32")

        top5 = weights.sort_values(ascending=False).head(5).index.tolist()
        top5_weighted = weighted[top5].sum(axis=1)
        concentration = top5_weighted / wrs.replace(0, np.nan)
        out["concentration_ratio"] = concentration.clip(-5, 5).astype("float32")

        for sector_name, out_col in (
            ("bank", "sector_bank_flow"),
            ("it", "sector_it_flow"),
            ("energy", "sector_energy_flow"),
            ("industrial", "sector_industrial_flow"),
        ):
            members = [sym for sym, sector in sector_map.items() if sector_name in sector]
            if members:
                out[out_col] = weighted[members].sum(axis=1).astype("float32")
            else:
                out[out_col] = np.float32(0.0)

        adv = (ret_df > 0).sum(axis=1).astype("float32")
        dec = (ret_df < 0).sum(axis=1).astype("float32")
        total = np.float32(max(1, ret_df.shape[1]))
        adv_ratio = adv / total

        out["const_adv_dec"] = (adv - dec).astype("float32")
        out["adr"] = adv_ratio.ewm(span=5, adjust=False).mean().astype("float32")

        if top5:
            out["top5_dispersion"] = ret_df[top5].std(axis=1).astype("float32")
        else:
            out["top5_dispersion"] = np.float32(0.0)

        stale_cols = [c for c in day_df.columns if c.endswith("_is_stale") and c != "idx_is_stale"]
        if stale_cols:
            out["stale_feed_count"] = day_df[stale_cols].fillna(False).astype(bool).sum(axis=1).astype("int8")
        else:
            out["stale_feed_count"] = np.int8(0)

        return out

    def compute_oi_features(self, option_chain_df: pd.DataFrame) -> pd.DataFrame:
        """Compute OI imbalance and flow features.

        Args:
            option_chain_df: Option-related per-bar dataframe.

        Returns:
            Dataframe with OI features.
        """
        out = pd.DataFrame(index=option_chain_df.index)

        if {"call_oi", "put_oi"}.issubset(option_chain_df.columns):
            call_oi = option_chain_df["call_oi"].astype("float32")
            put_oi = option_chain_df["put_oi"].astype("float32")
            oi_imb = (call_oi - put_oi) / (call_oi + put_oi + 1.0)

            d_call = call_oi.diff().fillna(0.0)
            d_put = put_oi.diff().fillna(0.0)
            oi_flow = (d_call - d_put) / (d_call.abs() + d_put.abs() + 1.0)

            out["oi_imbalance"] = oi_imb.ewm(span=5, adjust=False).mean().astype("float32")
            out["oi_flow"] = oi_flow.ewm(span=5, adjust=False).mean().astype("float32")
            return out

        # If pre-aggregated columns already exist (from ingestion), smooth and return.
        if "oi_imbalance" in option_chain_df.columns:
            out["oi_imbalance"] = (
                option_chain_df["oi_imbalance"].astype("float32").ewm(span=5, adjust=False).mean().astype("float32")
            )
        if "oi_flow" in option_chain_df.columns:
            out["oi_flow"] = option_chain_df["oi_flow"].astype("float32").ewm(span=5, adjust=False).mean().astype("float32")
        if "total_pcr_vol" in option_chain_df.columns:
            out["total_pcr_vol"] = option_chain_df["total_pcr_vol"].astype("float32")
        if "total_pcr_oi" in option_chain_df.columns:
            out["total_pcr_oi"] = option_chain_df["total_pcr_oi"].astype("float32")

        return out

    def compute_momentum_features(self, session_ret_series: pd.Series) -> pd.DataFrame:
        """Compute session-grouped momentum and momentum-decay features.

        Args:
            session_ret_series: Session return series.

        Returns:
            Dataframe with M_5/M_15/M_30/MDS signals.
        """
        ret = session_ret_series.astype("float32").fillna(0.0)
        m5 = ret.rolling(5, min_periods=1).sum()
        m15 = ret.rolling(15, min_periods=1).sum()
        m30 = ret.rolling(30, min_periods=1).sum()
        mds = (m5 - m15) / (m15.abs() + self.cfg.EPSILON)

        return pd.DataFrame(
            {
                "M_5": m5.astype("float32"),
                "M_15": m15.astype("float32"),
                "M_30": m30.astype("float32"),
                "MDS": mds.astype("float32"),
                "idx_mom_decay": mds.astype("float32"),
            },
            index=session_ret_series.index,
        )

    def compute_target(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute binary next-bar direction target as the last transformation.

        Args:
            df: Feature dataframe.

        Returns:
            Dataframe with idx_target_1m appended last.
        """
        if "idx_log_ret" not in df.columns:
            raise ValueError("idx_log_ret missing; cannot compute target.")

        target = (df["idx_log_ret"].shift(-1) > 0).astype("float32")
        if "session_date" in df.columns:
            last_idx = df.groupby("session_date", sort=True).tail(1).index
            target.loc[last_idx] = np.nan

            unexpected_nan = target.isna() & ~target.index.isin(last_idx)
            if unexpected_nan.any():
                raise AssertionError("idx_target_1m contains NaN outside the last bar of session.")

        valid = target.dropna()
        if not valid.isin([0.0, 1.0]).all():
            raise AssertionError("idx_target_1m contains values outside {0,1}.")

        df["idx_target_1m"] = target
        return df

    @staticmethod
    def _clean_path(clean_root: Path, trading_date: date) -> Path:
        return clean_root / f"{trading_date.year}" / f"{trading_date.month:02d}" / f"clean_{trading_date.isoformat()}.parquet"

    @staticmethod
    def _feature_path(feature_root: Path, trading_date: date) -> Path:
        return feature_root / f"{trading_date.year}" / f"{trading_date.month:02d}" / f"features_{trading_date.isoformat()}.parquet"

    def run_day(self, trading_date: date) -> None:
        """Run full feature engineering for one trading day.

        Args:
            trading_date: Trading date to process.
        """
        start_ts = perf_counter()
        in_path = self._clean_path(self.cfg.clean_root, trading_date)
        if not in_path.exists():
            raise FileNotFoundError(f"Clean parquet missing for {trading_date}: {in_path}")

        day_df = pd.read_parquet(in_path)
        day_df["timestamp"] = pd.to_datetime(day_df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        day_df = day_df.set_index("timestamp").sort_index()

        if "session_date" not in day_df.columns:
            day_df["session_date"] = pd.DatetimeIndex(pd.DatetimeIndex(day_df.index).date)

        feature_df = pd.DataFrame(index=day_df.index)

        passthrough_cols = [
            "session_date",
            "bar_index_in_session",
            "idx_log_ret",
            "idx_overnight_ret",
            "idx_vol_zscore",
            "days_to_expiry",
            "dte_normalized",
            "is_expiry_day",
            "regime_flag",
            "synth_atm_iv",
            "iv_skew_25d",
            "iv_butterfly",
            "opt_delta_net",
            "opt_gamma_net",
            "opt_vega_net",
            "gex",
            "total_pcr_vol",
            "total_pcr_oi",
            "oi_imbalance",
            "oi_flow",
        ]
        for col in passthrough_cols:
            if col in day_df.columns:
                feature_df[col] = day_df[col]

        # Session-wise VWAP deviation.
        feature_df["idx_vwap_dev"] = np.float32(np.nan)
        for _, idx in day_df.groupby("session_date", sort=True).groups.items():
            sess_idx = pd.Index(idx)
            feature_df.loc[sess_idx, "idx_vwap_dev"] = self.compute_vwap_deviation(day_df.loc[sess_idx])

        # Session-wise realized volatility metrics.
        if "idx_log_ret" in day_df.columns:
            ret = day_df["idx_log_ret"].astype("float32").fillna(0.0)
            grp = ret.groupby(day_df["session_date"], sort=True)
            rv5 = grp.transform(
                lambda s: np.sqrt(s.pow(2).rolling(5, min_periods=1).sum() * self.cfg.ANNUALIZATION)
            )
            rv30 = grp.transform(
                lambda s: np.sqrt(s.pow(2).rolling(30, min_periods=1).sum() * self.cfg.ANNUALIZATION)
            )
            vov = rv5.groupby(day_df["session_date"], sort=True).transform(
                lambda s: s.rolling(12, min_periods=2).std()
            )

            feature_df["idx_rv5m"] = rv5.astype("float32")
            feature_df["idx_rv30m"] = rv30.astype("float32")
            feature_df["idx_vov"] = vov.astype("float32")
        else:
            feature_df["idx_rv5m"] = np.float32(np.nan)
            feature_df["idx_rv30m"] = np.float32(np.nan)
            feature_df["idx_vov"] = np.float32(np.nan)

        # Momentum features.
        momentum = pd.DataFrame(index=day_df.index)
        for _, idx in day_df.groupby("session_date", sort=True).groups.items():
            sess_idx = pd.Index(idx)
            ret = day_df.loc[sess_idx, "idx_log_ret"].astype("float32") if "idx_log_ret" in day_df.columns else pd.Series(0.0, index=sess_idx)
            momentum.loc[sess_idx, ["M_5", "M_15", "M_30", "MDS", "idx_mom_decay"]] = self.compute_momentum_features(ret)
        feature_df = pd.concat([feature_df, momentum], axis=1)

        # Cross-sectional constituent features.
        const_feat = self.compute_constituent_features(day_df)
        feature_df = pd.concat([feature_df, const_feat], axis=1)

        # OI features and smoothing.
        oi_columns = [c for c in ["oi_imbalance", "oi_flow", "total_pcr_vol", "total_pcr_oi"] if c in feature_df.columns]
        if oi_columns:
            oi_feat = self.compute_oi_features(feature_df[oi_columns])
            feature_df = feature_df.drop(columns=[c for c in oi_feat.columns if c in feature_df.columns]).join(oi_feat)

        # Volatility risk premium.
        if "synth_atm_iv" in feature_df.columns and "idx_rv30m" in feature_df.columns:
            feature_df["idx_vrp"] = (
                feature_df["synth_atm_iv"].astype("float32") - feature_df["idx_rv30m"].astype("float32")
            ).astype("float32")

        # Ensure float32/int32 dtypes as required.
        for col in feature_df.columns:
            if col == "idx_target_1m":
                continue
            if feature_df[col].dtype == bool:
                continue
            if str(feature_df[col].dtype).startswith("int"):
                if col.endswith("_volume"):
                    feature_df[col] = feature_df[col].astype("int32")
                elif col in {"bar_index_in_session"}:
                    feature_df[col] = feature_df[col].astype("int16")
                else:
                    feature_df[col] = feature_df[col].astype("int8")
            elif col != "session_date":
                feature_df[col] = pd.to_numeric(feature_df[col], errors="coerce").astype("float32")

        feature_df = self.compute_target(feature_df)

        # Drop last bar of each session (target undefined) to enforce 374 rows/day.
        feature_df = feature_df[feature_df["idx_target_1m"].notna()].copy()
        feature_df["idx_target_1m"] = feature_df["idx_target_1m"].astype("int8")

        out_path = self._feature_path(self.cfg.feature_root, trading_date)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        feature_df.reset_index().to_parquet(out_path, index=False, compression="snappy")

        logger.info(
            "Feature engineering complete for {} rows={} cols={} elapsed={:.2f}s",
            trading_date,
            len(feature_df),
            len(feature_df.columns),
            perf_counter() - start_ts,
        )

        del day_df, feature_df, const_feat
        gc.collect()

    def run(self, date_range: list[date]) -> None:
        """Run feature engineering across multiple days.

        Args:
            date_range: Trading dates to process.
        """
        for trading_date in date_range:
            self.run_day(trading_date)
