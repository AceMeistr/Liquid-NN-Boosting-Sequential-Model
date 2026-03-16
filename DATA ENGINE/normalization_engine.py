from __future__ import annotations

import gc
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from loguru import logger
from scipy.stats import pearsonr, skew
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import PowerTransformer, RobustScaler, StandardScaler

from config import PipelineConfig
from logging_utils import configure_logging


def _pearson_statistic(x: pd.Series, y: pd.Series) -> float:
    """Return pearson correlation statistic across scipy return shapes."""
    x_num = pd.to_numeric(x, errors="coerce")
    y_num = pd.to_numeric(y, errors="coerce")
    valid = x_num.notna() & y_num.notna()
    x_num, y_num = x_num[valid], y_num[valid]
    if len(x_num) < 3:
        return 0.0
    if float(x_num.std(ddof=0)) < 1e-12 or float(y_num.std(ddof=0)) < 1e-12:
        return 0.0

    result = pearsonr(x_num.to_numpy(), y_num.to_numpy())
    statistic = getattr(result, "statistic", None)
    if statistic is not None:
        return float(statistic)
    if isinstance(result, tuple):
        first: Any = result[0]
        return float(first)
    return float(result)


class NormalizationEngine:
    """Normalization and split engine for LNN/XGB training datasets."""

    def __init__(self, config: PipelineConfig):
        """Initialize normalization engine.

        Args:
            config: Pipeline configuration instance.
        """
        self.cfg = config
        configure_logging("normalization_engine", self.cfg.LOG_ROOT)

    def session_aware_rolling_zscore(
        self,
        series: pd.Series,
        session_col: pd.Series,
        window: int = 60,
    ) -> pd.Series:
        """Compute session-bounded rolling z-score with expanding warm-up.

        Uses a rolling window once min_obs samples are available, falling back
        to an expanding window for early-session bars. Sigma is floored at 1e-6
        to prevent numerically extreme z-scores.

        Args:
            series: Input feature series.
            session_col: Session labels aligned with series index.
            window: Rolling window length.

        Returns:
            Session-aware clipped z-score series.
        """
        vals = pd.to_numeric(series, errors="coerce")
        grouped = vals.groupby(session_col, sort=True)

        min_obs = max(20, window // 3)
        roll_mu = grouped.transform(lambda s: s.rolling(window, min_periods=min_obs).mean())
        roll_sigma = grouped.transform(lambda s: s.rolling(window, min_periods=min_obs).std())

        # Expanding fallback for early-session bars before min_obs is reached.
        exp_mu = grouped.transform(lambda s: s.expanding(min_periods=2).mean())
        exp_sigma = grouped.transform(lambda s: s.expanding(min_periods=2).std())

        mu = roll_mu.where(roll_mu.notna(), exp_mu)
        sigma = roll_sigma.where(roll_sigma.notna(), exp_sigma)

        # Physically meaningful sigma floor to prevent extreme z-scores.
        sigma = sigma.clip(lower=1e-6)

        z = (vals - mu) / sigma
        z = z.clip(-self.cfg.ZSCORE_CLIP, self.cfg.ZSCORE_CLIP).astype("float32")

        self._last_roll_mu = mu.astype("float32")
        self._last_roll_std = sigma.astype("float32")
        return z

    @staticmethod
    def _iqr_clip(
        train_vals: pd.DataFrame,
        val_vals: pd.DataFrame,
        k: float = 4.0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Clip extreme outliers using IQR fences from training data.

        Args:
            train_vals: Training feature values.
            val_vals: Validation feature values.
            k: IQR multiplier for fence width.

        Returns:
            Clipped (train, val) tuple.
        """
        q1 = train_vals.quantile(0.25)
        q3 = train_vals.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - k * iqr
        upper = q3 + k * iqr
        return train_vals.clip(lower, upper, axis=1), val_vals.clip(lower, upper, axis=1)

    def build_purged_splits(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
    ) -> list[tuple[pd.Index, pd.Index]]:
        """Build purged time-series folds with embargo.

        Args:
            df: Full feature dataframe sorted by time.
            n_splits: Number of temporal folds.

        Returns:
            List of (train_idx, val_idx) index tuples.
        """
        ordered = df.sort_values("timestamp").reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        splits: list[tuple[pd.Index, pd.Index]] = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(ordered), start=1):
            train_part = ordered.iloc[train_idx]
            val_part = ordered.iloc[val_idx]

            if train_part.empty or val_part.empty:
                continue

            # Enforce train end at session boundary by trimming to full sessions.
            last_train_session = train_part["session_date"].iloc[-1]
            train_idx = train_part.index[train_part["session_date"] <= last_train_session]
            train_part = ordered.loc[train_idx]

            train_end = pd.Timestamp(train_part["timestamp"].iloc[-1])
            embargo_end = train_end + timedelta(minutes=self.cfg.EMBARGO_MINUTES)

            val_mask = val_part["timestamp"] >= embargo_end
            val_part = val_part.loc[val_mask]
            if val_part.empty:
                logger.warning("Fold {} has empty validation set after embargo.", fold)
                continue

            # Align validation start to next session open where possible.
            if "bar_index_in_session" in val_part.columns:
                open_rows = val_part[val_part["bar_index_in_session"] == 0]
                if not open_rows.empty:
                    first_open_ts = open_rows["timestamp"].iloc[0]
                    val_part = val_part[val_part["timestamp"] >= first_open_ts]

            if val_part.empty:
                logger.warning("Fold {} has empty validation set after session-open alignment.", fold)
                continue

            # Leakage sanity check around split boundary.
            feature_cols = [
                c
                for c in ordered.columns
                if c
                not in {
                    "timestamp",
                    "session_date",
                    "idx_target_1m",
                    "is_expiry_day",
                }
                and pd.api.types.is_numeric_dtype(ordered[c])
            ]
            for col in feature_cols:
                train_tail = train_part[col].tail(100).astype("float32")
                val_head_target = val_part["idx_target_1m"].head(100).astype("float32")
                if len(train_tail) < 3 or len(val_head_target) < 3:
                    continue
                n = min(len(train_tail), len(val_head_target))
                corr = _pearson_statistic(train_tail.tail(n), val_head_target.head(n))
                if np.isfinite(corr) and abs(corr) >= 0.05:
                    logger.warning("Fold {} leakage warning: feature={} corr={:.4f}", fold, col, corr)

            splits.append((pd.Index(train_part.index), pd.Index(val_part.index)))

        return splits

    def fit_and_apply_lnn_scalers(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        fold_id: int,
        scaler_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply LNN scaling strategy for one fold.

        Args:
            train_df: Training fold data.
            val_df: Validation fold data.
            fold_id: Fold id.
            scaler_dir: Directory to persist fitted scalers.

        Returns:
            Tuple of scaled train and validation dataframes.
        """
        scaler_dir.mkdir(parents=True, exist_ok=True)
        train = train_df.copy()
        val = val_df.copy()

        price_features = sorted(
            {
                *[c for c in train.columns if c.endswith("_log_ret")],
                "wrs_residual",
                "sector_bank_flow",
                "sector_it_flow",
                "sector_energy_flow",
                "sector_industrial_flow",
                "top5_dispersion",
                "MDS",
                "M_5",
                "M_15",
                "M_30",
                "idx_mom_decay",
                "idx_vwap_dev",
                "idx_vrp",
            }
        )
        price_features = [c for c in price_features if c in train.columns and c in val.columns]

        vol_features = [
            c
            for c in [
                "synth_atm_iv",
                "idx_rv5m",
                "idx_rv30m",
                "idx_vrp",
                "iv_skew_25d",
                "iv_butterfly",
            ]
            if c in train.columns and c in val.columns
        ]

        for col in price_features:
            train[col] = self.session_aware_rolling_zscore(train[col], train["session_date"], window=self.cfg.ROLLING_WINDOW_Z)
            val[col] = self.session_aware_rolling_zscore(val[col], val["session_date"], window=self.cfg.ROLLING_WINDOW_Z)

        if "idx_log_ret" in train.columns:
            _ = self.session_aware_rolling_zscore(train["idx_log_ret"], train["session_date"], window=self.cfg.ROLLING_WINDOW_Z)
            train["roll_mu_60"] = self._last_roll_mu.astype("float32")
            train["roll_std_60"] = self._last_roll_std.astype("float32")

            _ = self.session_aware_rolling_zscore(val["idx_log_ret"], val["session_date"], window=self.cfg.ROLLING_WINDOW_Z)
            val["roll_mu_60"] = self._last_roll_mu.astype("float32")
            val["roll_std_60"] = self._last_roll_std.astype("float32")

        if vol_features:
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            train_vals = train[vol_features].replace([np.inf, -np.inf], np.nan)
            train_medians = train_vals.median().fillna(0.0)
            train_vals = train_vals.fillna(train_medians)
            val_vals = val[vol_features].replace([np.inf, -np.inf], np.nan).fillna(train_medians)

            train_vals, val_vals = self._iqr_clip(train_vals, val_vals)

            pt.fit(train_vals)
            train.loc[:, vol_features] = pt.transform(train_vals).astype("float32")
            val.loc[:, vol_features] = pt.transform(val_vals).astype("float32")
            joblib.dump(pt, scaler_dir / f"pt_iv_fold{fold_id}.pkl")

            iv_like = [c for c in vol_features if "iv" in c]
            for col in iv_like:
                col_values = train[col].astype("float32")
                if float(col_values.std(ddof=0)) < 1e-8:
                    continue
                col_skew = float(skew(col_values, nan_policy="omit"))
                if np.isfinite(col_skew) and abs(col_skew) >= 0.5:
                    logger.warning("Fold {}: transformed skew too high col={} skew={:.4f}", fold_id, col, col_skew)

        return train, val

    def fit_and_apply_xgb_scalers(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        fold_id: int,
        scaler_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply XGB scaling strategy for one fold.

        Args:
            train_df: Training fold data.
            val_df: Validation fold data.
            fold_id: Fold id.
            scaler_dir: Directory for serialized scalers.

        Returns:
            Tuple of scaled train and validation dataframes.
        """
        scaler_dir.mkdir(parents=True, exist_ok=True)
        train = train_df.copy()
        val = val_df.copy()

        volume_features = [
            c
            for c in [
                *[col for col in train.columns if col.endswith("_volume")],
                "idx_vol_zscore",
                "stale_feed_count",
            ]
            if c in train.columns and c in val.columns
        ]

        greeks_features = [
            c
            for c in [
                "opt_delta_net",
                "opt_gamma_net",
                "opt_vega_net",
                "gex",
                "const_adv_dec",
                "adr",
                "oi_imbalance",
                "oi_flow",
            ]
            if c in train.columns and c in val.columns
        ]

        iv_features = [
            c
            for c in [
                "synth_atm_iv",
                "iv_skew_25d",
                "iv_butterfly",
                "total_pcr_oi",
                "total_pcr_vol",
            ]
            if c in train.columns and c in val.columns
        ]

        if volume_features:
            rs = RobustScaler()
            for col in volume_features:
                train[col] = pd.to_numeric(train[col], errors="coerce").astype("float32")
                val[col] = pd.to_numeric(val[col], errors="coerce").astype("float32")
            train_vals = train[volume_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            val_vals = val[volume_features].replace([np.inf, -np.inf], np.nan).fillna(0.0)
            train_vals, val_vals = self._iqr_clip(train_vals, val_vals)
            rs.fit(train_vals)
            train.loc[:, volume_features] = rs.transform(train_vals).astype("float32")
            val.loc[:, volume_features] = rs.transform(val_vals).astype("float32")
            joblib.dump(rs, scaler_dir / f"xgb_volume_fold{fold_id}.pkl")

        if greeks_features:
            ss = StandardScaler()
            train_vals = train[greeks_features].replace([np.inf, -np.inf], np.nan)
            train_medians = train_vals.median().fillna(0.0)
            train_vals = train_vals.fillna(train_medians)
            val_vals = val[greeks_features].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
            ss.fit(train_vals)
            train.loc[:, greeks_features] = ss.transform(train_vals).astype("float32")
            val.loc[:, greeks_features] = ss.transform(val_vals).astype("float32")
            joblib.dump(ss, scaler_dir / f"xgb_greeks_fold{fold_id}.pkl")

        if iv_features:
            pt = PowerTransformer(method="yeo-johnson", standardize=True)
            train_vals = train[iv_features].replace([np.inf, -np.inf], np.nan)
            train_medians = train_vals.median().fillna(0.0)
            train_vals = train_vals.fillna(train_medians)
            val_vals = val[iv_features].replace([np.inf, -np.inf], np.nan).fillna(train_medians)
            train_vals, val_vals = self._iqr_clip(train_vals, val_vals)
            pt.fit(train_vals)
            train.loc[:, iv_features] = pt.transform(train_vals).astype("float32")
            val.loc[:, iv_features] = pt.transform(val_vals).astype("float32")
            joblib.dump(pt, scaler_dir / f"xgb_iv_fold{fold_id}.pkl")

        return train, val

    @staticmethod
    def _feature_path(feature_root: Path, trading_date: date) -> Path:
        return feature_root / f"{trading_date.year}" / f"{trading_date.month:02d}" / f"features_{trading_date.isoformat()}.parquet"

    @staticmethod
    def _load_parquet_batches(path: Path, batch_size: int = 50_000) -> list[pd.DataFrame]:
        pf = pq.ParquetFile(path)
        frames: list[pd.DataFrame] = []
        for batch in pf.iter_batches(batch_size=batch_size):
            frames.append(batch.to_pandas())
        return frames

    def _load_feature_range(self, feature_dir: Path, date_range: Iterable[date]) -> pd.DataFrame:
        frames: list[pd.DataFrame] = []
        for d in date_range:
            path = self._feature_path(feature_dir, d)
            if not path.exists():
                logger.warning("Skipping missing feature file: {}", path)
                continue
            frames.extend(self._load_parquet_batches(path, batch_size=50_000))

        if not frames:
            raise ValueError("No feature frames loaded for requested date range.")

        df = pd.concat(frames, ignore_index=True)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        return df

    def _validate_output_files(self, output_dir: Path, scaler_dir: Path, n_folds: int) -> None:
        for fold in range(1, n_folds + 1):
            fold_dir = output_dir / f"split_{fold}"
            for name in ["lnn_train.parquet", "lnn_val.parquet", "xgb_train.parquet", "xgb_val.parquet"]:
                path = fold_dir / name
                if not path.exists():
                    raise FileNotFoundError(f"Missing normalized output: {path}")

            for scaler_name in [
                f"pt_iv_fold{fold}.pkl",
                f"xgb_volume_fold{fold}.pkl",
                f"xgb_greeks_fold{fold}.pkl",
                f"xgb_iv_fold{fold}.pkl",
            ]:
                scaler_path = scaler_dir / scaler_name
                if not scaler_path.exists():
                    raise FileNotFoundError(f"Missing scaler file: {scaler_path}")
                _ = joblib.load(scaler_path)

    def run(
        self,
        feature_dir: Path,
        output_dir: Path,
        scaler_dir: Path,
        date_range: list[date],
    ) -> None:
        """Run normalization stage over all dates and save fold outputs.

        Args:
            feature_dir: Input feature parquet directory.
            output_dir: Output normalized directory.
            scaler_dir: Scaler persistence directory.
            date_range: Trading dates to include.
        """
        df = self._load_feature_range(feature_dir, date_range)
        df = (
            df.sort_values("timestamp")
            .drop_duplicates(subset=["timestamp"])
            .reset_index(drop=True)
        )
        if not df["timestamp"].is_monotonic_increasing:
            raise AssertionError("Timestamp index is not monotonic after sorting.")

        splits = self.build_purged_splits(df, n_splits=5)
        if not splits:
            raise ValueError("No valid train/validation splits produced.")

        for fold, (train_idx, val_idx) in enumerate(splits, start=1):
            train_df = df.iloc[train_idx].copy()
            val_df = df.iloc[val_idx].copy()

            lnn_train, lnn_val = self.fit_and_apply_lnn_scalers(train_df, val_df, fold, scaler_dir)
            xgb_train, xgb_val = self.fit_and_apply_xgb_scalers(train_df, val_df, fold, scaler_dir)

            # Post-scaling Inf cleanup
            for out_df, label in [
                (lnn_train, "lnn_train"), (lnn_val, "lnn_val"),
                (xgb_train, "xgb_train"), (xgb_val, "xgb_val"),
            ]:
                num_cols = out_df.select_dtypes(include=[np.number]).columns
                inf_mask = np.isinf(out_df[num_cols].to_numpy())
                if inf_mask.any():
                    logger.warning("Fold {}: {} has {} Inf values, replacing", fold, label, int(inf_mask.sum()))
                    out_df[num_cols] = out_df[num_cols].replace([np.inf, -np.inf], np.nan)

            fold_dir = output_dir / f"split_{fold}"
            fold_dir.mkdir(parents=True, exist_ok=True)

            lnn_train.to_parquet(fold_dir / "lnn_train.parquet", index=False, compression="snappy")
            lnn_val.to_parquet(fold_dir / "lnn_val.parquet", index=False, compression="snappy")
            xgb_train.to_parquet(fold_dir / "xgb_train.parquet", index=False, compression="snappy")
            xgb_val.to_parquet(fold_dir / "xgb_val.parquet", index=False, compression="snappy")

            logger.info(
                "Saved fold={} train_rows={} val_rows={}",
                fold,
                len(train_df),
                len(val_df),
            )

            del train_df, val_df, lnn_train, lnn_val, xgb_train, xgb_val
            gc.collect()

        self._validate_output_files(output_dir, scaler_dir, n_folds=len(splits))
        logger.info("Normalization stage complete. folds={}", len(splits))
