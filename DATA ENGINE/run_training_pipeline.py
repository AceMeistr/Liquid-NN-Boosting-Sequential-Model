"""
run_training_pipeline.py
========================
One-command pipeline runner for the DATA ENGINE project.

Steps executed (1-minute data only):
  1. Convert DAT/ CSVs → TRAINING DATA/raw/   (Parquet, Snappy-compressed)
  2. Preprocess each trading day                → TRAINING DATA/clean/
  3. Feature-engineer each trading day          → TRAINING DATA/features/
  4. Normalize + build purged fold splits       → TRAINING DATA/normalized/  +  TRAINING DATA/scalers/

All outputs go to ./TRAINING DATA — completely isolated from the extractor's DAT/ folder.

Usage:
    python run_training_pipeline.py

Run from the DATA ENGINE project folder (DATA ENGINE/).
This script never modifies the extractor project in any way.
"""
from __future__ import annotations

import gc
import os
import sys
import importlib
from datetime import date, datetime
from pathlib import Path
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
#  DATE RANGE CONFIGURATION
#  Adjust these to control which days are processed.
# ─────────────────────────────────────────────────────────────────────────────

PIPELINE_START      = date(2022, 1,  1)   # T1/T2/T3 data starts here
PIPELINE_END        = date(2026, 3, 10)   # adjust to today or specific end date

# ─────────────────────────────────────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────────────────────────────────────

_HERE    = Path(__file__).parent.resolve()   # DATA ENGINE folder
_DAT     = Path(
    r"C:\Users\sasan\Downloads\MFT\DATA Importing\DAT"
)
_TRAIN   = _HERE / "TRAINING DATA"          # all outputs go here; never touches DAT/

# ─────────────────────────────────────────────────────────────────────────────
#  SETUP: override DATA_ROOT env before config is imported
# ─────────────────────────────────────────────────────────────────────────────

os.environ["DATA_ROOT"] = str(_TRAIN)
os.chdir(_HERE)                              # ensure relative imports in engines work

# Import engines after env is set
from loguru import logger
from logging_utils import configure_logging
from config import get_config
import pandas_market_calendars as mcal

from preprocessing_engine import PreprocessingEngine
from feature_engine import FeatureEngine
from normalization_engine import NormalizationEngine

# Import dat_connector functions directly (avoid subprocess overhead)
_dc = importlib.import_module("dat_connector")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _trading_days(start: date, end: date) -> list[date]:
    """Return BSE trading days between start and end (inclusive)."""
    cal = mcal.get_calendar("BSE")
    sched = cal.schedule(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
    )
    return [d.date() for d in sched.index.tolist()]


# ──────────────────────────────────────────────────────────────────────────────
#  DAT SOURCE LAYOUT
#  Maps symbol prefix (used in column names) → CSV filename stem in DAT/
# ──────────────────────────────────────────────────────────────────────────────
_SYMBOL_MAP = {
    "idx":        "sensex_index",
    "vix":        "india_vix",
    "BAJFINANCE": "constituent_BAJFINANCE",
    "BHARTIARTL": "constituent_BHARTIARTL",
    "HDFCBANK":   "constituent_HDFCBANK",
    "ICICIBANK":  "constituent_ICICIBANK",
    "INFY":       "constituent_INFY",
    "ITC":        "constituent_ITC",
    "LT":         "constituent_LT",
    "RELIANCE":   "constituent_RELIANCE",
    "SBIN":       "constituent_SBIN",
    "TCS":        "constituent_TCS",
    "opt":        "sensex_options_weekly",
}

_IST = "Asia/Kolkata"

# The wide-format requires prefixing columns.
# _OHLCV handled standard bars. Options have many more cols (Greeks, IV, etc.)
# If the column is 'timestamp' we keep it. For OHLCV we prefix.
# For options specific columns (synth_atm_iv, opt_delta_net, etc) we pass them through.
_OHLCV = ["open", "high", "low", "close", "volume"]

# Columns the FeatureEngine explicitly looks for in the merged master dataframe:
_OPTIONS_COLS = {
    "synth_atm_iv": "synth_atm_iv",
    "iv_skew_25d": "iv_skew_25d",
    "iv_butterfly": "iv_butterfly",
    "opt_delta_net": "opt_delta_net",
    "opt_gamma_net": "opt_gamma_net",
    "opt_vega_net": "opt_vega_net",
    "gex": "gex",
    "total_pcr_vol": "total_pcr_vol",
    "total_pcr_oi": "total_pcr_oi",
    "oi_imbalance": "oi_imbalance",
    "oi_flow": "oi_flow",
}

def _ensure_static_files(cfg) -> None:
    """Create empty placeholder CSVs for required static files if missing."""
    stubs = {
        cfg.CA_TABLE_PATH:          "symbol,ex_date,split_ratio,dividend_amount\n",
        cfg.WEIGHTS_PATH:           "symbol,weight\n",
        cfg.REGIME_TABLE_PATH:      "date_from,date_to,expiry_weekday,regime_flag\n2000-01-01,2030-12-31,3,0\n",
        cfg.RFR_PATH:               "date,rate\n",
        cfg.INSTRUMENT_MASTER_PATH: "symbol,segment,instrument_key\n",
    }
    for path, header in stubs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text(header, encoding="utf-8")


def _load_dat_csv(dat_dir: Path, stem: str) -> pd.DataFrame:
    """Load a 1m DAT CSV, parse timestamp as IST DatetimeIndex."""
    csv_path = dat_dir / f"{stem}_1m.csv"
    if not csv_path.exists():
        return pd.DataFrame()
    chunks = []
    for chunk in pd.read_csv(csv_path, chunksize=100_000, low_memory=False):
        if "timestamp" in chunk.columns:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"], format="mixed", utc=False)
            if chunk["timestamp"].dt.tz is None:
                chunk["timestamp"] = chunk["timestamp"].dt.tz_localize(_IST, ambiguous="NaT", nonexistent="NaT")
            else:
                chunk["timestamp"] = chunk["timestamp"].dt.tz_convert(_IST)
        chunks.append(chunk)
    if not chunks:
        return pd.DataFrame()
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def _build_session_date(ts_series: pd.Series) -> pd.Series:
    return ts_series.dt.date


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 1 — Build per-day master_{date}.parquet files in raw/{YYYY}/{MM}/
# ─────────────────────────────────────────────────────────────────────────────

def _read_csv_day(csv_path: Path, date_str: str) -> pd.DataFrame:
    """Read a CSV file iteratively and extract rows matching the target date."""
    if not csv_path.exists():
        return pd.DataFrame()

    chunks = []
    # Use chunking to limit RAM; filter strictly to date_str matching
    for chunk in pd.read_csv(csv_path, chunksize=50_000, low_memory=False):
        if "timestamp" not in chunk.columns:
            continue
        # Use fast string prefix matching for 'YYYY-MM-DD' since timestamps are ISO
        mask = chunk["timestamp"].str.startswith(date_str)
        if mask.any():
            chunks.append(chunk[mask])
        elif chunks and not mask.any():
            # If we already found the day and now we see no matches, we've passed the day
            # since the CSV is chronologically sorted. Safe to stop reading early.
            break

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="mixed", utc=False)
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(_IST, ambiguous="NaT", nonexistent="NaT")
    else:
        df["timestamp"] = df["timestamp"].dt.tz_convert(_IST)
    return df


def step_convert(cfg) -> None:
    logger.info("=" * 55)
    logger.info("STEP 1  — Building per-day raw Parquet files (Streaming)")
    logger.info(f"          Source: {_DAT}")
    logger.info(f"          Target: {cfg.raw_root}")
    logger.info("=" * 55)

    if not _DAT.exists():
        logger.error(f"DAT folder not found: {_DAT}")
        sys.exit(1)

    _ensure_static_files(cfg)

    # First, quickly scan just the Sensex Index file to find all trading days
    idx_csv = _DAT / f"{_SYMBOL_MAP['idx']}_1m.csv"
    if not idx_csv.exists():
        logger.error(f"Sensex index missing: {idx_csv}")
        sys.exit(1)

    logger.info("Scanning index CSV to build target date list …")
    all_dates = set()
    for chunk in pd.read_csv(idx_csv, usecols=["timestamp"], chunksize=250_000):
        # Extract the date part "YYYY-MM-DD" via string slice for speed
        days = chunk["timestamp"].str.slice(0, 10).unique()
        all_dates.update(days)

    valid_dates = []
    for d_str in sorted(list(all_dates)):
        try:
            d = date.fromisoformat(d_str)
            if PIPELINE_START <= d <= PIPELINE_END:
                valid_dates.append(d)
        except ValueError:
            pass

    logger.info(f"Found {len(valid_dates)} trading days to process.")

    written = 0
    skipped = 0

    for d in valid_dates:
        d_str = d.isoformat()
        raw_path = cfg.raw_root / f"{d.year}" / f"{d.month:02d}" / f"master_{d_str}.parquet"
        if raw_path.exists():
            skipped += 1
            if skipped % 50 == 0:
                logger.info(f"  Skipped {skipped} existing days …")
            continue

        slices = []
        for prefix, stem in _SYMBOL_MAP.items():
            csv_path = _DAT / f"{stem}_1m.csv"
            df = _read_csv_day(csv_path, d_str)
            if df.empty:
                continue

            if stem == "sensex_options_weekly":
                from ingestion_engine import IngestionEngine

                class _DummyEngine:
                    def __init__(self, c): self.cfg = c
                
                dummy = _DummyEngine(cfg)
                feature_rows = []

                for ts, grp in df.groupby("timestamp"):
                    if "underlying" not in grp.columns:
                        continue
                    spot = float(grp["underlying"].iloc[0])
                    
                    calls = grp[["strike", "ce_iv", "ce_delta", "ce_gamma", "ce_vega", "ce_vol", "ce_oi"]].copy()
                    calls.columns = ["strike", "iv", "delta", "gamma", "vega", "volume", "oi"]
                    calls["option_type"] = "C"

                    puts = grp[["strike", "pe_iv", "pe_delta", "pe_gamma", "pe_vega", "pe_vol", "pe_oi"]].copy()
                    puts.columns = ["strike", "iv", "delta", "gamma", "vega", "volume", "oi"]
                    puts["option_type"] = "P"

                    opt_df = pd.concat([calls, puts], ignore_index=True).set_index("strike")
                    
                    # Compute using native engine math!
                    feat_dict = IngestionEngine._derive_option_features(dummy, opt_df, spot)
                    feat_dict["timestamp"] = ts
                    feature_rows.append(feat_dict)

                if feature_rows:
                    res_df = pd.DataFrame(feature_rows).set_index("timestamp").sort_index()
                    res_df = res_df[~res_df.index.duplicated(keep="first")]
                    slices.append(res_df)
                continue

            # Rename and format normal OHLCV symbols
            cols_keep = {"timestamp": "timestamp"}
            for col in _OHLCV:
                if col in df.columns:
                    cols_keep[col] = f"{prefix}_{col}"
            
            # Allow pre-calculated options features to pass through unchanged 
            # (they don't need the 'opt_' prefix, FeatureEngine expects them as-is)
            for col, mapped_name in _OPTIONS_COLS.items():
                if col in df.columns:
                    cols_keep[col] = mapped_name

            df = df.rename(columns=cols_keep)[list(cols_keep.values())]
            df = df.set_index("timestamp").sort_index()

            # Handle duplicate timestamps gently (options chain can have multiple rows per timestamp right now)
            # Take the first occurrence (closest to ATM logic historically, or clean it later)
            df = df[~df.index.duplicated(keep="first")]
            slices.append(df)

        if not slices:
            continue

        master = slices[0]
        for s in slices[1:]:
            master = master.join(s, how="outer")

        master["session_date"] = d_str
        master["bar_index_in_session"] = range(len(master))

        raw_path.parent.mkdir(parents=True, exist_ok=True)
        master.reset_index().to_parquet(raw_path, index=False, compression="snappy")
        written += 1

        del master
        del slices
        gc.collect()

        if written % 10 == 0:
            logger.info(f"  Written {written}/{len(valid_dates)} days … (latest: {d_str})")

    logger.success(f"STEP 1 complete: {written} written, {skipped} already existed.")
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 2 — Preprocessing  (raw → clean, one day at a time)
# ─────────────────────────────────────────────────────────────────────────────

def step_preprocess(cfg, trading_days: list[date]) -> None:
    logger.info("=" * 55)
    logger.info(f"STEP 2  — Preprocessing  ({len(trading_days)} trading days)")
    logger.info("=" * 55)

    engine    = PreprocessingEngine(cfg)
    completed = 0
    for d in trading_days:
        clean_path = cfg.clean_root / f"{d.strftime('%Y-%m-%d')}.parquet"
        if clean_path.exists():
            continue   # already done, skip
        try:
            engine.run_day(d)
            completed += 1
        except Exception as exc:
            logger.warning(f"  Preprocess skipped {d}: {exc}")
        if completed % 50 == 0 and completed:
            gc.collect()

    logger.success(f"STEP 2 complete. Processed {completed} new day(s).")
    del engine
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 3 — Feature Engineering  (clean → features, one day at a time)
# ─────────────────────────────────────────────────────────────────────────────

def step_features(cfg, trading_days: list[date]) -> None:
    logger.info("=" * 55)
    logger.info(f"STEP 3  — Feature Engineering  ({len(trading_days)} trading days)")
    logger.info("=" * 55)

    engine    = FeatureEngine(cfg)
    completed = 0
    for d in trading_days:
        feat_path = cfg.feature_root / f"{d.strftime('%Y-%m-%d')}.parquet"
        if feat_path.exists():
            continue   # already done, skip
        try:
            engine.run_day(d)
            completed += 1
        except Exception as exc:
            logger.warning(f"  Features skipped {d}: {exc}")
        if completed % 50 == 0 and completed:
            gc.collect()

    logger.success(f"STEP 3 complete. Processed {completed} new day(s).")
    del engine
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  STEP 4 — Normalization + Fold Splits  (features → normalized + scalers)
# ─────────────────────────────────────────────────────────────────────────────

def step_normalize(cfg, trading_days: list[date]) -> None:
    logger.info("=" * 55)
    logger.info(f"STEP 4  — Normalization  ({len(trading_days)} trading days)")
    logger.info("=" * 55)

    engine = NormalizationEngine(cfg)
    engine.run(
        feature_dir = cfg.feature_root,
        output_dir  = cfg.normalized_root,
        scaler_dir  = cfg.scaler_root,
        date_range  = trading_days,
    )
    logger.success("STEP 4 complete.")
    del engine
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    configure_logging("training_pipeline", _TRAIN / "logs")
    cfg = get_config()

    # Announce output location clearly
    logger.info(f"")
    logger.info(f"  DATA ENGINE — TRAINING PIPELINE")
    logger.info(f"  Source  :  {_DAT}")
    logger.info(f"  Output  :  {_TRAIN}")
    logger.info(f"  Range   :  {PIPELINE_START} → {PIPELINE_END}")
    logger.info(f"")

    # Ensure folder layout inside TRAINING DATA
    cfg.ensure_directories()

    trading_days = _trading_days(PIPELINE_START, PIPELINE_END)
    logger.info(f"Trading days in range: {len(trading_days)}")

    step_convert(cfg)
    step_preprocess(cfg, trading_days)
    step_features(cfg, trading_days)
    step_normalize(cfg, trading_days)

    logger.success(f"")
    logger.success(f"  ALL STEPS COMPLETE")
    logger.success(f"  Training data saved to: {_TRAIN}")
    logger.success(f"    raw/         ← 1m Parquet files from DAT/")
    logger.success(f"    clean/       ← validated + imputed bars")
    logger.success(f"    features/    ← Greeks, IVS, momentum, constituent features")
    logger.success(f"    normalized/  ← purged-fold normalized datasets")
    logger.success(f"    scalers/     ← fitted LNN + XGB scaler objects (.pkl)")
    logger.success(f"")


if __name__ == "__main__":
    main()
