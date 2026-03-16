from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import typer
from loguru import logger

from config import get_config
from feature_engine import FeatureEngine
from ingestion_engine import IngestionEngine
from logging_utils import configure_logging
from normalization_engine import NormalizationEngine, _pearson_statistic
from preprocessing_engine import PreprocessingEngine


app = typer.Typer(add_completion=False, no_args_is_help=True)


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _trading_days(start_date: date, end_date: date) -> list[date]:
    cal = mcal.get_calendar("BSE")
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
    return [d.date() for d in sessions]


def _iter_stage_files(stage_root: Path, pattern: str = "*.parquet") -> list[Path]:
    return sorted(stage_root.rglob(pattern))


def _date_filtered_files(paths: list[Path], start_date: date | None, end_date: date | None) -> list[Path]:
    if start_date is None and end_date is None:
        return paths

    kept: list[Path] = []
    for path in paths:
        token = path.stem.split("_")[-1]
        try:
            d = datetime.strptime(token, "%Y-%m-%d").date()
        except ValueError:
            kept.append(path)
            continue

        if start_date and d < start_date:
            continue
        if end_date and d > end_date:
            continue
        kept.append(path)
    return kept


def _load_stage_dataframe(stage: str, cfg, start_date: date | None, end_date: date | None) -> pd.DataFrame:
    if stage == "raw":
        paths = _date_filtered_files(_iter_stage_files(cfg.raw_root), start_date, end_date)
    elif stage == "clean":
        paths = _date_filtered_files(_iter_stage_files(cfg.clean_root), start_date, end_date)
    elif stage == "features":
        paths = _date_filtered_files(_iter_stage_files(cfg.feature_root), start_date, end_date)
    elif stage == "normalized":
        paths = _iter_stage_files(cfg.normalized_root)
    else:
        raise ValueError(f"Unsupported stage: {stage}")

    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:
            logger.warning("Failed reading {}: {}", path, exc)

    if not frames:
        raise ValueError(f"No parquet files found for stage={stage}")

    return pd.concat(frames, ignore_index=True)


def _validate_stage(
    stage: str,
    cfg,
    start_date: date | None = None,
    end_date: date | None = None,
) -> pd.DataFrame:
    df = _load_stage_dataframe(stage, cfg, start_date, end_date)
    checks: list[dict[str, object]] = []

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").dt.tz_convert("Asia/Kolkata")

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    exempt = pd.Series(False, index=df.index)
    if "bar_index_in_session" in df.columns:
        exempt = df["bar_index_in_session"].eq(0)

    nan_mask = df[numeric_cols].isna() if numeric_cols else pd.DataFrame(False, index=df.index, columns=[])
    if not nan_mask.empty:
        nan_rows = nan_mask.any(axis=1) & ~exempt
        inf_rows = np.isinf(df[numeric_cols]).any(axis=1)
        check1 = not (nan_rows.any() or inf_rows.any())
        details1 = f"nan_rows={int(nan_rows.sum())}, inf_rows={int(inf_rows.sum())}"
    else:
        check1 = True
        details1 = "no numeric columns"

    checks.append(
        {
            "check": "No NaN/Inf in features except session-open allowances",
            "pass": check1,
            "details": details1,
        }
    )

    if "idx_target_1m" in df.columns:
        target = pd.to_numeric(df["idx_target_1m"], errors="coerce")
        valid = target.dropna().isin([0, 1]).all()
        dtype_ok = str(df["idx_target_1m"].dtype) in {"int8", "Int8", "float32", "float64"}
        checks.append(
            {
                "check": "idx_target_1m has valid binary values and expected dtype",
                "pass": bool(valid and dtype_ok),
                "details": f"dtype={df['idx_target_1m'].dtype}",
            }
        )
    else:
        checks.append(
            {
                "check": "idx_target_1m has valid binary values and expected dtype",
                "pass": False,
                "details": "idx_target_1m missing",
            }
        )

    if "timestamp" in df.columns:
        ts = df["timestamp"].sort_values()
        monotonic = ts.is_monotonic_increasing
        no_dupes = not ts.duplicated().any()
        tz_ok = getattr(ts.dt.tz, "zone", None) == "Asia/Kolkata" or str(ts.dt.tz) == "Asia/Kolkata"
        checks.append(
            {
                "check": "Timestamps monotonic, unique, tz=Asia/Kolkata",
                "pass": bool(monotonic and no_dupes and tz_ok),
                "details": f"monotonic={monotonic}, duplicates={int(ts.duplicated().sum())}, tz={ts.dt.tz}",
            }
        )
    else:
        checks.append(
            {
                "check": "Timestamps monotonic, unique, tz=Asia/Kolkata",
                "pass": False,
                "details": "timestamp column missing",
            }
        )

    scaler_ok = True
    scaler_details: list[str] = []
    if stage == "normalized":
        for fold in range(1, 6):
            for scaler_name in [
                f"pt_iv_fold{fold}.pkl",
                f"xgb_volume_fold{fold}.pkl",
                f"xgb_greeks_fold{fold}.pkl",
                f"xgb_iv_fold{fold}.pkl",
            ]:
                scaler_path = cfg.scaler_root / scaler_name
                if not scaler_path.exists():
                    scaler_ok = False
                    scaler_details.append(f"missing:{scaler_name}")
                    continue
                try:
                    _ = joblib.load(scaler_path)
                except Exception:
                    scaler_ok = False
                    scaler_details.append(f"unloadable:{scaler_name}")
    else:
        scaler_details.append("skipped for non-normalized stage")

    checks.append(
        {
            "check": "Scaler .pkl files exist and are loadable",
            "pass": scaler_ok,
            "details": "; ".join(scaler_details) if scaler_details else "all scaler files OK",
        }
    )

    z_cols = [
        c
        for c in df.columns
        if c.endswith("_log_ret")
        or c in {
            "wrs_residual",
            "idx_mom_decay",
            "MDS",
            "idx_vwap_dev",
            "oi_imbalance",
            "oi_flow",
        }
    ]
    z_cols = [c for c in z_cols if pd.api.types.is_numeric_dtype(df[c])]

    stats_ok = True
    stats_notes: list[str] = []
    for col in z_cols:
        vals = pd.to_numeric(df[col], errors="coerce").dropna()
        if vals.empty:
            continue
        m = float(vals.mean())
        s = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
        if abs(m) > 0.3 or not (0.5 <= s <= 1.5):
            stats_ok = False
            stats_notes.append(f"{col}:mean={m:.3f},std={s:.3f}")

    checks.append(
        {
            "check": "Z-scored features have approx mean=0 and std=1",
            "pass": stats_ok,
            "details": "; ".join(stats_notes) if stats_notes else "all checked z-features within tolerance",
        }
    )

    leakage_ok = True
    leakage_notes: list[str] = []
    if stage == "normalized":
        split_dirs = sorted(cfg.normalized_root.glob("split_*"))
        for split_dir in split_dirs:
            train_path = split_dir / "lnn_train.parquet"
            val_path = split_dir / "lnn_val.parquet"
            if not train_path.exists() or not val_path.exists():
                continue

            train_df = pd.read_parquet(train_path)
            val_df = pd.read_parquet(val_path)
            if "idx_target_1m" not in val_df.columns:
                continue

            feat_cols = [
                c
                for c in train_df.columns
                if c not in {"timestamp", "session_date", "idx_target_1m", "is_expiry_day"}
                and pd.api.types.is_numeric_dtype(train_df[c])
            ]
            for col in feat_cols:
                tail = pd.to_numeric(train_df[col], errors="coerce").tail(100)
                head = pd.to_numeric(val_df["idx_target_1m"], errors="coerce").head(100)
                n = min(len(tail), len(head))
                if n < 3:
                    continue
                corr = _pearson_statistic(tail.tail(n), head.head(n))
                if np.isfinite(corr) and abs(corr) >= 0.05:
                    leakage_ok = False
                    leakage_notes.append(f"{split_dir.name}:{col}={corr:.4f}")

    checks.append(
        {
            "check": "Leakage check pearsonr(feature[-100], target[:100]) < 0.05",
            "pass": leakage_ok,
            "details": "; ".join(leakage_notes) if leakage_notes else "no leakage warnings",
        }
    )

    row_ok = True
    row_detail = "n/a"
    if "session_date" in df.columns:
        counts = df.groupby("session_date").size()
        if stage in {"features", "normalized"}:
            row_ok = bool((counts == 374).all())
            row_detail = f"min={int(counts.min())}, max={int(counts.max())}"
        else:
            row_ok = bool((counts >= 374).all())
            row_detail = f"min={int(counts.min())}, max={int(counts.max())}"

    checks.append(
        {
            "check": "Row count per day conforms to expected bars",
            "pass": row_ok,
            "details": row_detail,
        }
    )

    summary = pd.DataFrame(checks)
    return summary


@app.command("ingest")
def ingest_command(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Run ingestion for each trading day in range."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)
    engine = IngestionEngine(cfg)

    for d in _trading_days(_parse_date(start), _parse_date(end)):
        engine.run_day(d)


@app.command("preprocess")
def preprocess_command(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Run preprocessing for each trading day in range."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)
    engine = PreprocessingEngine(cfg)

    for d in _trading_days(_parse_date(start), _parse_date(end)):
        engine.run_day(d)


@app.command("features")
def features_command(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Run feature engineering for each trading day in range."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)
    engine = FeatureEngine(cfg)

    for d in _trading_days(_parse_date(start), _parse_date(end)):
        engine.run_day(d)


@app.command("normalize")
def normalize_command(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Run normalization over all feature files in date range."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)
    engine = NormalizationEngine(cfg)

    days = _trading_days(_parse_date(start), _parse_date(end))
    engine.run(cfg.feature_root, cfg.normalized_root, cfg.scaler_root, days)


@app.command("full-run")
def full_run_command(
    start: str = typer.Option(..., help="Start date YYYY-MM-DD"),
    end: str = typer.Option(..., help="End date YYYY-MM-DD"),
) -> None:
    """Run end-to-end pipeline day-by-day then normalize all folds."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)

    ingestion = IngestionEngine(cfg)
    preprocessing = PreprocessingEngine(cfg)
    features = FeatureEngine(cfg)
    normalize = NormalizationEngine(cfg)

    days = _trading_days(_parse_date(start), _parse_date(end))
    for d in days:
        ingestion.run_day(d)
        preprocessing.run_day(d)
        features.run_day(d)

    normalize.run(cfg.feature_root, cfg.normalized_root, cfg.scaler_root, days)


@app.command("validate")
def validate_command(
    stage: str = typer.Option(..., help="raw|clean|features|normalized"),
    start: str | None = typer.Option(None, help="Optional start date YYYY-MM-DD"),
    end: str | None = typer.Option(None, help="Optional end date YYYY-MM-DD"),
) -> None:
    """Validate pipeline outputs and print pass/fail summary table."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)

    start_date = _parse_date(start) if start else None
    end_date = _parse_date(end) if end else None

    summary = _validate_stage(stage, cfg, start_date, end_date)
    typer.echo(summary.to_string(index=False))


@app.command("refresh-instruments")
def refresh_instruments_command() -> None:
    """Refresh BSE instrument master from Upstox."""
    cfg = get_config()
    configure_logging("pipeline", cfg.LOG_ROOT)
    engine = IngestionEngine(cfg)
    engine.refresh_instrument_master()


if __name__ == "__main__":
    app()
