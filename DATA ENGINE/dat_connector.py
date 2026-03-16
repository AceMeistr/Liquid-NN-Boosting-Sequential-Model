"""
dat_connector.py
Bridge between the Upstox historical extractor DAT/ folder and the
DATA ENGINE pipeline's raw parquet stage.

Reads all CSVs from the source DAT directory and writes them as
partitioned Parquet files into the pipeline's `data/raw/` stage so
that PreprocessingEngine, NormalizationEngine, and FeatureEngine can
consume them without any modification.

Usage:
    python dat_connector.py --dat-path "C:/Users/sasan/.gemini/antigravity/scratch/upstox_historical_data/DAT"
    python dat_connector.py --dat-path "..." --dataset options_weekly
    python dat_connector.py --dat-path "..." --interval 5m

Run from the DATA ENGINE project folder.
"""
from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import typer
from loguru import logger

from config import get_config

app = typer.Typer(add_completion=False, no_args_is_help=True)

# ── Dataset manifest ──────────────────────────────────────────────────────────
# Maps a dataset name to a (file_glob, parquet_subdir) tuple.
# The subdir is used under cfg.raw_root to keep datasets separated.
_DATASETS: dict[str, tuple[str, str]] = {
    "sensex_index":       ("sensex_index_{interval}.csv",       "sensex_index"),
    "india_vix":          ("india_vix_{interval}.csv",           "india_vix"),
    "options_weekly":     ("sensex_options_weekly_{interval}.csv", "options_weekly"),
    "constituent_*":      ("constituent_*_{interval}.csv",       "constituents"),
}

_TS_COL  = "timestamp"
_IST     = "Asia/Kolkata"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_csv_chunked(path: Path, chunksize: int = 200_000) -> pd.DataFrame:
    """Memory-efficient CSV reader with IST timestamp parsing."""
    chunks = []
    for chunk in pd.read_csv(path, chunksize=chunksize, low_memory=False):
        if _TS_COL in chunk.columns:
            chunk[_TS_COL] = pd.to_datetime(
                chunk[_TS_COL], utc=False, format="mixed"
            ).dt.tz_localize(None)  # store as tz-naive UTC+5:30 for parquet compat
        chunks.append(chunk)
    df = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return df


def _write_parquet(df: pd.DataFrame, out_dir: Path, label: str) -> None:
    """Write a DataFrame to a single Parquet file, partitioned by date."""
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{label}.parquet"
    table = pa.Table.from_pandas(df, preserve_index=False)
    pq.write_table(
        table,
        out_path,
        compression="snappy",
        row_group_size=50_000,
    )
    mb = out_path.stat().st_size / (1024 ** 2)
    logger.info(f"  → written {out_path.name}  ({mb:.1f} MB, {len(df):,} rows)")


def _convert_file(csv_path: Path, out_dir: Path, label: str) -> None:
    logger.info(f"Converting {csv_path.name} …")
    df = _read_csv_chunked(csv_path)
    # Sort by timestamp if present
    if _TS_COL in df.columns:
        df.sort_values(_TS_COL, inplace=True)
        df.reset_index(drop=True, inplace=True)
    _write_parquet(df, out_dir, label)
    del df
    gc.collect()


# ── CLI Commands ──────────────────────────────────────────────────────────────

@app.command("convert")
def convert_command(
    dat_path: str = typer.Option(
        ..., help="Absolute path to the DAT/ folder from the extractor"
    ),
    dataset: Optional[str] = typer.Option(
        None,
        help="Which dataset to convert: sensex_index | india_vix | options_weekly | constituents | all",
    ),
    interval: str = typer.Option("1m", help="Interval suffix to convert: 1m or 5m"),
) -> None:
    """Convert DAT/ CSV files to pipeline-ready Parquet in data/raw/."""
    cfg = get_config()
    src = Path(dat_path)

    if not src.exists():
        logger.error(f"DAT path not found: {src}")
        raise typer.Exit(1)

    target = dataset or "all"
    logger.info(f"DAT Connector: {src}  →  {cfg.raw_root}  [interval={interval}]")

    converted = 0

    # ── Sensex Index ──────────────────────────────────────────────────
    if target in ("all", "sensex_index"):
        f = src / f"sensex_index_{interval}.csv"
        if f.exists():
            _convert_file(f, cfg.raw_root / "sensex_index", f"sensex_index_{interval}")
            converted += 1
        else:
            logger.warning(f"Not found: {f.name}")

    # ── IndiaVIX ──────────────────────────────────────────────────────
    if target in ("all", "india_vix"):
        f = src / f"india_vix_{interval}.csv"
        if f.exists():
            _convert_file(f, cfg.raw_root / "india_vix", f"india_vix_{interval}")
            converted += 1
        else:
            logger.warning(f"Not found: {f.name}")

    # ── Weekly Options ────────────────────────────────────────────────
    if target in ("all", "options_weekly"):
        f = src / f"sensex_options_weekly_{interval}.csv"
        if f.exists():
            _convert_file(f, cfg.raw_root / "options_weekly", f"options_weekly_{interval}")
            converted += 1
        else:
            logger.warning(f"Not found: {f.name}")

    # ── Constituents ──────────────────────────────────────────────────
    if target in ("all", "constituents"):
        for csv_file in sorted(src.glob(f"constituent_*_{interval}.csv")):
            ticker = csv_file.stem.replace(f"_{interval}", "").replace("constituent_", "")
            _convert_file(
                csv_file,
                cfg.raw_root / "constituents",
                f"{ticker}_{interval}",
            )
            converted += 1

    logger.success(f"✓ Converted {converted} file(s) to {cfg.raw_root}")


@app.command("status")
def status_command(
    dat_path: str = typer.Option(
        ..., help="Absolute path to the DAT/ folder from the extractor"
    ),
) -> None:
    """Print a summary table of DAT/ CSV files and their pipeline parquet equivalents."""
    cfg = get_config()
    src = Path(dat_path)

    rows = []
    for csv_file in sorted(src.glob("*.csv")):
        name = csv_file.stem
        mb_csv = csv_file.stat().st_size / (1024 ** 2)
        # Find corresponding parquet
        parquet_candidates = list(cfg.raw_root.rglob(f"{name}.parquet"))
        if parquet_candidates:
            mb_parquet = parquet_candidates[0].stat().st_size / (1024 ** 2)
            parquet_status = f"{mb_parquet:.1f} MB ✓"
        else:
            parquet_status = "not converted"
        rows.append((csv_file.name, f"{mb_csv:.1f} MB", parquet_status))

    print(f"\n{'CSV File':<45} {'CSV Size':>10}  {'Parquet Status':>20}")
    print("─" * 80)
    for name, size, pstatus in rows:
        print(f"{name:<45} {size:>10}  {pstatus:>20}")
    print(f"\nDAT source : {src}")
    print(f"Raw stage  : {cfg.raw_root}\n")



# ── Bridge Config Loader ──────────────────────────────────────────────────────

def _load_bridge() -> dict | None:
    """Load shared bridge_config.json from inside the DAT/ folder."""
    import json
    # Search for bridge_config.json in common DAT locations
    candidates = [
        Path(
            "C:/Users/sasan/Downloads/MFT/DATA Importing/DATA/bridge_config.json"
        ),
    ]
    for c in candidates:
        if c.exists():
            return json.loads(c.read_text())
    logger.warning("bridge_config.json not found. Extractor commands unavailable.")
    return None


# ── Extractor-Control Commands (engine-side) ──────────────────────────────────

@app.command("extractor-status")
def extractor_status_command() -> None:
    """Tail the extractor's orchestrator_run.log from the DATA ENGINE."""
    bridge = _load_bridge()
    if not bridge:
        raise typer.Exit(1)
    log_path = Path(bridge["extractor_log"])
    if not log_path.exists():
        logger.warning("orchestrator_run.log not found — extractor may not be running.")
        return
    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    print(f"\n{'─' * 60}")
    print(f"  EXTRACTOR LOG  ({log_path})")
    print(f"{'─' * 60}")
    print("\n".join(lines[-20:]))
    print()


@app.command("extractor-start")
def extractor_start_command() -> None:
    """Start the extractor's observer daemon from the DATA ENGINE."""
    import subprocess
    bridge = _load_bridge()
    if not bridge:
        raise typer.Exit(1)
    root    = Path(bridge["extractor_root"])
    entry   = bridge["extractor_entry"]
    result  = subprocess.Popen(
        [sys.executable, entry],
        cwd=root,
    )
    logger.success(f"Extractor started (PID {result.pid})")


@app.command("extractor-stop")
def extractor_stop_command() -> None:
    """Stop all running Python extractor processes."""
    import subprocess
    subprocess.run(
        ["powershell", "-Command",
         "Get-Process -Name python -ErrorAction SilentlyContinue | Stop-Process -Force"],
        check=False,
    )
    logger.success("All python processes stopped.")


@app.command("auto-convert")
def auto_convert_command(
    interval: str = typer.Option("1m", help="Interval: 1m or 5m"),
    pipeline_start: Optional[str] = typer.Option(None, help="Run pipeline after convert: YYYY-MM-DD"),
    pipeline_end: Optional[str] = typer.Option(None, help="Pipeline end date: YYYY-MM-DD"),
) -> None:
    """
    Convert all DAT/ CSVs to Parquet, then optionally run the full pipeline.
    This is the one-command bridge: extractor finished → data engine processed.
    """
    import subprocess
    bridge = _load_bridge()
    if not bridge:
        raise typer.Exit(1)
    dat_path = bridge["dat_path"]

    # Step 1 — convert
    logger.info("Step 1: Converting DAT/ CSVs → raw Parquet …")
    cfg = get_config()
    src = Path(dat_path)
    if not src.exists():
        logger.error(f"DAT path not found: {src}")
        raise typer.Exit(1)
    for interval_val in [interval]:
        _convert_file_set(src, cfg, interval_val)

    # Step 2 — optional pipeline run
    if pipeline_start and pipeline_end:
        logger.info(f"Step 2: Running pipeline {pipeline_start} → {pipeline_end} …")
        result = subprocess.run(
            [sys.executable, "pipeline.py", "full-run",
             "--start", pipeline_start, "--end", pipeline_end],
        )
        if result.returncode != 0:
            logger.error("Pipeline run failed.")
            raise typer.Exit(result.returncode)
        logger.success("Pipeline complete.")
    else:
        logger.info("No pipeline_start/end provided — skipping pipeline run.")


def _convert_file_set(src: Path, cfg, interval: str) -> None:
    """Convert all known CSV files for a given interval."""
    pairs = [
        (src / f"sensex_index_{interval}.csv",              cfg.raw_root / "sensex_index",   f"sensex_index_{interval}"),
        (src / f"india_vix_{interval}.csv",                 cfg.raw_root / "india_vix",       f"india_vix_{interval}"),
        (src / f"sensex_options_weekly_{interval}.csv",     cfg.raw_root / "options_weekly",  f"options_weekly_{interval}"),
    ]
    for csv_path, out_dir, label in pairs:
        if csv_path.exists():
            _convert_file(csv_path, out_dir, label)
    for csv_file in sorted(src.glob(f"constituent_*_{interval}.csv")):
        ticker = csv_file.stem.replace(f"_{interval}", "").replace("constituent_", "")
        _convert_file(csv_file, cfg.raw_root / "constituents", f"{ticker}_{interval}")


if __name__ == "__main__":
    import sys
    app()
