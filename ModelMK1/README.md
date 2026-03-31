# ModelMK1

Standalone Liquid NN + XGBoost hybrid for market range prediction.

## Design goals

- Clean-slate training artifacts in `outputs/`
- No mock-data pipeline or fallback paths
- Real data intake from `Training Data/` by default
- Auto device switching (`CUDA` when available, else `CPU`)
- Portable deployment (no system-specific absolute paths)

## Folder structure

- `main.py` â€” one entrypoint for build/tune/train/predict/backtest
- `src/modelmk1/` â€” all source code modules
- `Training Data/` â€” place training CSV/Parquet files here
- `outputs/` â€” generated model artifacts and metrics

## Required dataset columns

- `timestamp`
- `price`
- `high`
- `low`
- `volume`

## Commands

```bash
Route J (Dynamic Correlation Graph + T-GCN):

cd ModelMK1
/bin/python3 main.py build
/bin/python3 main.py pipeline --build-only
```

Start full training once real data exists in `Training Data/` (or pass `--data-path`):

```bash
cd ModelMK1
/bin/python3 main.py pipeline --trials 30
```

Optional explicit path:

```bash
cd ModelMK1
/bin/python3 main.py pipeline --data-path "./Training Data/sensex_ticks.parquet" --trials 30
```

Route J (Dynamic Correlation Graph + T-GCN):

```bash
cd ModelMK1
/bin/python3 main.py train-graph --data-path "./Training Data/sensex_constituents"
```

Route-J data sources supported:

- Directory of files (one file per constituent) with `timestamp` and a price-like column (`price`, `close`, `idx_close`, `last`, or `ltp`)
- Single long-format file with columns `timestamp`, `ticker`, and price-like column
- Single wide-format file with `timestamp` plus one column per constituent

XGBoost spectral route (top-10 constituent 10x10 correlation matrix + eigenvalue features):

```bash
cd ModelMK1
/bin/python3 main.py train-xgb-spectral --data-path "./Training Data/sensex_constituents" --top-n 10 --corr-window 60
```

Spectral route outputs are saved to `outputs/model/`:

- `sensex_top10_correlation_matrix.csv`
- `xgb_spectral_top10.json`
- `xgb_spectral_metrics.json`

Prediction and evaluation:

```bash
cd ModelMK1
/bin/python3 main.py predict
/bin/python3 main.py backtest
```
