# ModelMK1

Standalone Liquid NN + XGBoost hybrid for market range prediction.

## Design goals

- Clean-slate training artifacts in `outputs/`
- No mock-data pipeline or fallback paths
- Real data intake from `Training Data/` by default
- Auto device switching (`CUDA` when available, else `CPU`)
- Portable deployment (no system-specific absolute paths)

## Folder structure

- `main.py` — one entrypoint for build/tune/train/predict/backtest
- `src/modelmk1/` — all source code modules
- `Training Data/` — place training CSV/Parquet files here
- `outputs/` — generated model artifacts and metrics

## Required dataset columns

- `timestamp`
- `price`
- `high`
- `low`
- `volume`

## Commands

```bash
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

Prediction and evaluation:

```bash
cd ModelMK1
/bin/python3 main.py predict
/bin/python3 main.py backtest
```
