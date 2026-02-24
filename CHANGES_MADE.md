# Changes Made Summary

Date: 2026-02-24
Branch: RED-STAR

## Environment Cleanup
- Removed active virtual-environment variables from the shell session:
  - `VIRTUAL_ENV`
  - `CONDA_PREFIX`
  - `CONDA_DEFAULT_ENV`
  - `PYENV_VERSION`
  - `PIPENV_ACTIVE`
  - `POETRY_ACTIVE`
  - `PYTHONPATH`

## Runtime and CLI Hardening
### `ModelMK1/src/modelmk1/common/runtime.py`
- Added centralized logging setup via `configure_logging()`.
- Added retry helper `retry(...)` for transient operation failures.
- Added `safe_torch_load(...)` with `weights_only=True` fallback compatibility.

### `ModelMK1/main.py`
- Initializes centralized logging at startup.
- Added top-level command execution exception boundary with logged failures.
- Updated build call to schema-derived input size (`run_build(None, ...)`).

## Path and Data Robustness
### `ModelMK1/src/modelmk1/common/paths.py`
- Added robust dataset candidate scanning with retry on stat calls.
- Added warning logs for file race conditions during dataset discovery.

### `ModelMK1/src/modelmk1/data/loader.py`
- Added numeric coercion for price/high/low/volume.
- Added row-level validation filters:
  - positive prices/high/low
  - non-negative volume
  - `high >= low`
- Added logging for dropped invalid rows.
- Added argument validation for `seq_len` and `horizon`.
- Added guard rails for insufficient rows and empty feature set.

## Feature Engineering Improvements
### `ModelMK1/src/modelmk1/features/indicators.py`
- Added module logger.
- Added `get_feature_schema(include_stoch_rsi=...)` for explicit, reusable feature contracts.
- Added indicator row-drop logging after NaN/Inf cleanup.

## Model Build and Training Updates
### `ModelMK1/src/modelmk1/train/build_models.py`
- Refactored `run_build(...)` to accept optional `input_size`.
- Derives input size from `get_feature_schema(...)` when unset.
- CLI `--input-size` default changed to `None`.

### `ModelMK1/src/modelmk1/train/train_pipeline.py`
- Build stage now uses dynamic input size (`input_size=None`).

### `ModelMK1/src/modelmk1/train/train_lnn.py`
- Added module logger for structured epoch logging.
- Replaced epoch `print(...)` with `logger.info(...)`.
- Added CUDA cache cleanup per epoch (`torch.cuda.empty_cache()`).

### `ModelMK1/src/modelmk1/train/train_hybrid.py`
- Uses `safe_torch_load(...)` for checkpoint loading.
- XGBoost training now conditionally uses GPU (`use_gpu=(device.type == "cuda")`).

### `ModelMK1/src/modelmk1/models/xgb_model.py`
- Added `use_gpu` option in `train_xgb(...)`.
- Enables CUDA device mode for XGBoost when requested.

## Prediction Safety Updates
### `ModelMK1/src/modelmk1/predict/predict.py`
- Uses `safe_torch_load(...)` for model checkpoint loading.
- Added `Path` handling for manifest checkpoint path.

## Testing and Dependencies
### `requirements.txt`
- Added test dependency: `pytest>=8.0`.

### Added tests
- `ModelMK1/tests/test_indicators.py`
  - Validates feature schema consistency with and without Stoch RSI.
- `ModelMK1/tests/test_loader.py`
  - Validates loader output structure.
  - Validates supervised-data argument error handling.
- `ModelMK1/tests/test_runtime.py`
  - Validates retry behavior under transient failure.
  - Validates safe torch checkpoint loading roundtrip.

## Analysis Artifact
### `CODEBASE_ANALYSIS_REPORT.md`
- Added detailed architecture, robustness, deployment, CPU/GPU, and optimization audit report.

## Notes
- This summary reflects tracked code/document changes already present in the repository working tree at the time of generation.
- The environment-variable cleanup applies to the active shell session where commands were run.
