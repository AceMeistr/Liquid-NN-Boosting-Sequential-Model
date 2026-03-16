# Sensex MFT v3.0 Upstox Pipeline

Implementation scaffold generated from `Sensex_MFT_v3_Upstox.pdf` requirements.

## 1. Setup

```powershell
# from workspace root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` from `.env.example`:

```env
UPSTOX_API_KEY=...
UPSTOX_API_SECRET=...
```

Important:
- Access token is runtime prompted via `getpass`.
- Access token is never written to disk.

## 2. Required Data Files

Expected locations:
- `data/instruments/bse_instruments.csv`
- `data/instruments/sensex_weights.csv`
- `data/rfr/rbi_tbill_91d.csv`
- `data/regimes/expiry_regimes.csv`
- `data/corporate_actions/ca_table.csv`

## 3. CLI Commands

```powershell
python pipeline.py ingest --start YYYY-MM-DD --end YYYY-MM-DD
python pipeline.py preprocess --start YYYY-MM-DD --end YYYY-MM-DD
python pipeline.py features --start YYYY-MM-DD --end YYYY-MM-DD
python pipeline.py normalize --start YYYY-MM-DD --end YYYY-MM-DD
python pipeline.py full-run --start YYYY-MM-DD --end YYYY-MM-DD
python pipeline.py validate --stage raw
python pipeline.py validate --stage clean
python pipeline.py validate --stage features
python pipeline.py validate --stage normalized
python pipeline.py refresh-instruments
```

## 4. Tests

```powershell
python -m pytest -q
```

Current local status: `13 passed`.
