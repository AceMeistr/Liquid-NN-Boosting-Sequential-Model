from __future__ import annotations

import asyncio
import gc
import io
from datetime import date, timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
import pyarrow as pa
import pyarrow.parquet as pq
import upstox_client
from loguru import logger

from auth import get_upstox_client, handle_401_retry
from config import PipelineConfig
from logging_utils import configure_logging


class IngestionEngine:
    """Upstox-powered raw data ingestion engine."""

    def __init__(self, config: PipelineConfig):
        """Initialize ingestion engine and load static inputs.

        Args:
            config: Pipeline configuration instance.
        """
        self.cfg = config
        configure_logging("ingestion_engine", self.cfg.LOG_ROOT)
        self.cfg.assert_upstox_credentials()

        self.api_client = get_upstox_client()
        self.history_api = upstox_client.HistoryApi(self.api_client)
        self.option_api = self._build_option_api()
        self.instruments_api = self._build_instruments_api()
        self.instruments = pd.read_csv(self.cfg.INSTRUMENT_MASTER_PATH)
        self.weights = pd.read_csv(self.cfg.WEIGHTS_PATH)
        self._build_instrument_lookup()

    def _build_option_api(self) -> Any:
        """Return available options API implementation for installed SDK version."""
        option_chain_cls = getattr(upstox_client, "OptionChainApi", None)
        if option_chain_cls is not None:
            return option_chain_cls(self.api_client)

        options_cls = getattr(upstox_client, "OptionsApi", None)
        if options_cls is not None:
            return options_cls(self.api_client)
        raise RuntimeError("Upstox SDK does not expose an options chain API class.")

    def _build_instruments_api(self) -> Any | None:
        """Return instruments API if exposed by SDK, else None."""
        instruments_cls = getattr(upstox_client, "InstrumentsApi", None)
        if instruments_cls is not None:
            return instruments_cls(self.api_client)
        return None

    def _build_instrument_lookup(self) -> None:
        """Build in-memory (symbol, segment) -> instrument_key lookup."""
        df = self.instruments.copy()
        cols = {c.lower(): c for c in df.columns}
        symbol_col = cols.get("trading_symbol") or cols.get("tradingsymbol") or cols.get("symbol")
        seg_col = cols.get("segment") or cols.get("exchange_segment")
        key_col = cols.get("instrument_key") or cols.get("instrumentkey")

        self._instrument_lookup: dict[tuple[str, str], str] = {}
        if symbol_col is None or key_col is None:
            return

        for _, row in df.iterrows():
            symbol = str(row[symbol_col]).upper()
            segment = str(row[seg_col]).upper() if seg_col is not None else ""
            key = str(row[key_col])
            self._instrument_lookup[(symbol, segment)] = key

    def _call_get_instruments(self) -> Any:
        if self.instruments_api is None:
            return None

        for method_name in ("get_instruments", "get_instrument_master", "get_instruments_master"):
            fn = getattr(self.instruments_api, method_name, None)
            if fn is None:
                continue
            try:
                return handle_401_retry(fn, exchange="BSE", api_version="2.0")
            except TypeError:
                try:
                    return handle_401_retry(fn, exchange="BSE")
                except TypeError:
                    return handle_401_retry(fn)

        return None

    def _call_intra_day(self, instrument_key: str) -> Any:
        try:
            return handle_401_retry(
                self.history_api.get_intra_day_candle_data,
                instrument_key=instrument_key,
                interval="1minute",
                api_version="2.0",
            )
        except TypeError:
            return handle_401_retry(
                self.history_api.get_intra_day_candle_data,
                instrument_key,
                "1minute",
                "2.0",
            )

    def _call_option_chain(self, expiry_str: str) -> Any:
        if hasattr(self.option_api, "get_option_chain"):
            try:
                return handle_401_retry(
                    self.option_api.get_option_chain,
                    instrument_key="BSE_INDEX|SENSEX",
                    expiry_date=expiry_str,
                    api_version="2.0",
                )
            except TypeError:
                return handle_401_retry(
                    self.option_api.get_option_chain,
                    "BSE_INDEX|SENSEX",
                    expiry_str,
                    "2.0",
                )

        if hasattr(self.option_api, "get_put_call_option_chain"):
            try:
                return handle_401_retry(
                    self.option_api.get_put_call_option_chain,
                    instrument_key="BSE_INDEX|SENSEX",
                    expiry_date=expiry_str,
                )
            except TypeError:
                return handle_401_retry(
                    self.option_api.get_put_call_option_chain,
                    "BSE_INDEX|SENSEX",
                    expiry_str,
                )

        raise RuntimeError("No compatible option-chain method available in current Upstox SDK.")

    @staticmethod
    def _as_dict_list(payload: Any) -> list[dict[str, Any]]:
        """Normalize API payload into a list of dictionaries."""
        if payload is None:
            return []

        if isinstance(payload, list):
            rows = payload
        elif isinstance(payload, dict):
            rows = payload.get("data", payload.get("instruments", []))
            if isinstance(rows, dict):
                rows = [rows]
        elif isinstance(payload, (str, bytes)):
            text = payload.decode("utf-8") if isinstance(payload, bytes) else payload
            if "," in text:
                records = pd.read_csv(io.StringIO(text)).to_dict(orient="records")
                return [{str(k): v for k, v in rec.items()} for rec in records]
            return []
        else:
            data = getattr(payload, "data", None)
            if data is not None:
                rows = data
            else:
                rows = [payload]

        normalized: list[dict[str, Any]] = []
        for row in rows:
            if row is None:
                continue
            if isinstance(row, dict):
                normalized.append(row)
            elif hasattr(row, "to_dict"):
                normalized.append(row.to_dict())
            elif hasattr(row, "__dict__"):
                normalized.append(dict(row.__dict__))
        return normalized

    @staticmethod
    def _extract_candles(response: Any) -> list[list[Any]]:
        """Extract candle rows from Upstox history response."""
        candidates = [
            response,
            getattr(response, "data", None),
            getattr(getattr(response, "data", None), "candles", None),
        ]
        for candidate in candidates:
            if candidate is None:
                continue
            if isinstance(candidate, dict):
                candles = candidate.get("candles") or candidate.get("data", {}).get("candles")
                if candles:
                    return candles
            if isinstance(candidate, list):
                if candidate and isinstance(candidate[0], list):
                    return candidate
        return []

    def refresh_instrument_master(self) -> None:
        """Refresh and persist BSE instrument master from Upstox."""
        response = self._call_get_instruments()
        if response is None:
            logger.warning(
                "Current Upstox SDK does not expose an instruments endpoint. "
                "Retaining existing local instrument master at {}",
                self.cfg.INSTRUMENT_MASTER_PATH,
            )
            return

        if isinstance(response, (str, bytes)):
            text = response.decode("utf-8") if isinstance(response, bytes) else response
            df = pd.read_csv(io.StringIO(text))
        else:
            records = self._as_dict_list(response)
            if not records:
                raise ValueError("Upstox instrument master response is empty.")
            df = pd.DataFrame.from_records(records)

        df.to_csv(self.cfg.INSTRUMENT_MASTER_PATH, index=False)
        self.instruments = df
        self._build_instrument_lookup()
        logger.info("Saved {} instruments to {}", len(df), self.cfg.INSTRUMENT_MASTER_PATH)

    def get_instrument_key(self, symbol: str, segment: str = "BSE_EQ") -> str:
        """Resolve Upstox instrument key for symbol and segment.

        Args:
            symbol: Trading symbol.
            segment: Exchange segment, defaults to BSE_EQ.

        Returns:
            Instrument key string.

        Raises:
            KeyError: If symbol is not found in loaded master.
        """
        key = self._instrument_lookup.get((symbol.upper(), segment.upper()))
        if key:
            return key

        if segment.upper() == "BSE_EQ":
            fallback = f"BSE_EQ|{symbol.upper()}"
            logger.warning("Falling back to constructed key {} for missing symbol {}", fallback, symbol)
            return fallback

        raise KeyError(f"Symbol '{symbol}' not found for segment '{segment}'.")

    async def fetch_ohlcv_1min(self, instrument_key: str, trading_date: date) -> pd.DataFrame:
        """Fetch one-minute OHLCV and constrain to one trading session.

        Args:
            instrument_key: Upstox instrument key.
            trading_date: Session date in local exchange timezone.

        Returns:
            DataFrame indexed by Asia/Kolkata timestamps.
        """
        response = await asyncio.to_thread(self._call_intra_day, instrument_key)
        candles = self._extract_candles(response)

        if not candles:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows: list[list[Any]] = []
        for candle in candles:
            if isinstance(candle, dict):
                rows.append(
                    [
                        candle.get("timestamp") or candle.get("time"),
                        candle.get("open"),
                        candle.get("high"),
                        candle.get("low"),
                        candle.get("close"),
                        candle.get("volume"),
                    ]
                )
            elif isinstance(candle, (list, tuple)) and len(candle) >= 6:
                rows.append(list(candle[:6]))

        if not rows:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])

        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        ts = ts.dt.tz_convert("Asia/Kolkata")
        df["timestamp"] = ts
        df = df.dropna(subset=["timestamp"]).set_index("timestamp").sort_index()

        df = df.astype(
            {
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "int32",
            }
        )

        local_dates = pd.DatetimeIndex(df.index).date
        df = df.loc[local_dates == trading_date]
        if not df.empty:
            df = df.between_time(self.cfg.BSE_SESSION_START, self.cfg.BSE_SESSION_END)

        return df

    async def fetch_all_constituents(self, trading_date: date) -> dict[str, pd.DataFrame]:
        """Fetch one-minute OHLCV for all Sensex constituents concurrently.

        Args:
            trading_date: Session date.

        Returns:
            Symbol-to-dataframe mapping.
        """
        semaphore = asyncio.Semaphore(8)
        symbols = sorted(set(self.weights["symbol"].astype(str).tolist()))

        async def _fetch(symbol: str) -> tuple[str, pd.DataFrame]:
            async with semaphore:
                try:
                    key = self.get_instrument_key(symbol, segment="BSE_EQ")
                    data = await self.fetch_ohlcv_1min(key, trading_date)
                    return symbol.lower(), data
                except Exception as exc:
                    logger.warning("Constituent fetch failed for {}: {}", symbol, exc)
                    return symbol.lower(), pd.DataFrame()

        results = await asyncio.gather(*[_fetch(sym) for sym in symbols])
        return dict(results)

    def _get_regime_row(self, trading_date: date) -> pd.Series | None:
        regime_df = pd.read_csv(
            self.cfg.REGIME_TABLE_PATH,
            parse_dates=["date_from", "date_to"],
        )
        current = pd.Timestamp(trading_date)
        mask = (regime_df["date_from"] <= current) & (regime_df["date_to"] >= current)
        if mask.any():
            return regime_df.loc[mask].iloc[0]
        return None

    def _next_expiry(self, trading_date: date) -> date:
        regime_row = self._get_regime_row(trading_date)
        expiry_weekday = int(regime_row["expiry_weekday"]) if regime_row is not None else 3

        cal = mcal.get_calendar("BSE")
        schedule = cal.schedule(
            start_date=trading_date,
            end_date=trading_date + timedelta(days=45),
        )
        sessions = pd.DatetimeIndex(schedule.index).tz_localize(None)
        candidates = [d.date() for d in sessions if d.weekday() == expiry_weekday and d.date() >= trading_date]
        if not candidates:
            raise ValueError(f"No expiry session found for trading_date={trading_date}")
        return candidates[0]

    @staticmethod
    def _extract_option_rows(response: Any) -> list[dict[str, Any]]:
        """Extract flattened option rows from option-chain response."""
        rows: list[dict[str, Any]] = []

        def _as_dict(obj: Any) -> dict[str, Any]:
            if isinstance(obj, dict):
                return obj
            if hasattr(obj, "to_dict"):
                return obj.to_dict()
            if hasattr(obj, "__dict__"):
                return dict(obj.__dict__)
            return {}

        def _opt_field(opt_data: dict[str, Any], *keys: str, default: Any = np.nan) -> Any:
            for key in keys:
                if key in opt_data and opt_data[key] is not None:
                    return opt_data[key]

            market_data = _as_dict(opt_data.get("market_data"))
            greeks = _as_dict(opt_data.get("option_greeks"))
            for key in keys:
                if key in market_data and market_data[key] is not None:
                    return market_data[key]
                if key in greeks and greeks[key] is not None:
                    return greeks[key]
            return default

        payload = response
        if hasattr(response, "data"):
            payload = response.data
        if isinstance(payload, dict):
            payload = payload.get("data", payload)
        if payload is None:
            return rows

        records: list[Any]
        if isinstance(payload, list):
            records = payload
        else:
            records = [payload]

        for record in records:
            rec = _as_dict(record)
            if not isinstance(rec, dict):
                continue

            strike = rec.get("strike_price") or rec.get("strike") or rec.get("strikePrice")
            if strike is None:
                continue

            # Typical response shape with call_options and put_options.
            if "call_options" in rec or "put_options" in rec:
                for side_key, opt_type in (("call_options", "C"), ("put_options", "P")):
                    opt = _as_dict(rec.get(side_key) or {})
                    if not opt:
                        continue
                    rows.append(
                        {
                            "strike": strike,
                            "option_type": opt_type,
                            "iv": _opt_field(opt, "iv", "implied_volatility"),
                            "delta": _opt_field(opt, "delta"),
                            "gamma": _opt_field(opt, "gamma"),
                            "theta": _opt_field(opt, "theta"),
                            "vega": _opt_field(opt, "vega"),
                            "last_price": _opt_field(opt, "last_price", "ltp"),
                            "volume": _opt_field(opt, "volume", default=0),
                            "oi": _opt_field(opt, "oi", "open_interest", default=0),
                        }
                    )
                continue

            rows.append(
                {
                    "strike": strike,
                    "option_type": rec.get("option_type") or rec.get("type") or "",
                    "iv": rec.get("iv") or rec.get("implied_volatility") or np.nan,
                    "delta": rec.get("delta", np.nan),
                    "gamma": rec.get("gamma", np.nan),
                    "theta": rec.get("theta", np.nan),
                    "vega": rec.get("vega", np.nan),
                    "last_price": rec.get("last_price") or rec.get("ltp") or np.nan,
                    "volume": rec.get("volume", 0),
                    "oi": rec.get("oi") or rec.get("open_interest") or 0,
                }
            )
        return rows

    def fetch_option_chain(self, trading_date: date, spot_price: float) -> pd.DataFrame:
        """Fetch near-week option chain and keep nearest 10 OTM calls/puts.

        Args:
            trading_date: Session date.
            spot_price: Spot index price used for OTM selection.

        Returns:
            DataFrame indexed by strike with filtered option rows.
        """
        expiry = self._next_expiry(trading_date)
        response = self._call_option_chain(expiry.strftime("%Y-%m-%d"))
        rows = self._extract_option_rows(response)
        del response

        if not rows:
            return pd.DataFrame(
                columns=[
                    "option_type",
                    "iv",
                    "delta",
                    "gamma",
                    "theta",
                    "vega",
                    "last_price",
                    "volume",
                    "oi",
                ]
            )

        opt_df = pd.DataFrame.from_records(rows)
        opt_df["strike"] = pd.to_numeric(opt_df["strike"], errors="coerce")
        opt_df = opt_df.dropna(subset=["strike"])
        if not np.isfinite(spot_price):
            spot_price = float(opt_df["strike"].median())
        opt_df["distance"] = (opt_df["strike"] - float(spot_price)).abs()

        calls = opt_df[(opt_df["strike"] > spot_price) & (opt_df["option_type"].str.upper() == "C")]
        puts = opt_df[(opt_df["strike"] < spot_price) & (opt_df["option_type"].str.upper() == "P")]

        calls = calls.nsmallest(10, columns="distance")
        puts = puts.nsmallest(10, columns="distance")
        selected = pd.concat([calls, puts], ignore_index=True)
        if selected.empty:
            selected = opt_df.nsmallest(20, columns="distance").copy()

        num_cols = ["iv", "delta", "gamma", "theta", "vega", "last_price"]
        for col in num_cols:
            selected[col] = pd.to_numeric(selected[col], errors="coerce").astype("float32")

        selected["volume"] = pd.to_numeric(selected["volume"], errors="coerce").fillna(0).astype("int32")
        selected["oi"] = pd.to_numeric(selected["oi"], errors="coerce").fillna(0).astype("int32")

        selected = selected.drop(columns=["distance"]).set_index("strike").sort_index()
        gc.collect()
        return selected

    def build_master_clock(self, trading_date: date) -> pd.DatetimeIndex:
        """Build one-minute session clock for a trading day."""
        master_clock = pd.date_range(
            start=f"{trading_date} {self.cfg.BSE_SESSION_START}:00",
            end=f"{trading_date} {self.cfg.BSE_SESSION_END}:00",
            freq="1min",
            tz="Asia/Kolkata",
        )
        if len(master_clock) != self.cfg.BARS_PER_SESSION:
            raise ValueError(
                f"Expected {self.cfg.BARS_PER_SESSION} bars, got {len(master_clock)} for {trading_date}."
            )
        return master_clock

    @staticmethod
    def _compute_stale_flags(volume: pd.Series) -> pd.Series:
        zeros = volume.fillna(0).eq(0)
        groups = (~zeros).cumsum()
        run_len = zeros.groupby(groups).cumcount() + 1
        return zeros & run_len.gt(5)

    def align_to_master_clock(
        self,
        df: pd.DataFrame,
        master_clock: pd.DatetimeIndex,
        symbol: str,
    ) -> pd.DataFrame:
        """Align instrument frame to master clock with bounded fill rules.

        Args:
            df: Source OHLCV frame.
            master_clock: Session time index.
            symbol: Symbol prefix used in output columns.

        Returns:
            375-row aligned dataframe.
        """
        clock_df = pd.DataFrame(index=master_clock)
        symbol = symbol.lower()

        if df.empty:
            merged = clock_df.assign(open=np.nan, high=np.nan, low=np.nan, close=np.nan, volume=0)
        else:
            rhs = df[["open", "high", "low", "close", "volume"]].sort_index().copy()
            left = clock_df.reset_index().rename(columns={"index": "timestamp"})
            right = rhs.reset_index().rename(columns={rhs.index.name or "index": "timestamp"})

            merged = pd.merge_asof(
                left.sort_values("timestamp"),
                right.sort_values("timestamp"),
                on="timestamp",
                direction="backward",
                tolerance=pd.Timedelta("60s"),
            ).set_index("timestamp")

        merged["close"] = merged["close"].ffill(limit=5)
        merged["open"] = merged["open"].fillna(merged["close"])
        merged["high"] = merged["high"].fillna(merged["close"])
        merged["low"] = merged["low"].fillna(merged["close"])
        merged["volume"] = merged["volume"].fillna(0)
        merged["is_stale"] = self._compute_stale_flags(merged["volume"]) 

        merged = merged.astype(
            {
                "open": "float32",
                "high": "float32",
                "low": "float32",
                "close": "float32",
                "volume": "int32",
                "is_stale": "bool",
            }
        )

        merged = merged.rename(columns={col: f"{symbol}_{col}" for col in merged.columns})
        if len(merged) != self.cfg.BARS_PER_SESSION:
            raise ValueError(f"Aligned frame for {symbol} has {len(merged)} rows, expected 375.")
        return merged

    def _derive_option_features(self, option_df: pd.DataFrame, spot_price: float) -> dict[str, np.float32]:
        """Compute lightweight derived option features from selected strikes."""
        fields = {
            "synth_atm_iv": np.float32(np.nan),
            "iv_skew_25d": np.float32(np.nan),
            "iv_butterfly": np.float32(np.nan),
            "opt_delta_net": np.float32(np.nan),
            "opt_gamma_net": np.float32(np.nan),
            "opt_vega_net": np.float32(np.nan),
            "gex": np.float32(np.nan),
            "total_pcr_vol": np.float32(np.nan),
            "total_pcr_oi": np.float32(np.nan),
            "oi_imbalance": np.float32(np.nan),
            "oi_flow": np.float32(np.nan),
        }

        if option_df.empty or "option_type" not in option_df.columns:
            return fields

        calls = option_df[option_df["option_type"].str.upper() == "C"].copy()
        puts = option_df[option_df["option_type"].str.upper() == "P"].copy()

        if not option_df.index.empty and np.isfinite(spot_price):
            strikes = pd.to_numeric(pd.Series(option_df.index), errors="coerce").to_numpy(dtype=np.float64)
            valid_mask = np.isfinite(strikes)
            if valid_mask.any():
                valid_positions = np.flatnonzero(valid_mask)
                ranked = valid_positions[np.argsort(np.abs(strikes[valid_mask] - spot_price))][:2]
                nearest = option_df.iloc[ranked]
                fields["synth_atm_iv"] = np.float32(np.nanmean(nearest["iv"].astype("float32")))

        call_iv = calls["iv"].astype("float32").dropna()
        put_iv = puts["iv"].astype("float32").dropna()
        if not call_iv.empty and not put_iv.empty and not np.isnan(fields["synth_atm_iv"]):
            skew = float(put_iv.mean() - call_iv.mean())
            bfly = float((put_iv.mean() + call_iv.mean()) / 2.0 - fields["synth_atm_iv"])
            fields["iv_skew_25d"] = np.float32(skew)
            fields["iv_butterfly"] = np.float32(bfly)

        signed_delta = option_df["delta"].astype("float32") * np.where(
            option_df["option_type"].str.upper().eq("C"),
            1.0,
            -1.0,
        )
        fields["opt_delta_net"] = np.float32(np.nansum(signed_delta))
        fields["opt_gamma_net"] = np.float32(np.nansum(option_df["gamma"].astype("float32")))
        fields["opt_vega_net"] = np.float32(np.nansum(option_df["vega"].astype("float32")))
        fields["gex"] = np.float32(
            np.nansum(option_df["gamma"].astype("float32") * option_df["oi"].astype("float32"))
            * self.cfg.SENSEX_CONTRACT_SIZE
        )

        call_vol = float(calls["volume"].sum()) if not calls.empty else 0.0
        put_vol = float(puts["volume"].sum()) if not puts.empty else 0.0
        call_oi = float(calls["oi"].sum()) if not calls.empty else 0.0
        put_oi = float(puts["oi"].sum()) if not puts.empty else 0.0

        fields["total_pcr_vol"] = np.float32(put_vol / (call_vol + 1.0))
        fields["total_pcr_oi"] = np.float32(put_oi / (call_oi + 1.0))
        fields["oi_imbalance"] = np.float32((call_oi - put_oi) / (call_oi + put_oi + 1.0))
        fields["oi_flow"] = np.float32((call_vol - put_vol) / (abs(call_vol) + abs(put_vol) + 1.0))

        return fields

    @staticmethod
    def _persist_with_metadata(df: pd.DataFrame, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = df.reset_index().rename(columns={"index": "timestamp"})
        table = pa.Table.from_pandas(payload, preserve_index=False)

        metadata = dict(table.schema.metadata or {})
        metadata[b"schema_version"] = b"3.0"
        table = table.replace_schema_metadata(metadata)

        pq.write_table(table, output_path, compression="snappy")

    def run_day(self, trading_date: date) -> None:
        """Run complete ingestion workflow for one trading day.

        Args:
            trading_date: Trading date to ingest.
        """
        start_ts = perf_counter()

        master_clock = self.build_master_clock(trading_date)
        idx_df = asyncio.run(self.fetch_ohlcv_1min("BSE_INDEX|SENSEX", trading_date))
        idx_aligned = self.align_to_master_clock(idx_df, master_clock, "idx")

        constituents = asyncio.run(self.fetch_all_constituents(trading_date))
        aligned_parts = [idx_aligned]
        for symbol, frame in constituents.items():
            aligned_parts.append(self.align_to_master_clock(frame, master_clock, symbol))

        master_df = pd.concat(aligned_parts, axis=1)

        spot_series = master_df["idx_close"].dropna()
        spot = float(spot_series.iloc[-1]) if not spot_series.empty else np.nan
        option_df = self.fetch_option_chain(trading_date, spot)
        opt_features = self._derive_option_features(option_df, spot)
        for key, value in opt_features.items():
            master_df[key] = np.float32(value)

        master_df["session_date"] = pd.DatetimeIndex(pd.DatetimeIndex(master_df.index).date)
        master_df["bar_index_in_session"] = np.arange(len(master_df), dtype="int16")

        if len(master_df) != self.cfg.BARS_PER_SESSION:
            raise AssertionError(f"Expected 375 rows, got {len(master_df)}")

        output_path = (
            self.cfg.raw_root
            / f"{trading_date.year}"
            / f"{trading_date.month:02d}"
            / f"master_{trading_date.isoformat()}.parquet"
        )
        self._persist_with_metadata(master_df, output_path)

        stale_cols = [c for c in master_df.columns if c.endswith("_is_stale")]
        stale_count = int(master_df[stale_cols].sum().sum()) if stale_cols else 0
        null_count = int(master_df.isna().sum().sum())
        elapsed = perf_counter() - start_ts

        logger.info(
            "Ingestion complete for {} rows={} nulls={} stale={} elapsed={:.2f}s",
            trading_date,
            len(master_df),
            null_count,
            stale_count,
            elapsed,
        )

        del idx_df, idx_aligned, constituents, aligned_parts, option_df, master_df
        gc.collect()
