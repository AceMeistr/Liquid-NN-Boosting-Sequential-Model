from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineConfig(BaseSettings):
    """Central settings object for the Sensex MFT pipeline."""

    # Keep env-backed defaults to avoid hard failures during non-network operations
    # (e.g. tests, static checks) while still enforcing non-empty values before API use.
    UPSTOX_API_KEY: str = ""
    UPSTOX_API_SECRET: str = ""

    DATA_ROOT: Path = Path("./data")
    LOG_ROOT: Path = Path("./logs")

    ROLLING_WINDOW_Z: int = 60
    ROLLING_WINDOW_RV: int = 5
    ROLLING_WINDOW_ATR: int = 14
    IV_MIN_VALID_STRIKES: int = 6
    GAMMA_MIN_T_SECONDS: int = 300
    EMBARGO_MINUTES: int = 30
    ZSCORE_CLIP: float = 3.5
    EPSILON: float = 1e-8

    BSE_SESSION_START: str = "09:15"
    BSE_SESSION_END: str = "15:29"
    BARS_PER_SESSION: int = 375

    ANNUALIZATION: float = 94500.0
    SENSEX_CONTRACT_SIZE: int = 50

    RFR_PATH: Path = Path("./data/rfr/rbi_tbill_91d.csv")
    WEIGHTS_PATH: Path = Path("./data/instruments/sensex_weights.csv")
    REGIME_TABLE_PATH: Path = Path("./data/regimes/expiry_regimes.csv")
    CA_TABLE_PATH: Path = Path("./data/corporate_actions/ca_table.csv")
    INSTRUMENT_MASTER_PATH: Path = Path("./data/instruments/bse_instruments.csv")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def raw_root(self) -> Path:
        """Return raw parquet output root."""
        return self.DATA_ROOT / "raw"

    @property
    def clean_root(self) -> Path:
        """Return clean parquet output root."""
        return self.DATA_ROOT / "clean"

    @property
    def feature_root(self) -> Path:
        """Return feature parquet output root."""
        return self.DATA_ROOT / "features"

    @property
    def normalized_root(self) -> Path:
        """Return normalized parquet output root."""
        return self.DATA_ROOT / "normalized"

    @property
    def scaler_root(self) -> Path:
        """Return scaler persistence root."""
        return self.DATA_ROOT / "scalers"

    def ensure_directories(self) -> None:
        """Create required directory structure if missing."""
        paths = [
            self.DATA_ROOT,
            self.LOG_ROOT,
            self.raw_root,
            self.clean_root,
            self.feature_root,
            self.normalized_root,
            self.scaler_root,
            self.RFR_PATH.parent,
            self.WEIGHTS_PATH.parent,
            self.REGIME_TABLE_PATH.parent,
            self.CA_TABLE_PATH.parent,
            self.INSTRUMENT_MASTER_PATH.parent,
        ]
        for path in paths:
            path.mkdir(parents=True, exist_ok=True)

    def assert_upstox_credentials(self) -> None:
        """Validate that API key and secret are present before network operations."""
        if not self.UPSTOX_API_KEY or not self.UPSTOX_API_SECRET:
            raise ValueError(
                "Missing UPSTOX_API_KEY/UPSTOX_API_SECRET. "
                "Set them in .env before running ingestion or auth-required commands."
            )


@lru_cache(maxsize=1)
def get_config() -> PipelineConfig:
    """Build and cache pipeline configuration from environment and defaults."""
    cfg = PipelineConfig()
    cfg.ensure_directories()
    return cfg
