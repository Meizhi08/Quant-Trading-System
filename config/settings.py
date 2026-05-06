"""Central configuration — all secrets come from .env, never hardcoded."""

from pydantic_settings import BaseSettings
from pydantic import Field
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent


class Settings(BaseSettings):
    # ── Claude AI ────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", env="ANTHROPIC_API_KEY")
    claude_model: str = "claude-sonnet-4-6"
    claude_max_tokens: int = 2048

    # ── Questrade API ─────────────────────────────────────────────────────────
    questrade_refresh_token: str = Field("", env="QUESTRADE_REFRESH_TOKEN")
    questrade_account_id: str = Field("", env="QUESTRADE_ACCOUNT_ID")

    # ── Risk defaults ─────────────────────────────────────────────────────────
    max_daily_loss_pct: float = 0.02
    max_position_pct: float = 0.30
    kelly_fraction: float = 0.5

    # ── Data cache ────────────────────────────────────────────────────────────
    cache_dir: Path = BASE_DIR / "cache"
    cache_ttl_minutes: int = 60          # longer TTL — yfinance is rate-limited

    # ── Backtest defaults ─────────────────────────────────────────────────────
    initial_cash: float = 10_000.0       # CAD
    # Questrade: $0.01/share, min $4.95, max $9.95 per side
    commission_per_share: float = 0.01
    commission_min: float = 4.95
    commission_max: float = 9.95
    stamp_tax: float = 0.0               # no stamp tax in Canada/US
    slippage: float = 0.001

    # ── Market ────────────────────────────────────────────────────────────────
    market_index: str = "^GSPC"          # S&P 500; use "^GSPTSE" for TSX
    default_currency: str = "CAD"

    class Config:
        env_file = BASE_DIR / ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"  # ignore any extra env vars


settings = Settings()
