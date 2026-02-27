"""
Configuration for the Market Data Service.
Reads Alpaca API keys from .env file in the project root.
"""
import os
from pathlib import Path

# Load .env from project root if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass

# Alpaca API credentials (free account — no broker dependency)
ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")

# Cache TTL in seconds (default 5 minutes — matches Yahoo Finance delay)
CACHE_TTL_SECONDS: int = int(os.getenv("MARKET_DATA_CACHE_TTL", "300"))
