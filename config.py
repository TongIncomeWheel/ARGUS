"""
Configuration for Income Wheel App
"""
import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
BACKUP_DIR = DATA_DIR / "backups"
LOGS_DIR = BASE_DIR / "logs"

# Excel file location (Income Wheel data file)
EXCEL_PATH = str(DATA_DIR / "income_wheel_data.xlsx")

# Tickers to track (configurable)
TICKERS = ["MARA", "SPY", "CRCL", "ETHA", "SOL"]

# Trading parameters
WEEKLY_TARGET_PCT = 0.25  # 25% of capital deployed per week
TRADING_DAYS_PER_WEEK = 5

# IBKR TWS settings
IBKR_HOST = "127.0.0.1"
IBKR_PORT = 7497  # 7497 for live, 7496 for paper
IBKR_CLIENT_ID = 100

# Risk thresholds
CALL_RISK_HIGH_DTE = 7  # Days to expiry to flag high call risk
CALL_RISK_MEDIUM_BUFFER = 0.98  # Price within 2% of strike = medium risk
EXPIRING_SOON_DTE = 14  # Flag positions expiring within 14 days (2 weeks)

# Capital calculation
MARGIN_REQUIREMENT_PCT = 0.20  # 20% margin requirement for CSP

# Backup settings
BACKUP_RETENTION_DAYS = 7

# Logging
LOG_LEVEL = "INFO"
