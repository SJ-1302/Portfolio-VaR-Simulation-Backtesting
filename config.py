"""
Configuration file for Portfolio VaR Simulation & Backtesting
=============================================================
Contains all configurable parameters: tickers, weights, date ranges,
confidence levels, simulation parameters, and backtesting settings.
"""

# ──────────────────────────────────────────────────────────────
# Portfolio Configuration
# ──────────────────────────────────────────────────────────────

# NIFTY Sector ETFs — diversified across sectors
TICKERS = [
    "NIFTYBEES.NS",    # Nifty 50 Index
    "BANKBEES.NS",     # Banking Sector
    "ITBEES.NS",       # IT Sector
    "CPSEETF.NS",      # CPSE / PSU / Energy Sector
    "JUNIORBEES.NS",   # Nifty Next 50
    "GOLDBEES.NS",     # Gold
]

# Friendly names for display
TICKER_NAMES = {
    "NIFTYBEES.NS":   "Nifty 50",
    "BANKBEES.NS":    "Bank Nifty",
    "ITBEES.NS":      "IT Sector",
    "CPSEETF.NS":     "CPSE / PSU",
    "JUNIORBEES.NS":  "Nifty Next 50",
    "GOLDBEES.NS":    "Gold",
}

# Portfolio weights (must sum to 1.0)
WEIGHTS = [0.20, 0.20, 0.15, 0.15, 0.15, 0.15]

# ──────────────────────────────────────────────────────────────
# Data Configuration
# ──────────────────────────────────────────────────────────────

START_DATE = "2020-01-01"
END_DATE = "2024-07-31"

# ──────────────────────────────────────────────────────────────
# VaR Parameters
# ──────────────────────────────────────────────────────────────

# Confidence levels for VaR estimation
CONFIDENCE_LEVELS = [0.95, 0.99]

# Time horizon (in trading days); 1 = daily VaR
TIME_HORIZON = 1

# Initial portfolio value (in INR)
PORTFOLIO_VALUE = 10_000_000  # ₹1 Crore

# ──────────────────────────────────────────────────────────────
# Monte Carlo Simulation Parameters
# ──────────────────────────────────────────────────────────────

NUM_SIMULATIONS = 10_000       # Number of simulated paths
SIMULATION_DAYS = 1            # Forecast horizon per simulation

# Random seed for reproducibility
RANDOM_SEED = 42

# ──────────────────────────────────────────────────────────────
# Backtesting Parameters
# ──────────────────────────────────────────────────────────────

# Lookback windows (in trading days) for rolling VaR
LOOKBACK_WINDOWS = [60, 120, 250, 500]

# Default lookback window
DEFAULT_LOOKBACK = 250

# Kupiec's test significance level
KUPIEC_SIGNIFICANCE = 0.05

# Out-of-sample split ratio (for backtesting period)
BACKTEST_SPLIT = 0.3  # Last 30% of data used for backtesting

# ──────────────────────────────────────────────────────────────
# Visualization Settings
# ──────────────────────────────────────────────────────────────

FIGURE_DPI = 150
FIGURE_STYLE = "seaborn-v0_8-darkgrid"
COLOR_PALETTE = {
    "primary":    "#2563EB",   # Blue
    "secondary":  "#7C3AED",   # Purple
    "success":    "#059669",   # Green
    "danger":     "#DC2626",   # Red
    "warning":    "#D97706",   # Amber
    "info":       "#0891B2",   # Cyan
    "dark":       "#1F2937",   # Dark gray
    "light":      "#F3F4F6",   # Light gray
    "var_95":     "#F59E0B",   # Gold for 95% VaR
    "var_99":     "#EF4444",   # Red for 99% VaR
}

# Output directory for saved plots
OUTPUT_DIR = "output"
