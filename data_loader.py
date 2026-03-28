"""
Data Loader Module
==================
Fetches historical price data for NIFTY sector ETFs using yfinance,
computes daily log returns, and constructs the weighted portfolio return series.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS, TICKER_NAMES, WEIGHTS,
    START_DATE, END_DATE
)


def fetch_price_data(
    tickers: list = TICKERS,
    start: str = START_DATE,
    end: str = END_DATE
) -> pd.DataFrame:
    """
    Download adjusted closing prices for all tickers.

    Parameters
    ----------
    tickers : list
        List of Yahoo Finance ticker symbols.
    start : str
        Start date in 'YYYY-MM-DD' format.
    end : str
        End date in 'YYYY-MM-DD' format.

    Returns
    -------
    pd.DataFrame
        DataFrame of adjusted closing prices with tickers as columns.
    """
    print(f"\n{'='*60}")
    print("  FETCHING HISTORICAL PRICE DATA")
    print(f"{'='*60}")
    print(f"  Period : {start} → {end}")
    print(f"  Tickers: {len(tickers)} NIFTY sector ETFs")
    print(f"{'='*60}\n")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=True
    )

    # Handle various yfinance column structures
    if isinstance(data.columns, pd.MultiIndex):
        # Try to extract Close prices
        if "Close" in data.columns.get_level_values(0):
            prices = data["Close"].copy()
        else:
            # Some yfinance versions put ticker at level 0
            prices = pd.DataFrame()
            for t in tickers:
                if t in data.columns.get_level_values(0):
                    prices[t] = data[t]["Close"]
                elif t in data.columns.get_level_values(1):
                    prices[t] = data["Close"][t]
    else:
        # Single ticker case
        prices = data[["Close"]].copy()
        prices.columns = tickers

    # Forward-fill small gaps, then drop remaining NaN rows
    prices = prices.ffill().dropna()

    # Rename columns to friendly names
    rename_map = {t: TICKER_NAMES.get(t, t) for t in prices.columns if t in TICKER_NAMES}
    prices = prices.rename(columns=rename_map)

    print(f"\n  ✓ Downloaded {len(prices)} trading days of data")
    print(f"  ✓ Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  ✓ ETFs: {list(prices.columns)}\n")

    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute daily logarithmic returns from price data.

    Parameters
    ----------
    prices : pd.DataFrame
        Adjusted closing prices.

    Returns
    -------
    pd.DataFrame
        Daily log returns (same columns as prices).
    """
    log_returns = np.log(prices / prices.shift(1)).dropna()
    return log_returns


def compute_portfolio_returns(
    returns: pd.DataFrame,
    weights: list = WEIGHTS
) -> pd.Series:
    """
    Compute weighted portfolio returns.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log returns for individual ETFs.
    weights : list
        Portfolio weights (must sum to 1.0).

    Returns
    -------
    pd.Series
        Daily portfolio returns.
    """
    w = np.array(weights)
    assert abs(w.sum() - 1.0) < 1e-6, f"Weights must sum to 1.0, got {w.sum():.4f}"

    portfolio_returns = returns.dot(w)
    portfolio_returns.name = "Portfolio"

    return portfolio_returns


def get_portfolio_data() -> tuple:
    """
    Main function to load all data needed for VaR analysis.

    Returns
    -------
    tuple : (prices, returns, portfolio_returns)
        - prices: pd.DataFrame of adjusted close prices
        - returns: pd.DataFrame of individual ETF log returns
        - portfolio_returns: pd.Series of weighted portfolio log returns
    """
    prices = fetch_price_data()
    returns = compute_returns(prices)
    portfolio_returns = compute_portfolio_returns(returns)

    # Print summary statistics
    print(f"{'='*60}")
    print("  PORTFOLIO RETURN STATISTICS")
    print(f"{'='*60}")
    print(f"  Mean daily return   : {portfolio_returns.mean():.6f} ({portfolio_returns.mean()*252:.4f} annualized)")
    print(f"  Std daily return    : {portfolio_returns.std():.6f} ({portfolio_returns.std()*np.sqrt(252):.4f} annualized)")
    print(f"  Skewness            : {portfolio_returns.skew():.4f}")
    print(f"  Kurtosis            : {portfolio_returns.kurtosis():.4f}")
    print(f"  Min daily return    : {portfolio_returns.min():.6f}")
    print(f"  Max daily return    : {portfolio_returns.max():.6f}")
    print(f"  Total observations  : {len(portfolio_returns)}")
    print(f"{'='*60}\n")

    return prices, returns, portfolio_returns


if __name__ == "__main__":
    prices, returns, portfolio_returns = get_portfolio_data()
    print(prices.tail())
    print(returns.tail())
    print(portfolio_returns.tail())
