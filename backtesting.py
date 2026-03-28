"""
Backtesting Module
==================
Implements VaR backtesting using:
  - Kupiec's Proportion of Failures (POF) Test
  - Rolling window VaR estimation
  - Window length calibration
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    CONFIDENCE_LEVELS, LOOKBACK_WINDOWS,
    DEFAULT_LOOKBACK, KUPIEC_SIGNIFICANCE, BACKTEST_SPLIT
)


def kupiec_pof_test(
    failures: int,
    total_obs: int,
    confidence_level: float,
    significance: float = KUPIEC_SIGNIFICANCE
) -> dict:
    """
    Kupiec's Proportion of Failures (POF) Test.

    Tests whether the observed number of VaR exceedances is consistent
    with the expected number under the null hypothesis.

    H₀: The observed failure rate equals the expected failure rate (1 - CL)
    H₁: The observed failure rate ≠ the expected failure rate

    Test Statistic:
        LR_POF = -2 * ln[(1-p)^(T-x) * p^x] + 2 * ln[(1-x/T)^(T-x) * (x/T)^x]

    where:
        p = expected failure rate (1 - confidence level)
        x = number of observed failures (exceedances)
        T = total number of observations

    Under H₀, LR_POF ~ χ²(1)

    Parameters
    ----------
    failures : int
        Number of observed VaR exceedances.
    total_obs : int
        Total number of backtesting observations.
    confidence_level : float
        VaR confidence level (e.g., 0.95 or 0.99).
    significance : float
        Significance level for the hypothesis test (default 0.05).

    Returns
    -------
    dict
        Test results including LR statistic, p-value, and decision.
    """
    p_expected = 1 - confidence_level  # Expected failure rate
    p_observed = failures / total_obs if total_obs > 0 else 0

    # Handle edge cases
    if failures == 0:
        # If no failures observed, use a small epsilon
        log_lr = -2 * (total_obs * np.log(1 - p_expected)) + \
                 -2 * (total_obs * np.log(1))
        # Simplified: LR ≈ 2 * T * log((1)/(1-p))
        lr_stat = 2 * total_obs * np.log(1 / (1 - p_expected))
    elif failures == total_obs:
        lr_stat = 2 * total_obs * np.log(1 / p_expected)
    else:
        # Full Kupiec LR formula
        log_unrestricted = (total_obs - failures) * np.log(1 - p_observed) + \
                           failures * np.log(p_observed)
        log_restricted = (total_obs - failures) * np.log(1 - p_expected) + \
                         failures * np.log(p_expected)
        lr_stat = -2 * (log_restricted - log_unrestricted)

    # p-value from chi-squared distribution with 1 degree of freedom
    p_value = 1 - stats.chi2.cdf(lr_stat, df=1)

    # Decision
    reject_h0 = p_value < significance
    decision = "REJECT H₀ (Model Inadequate)" if reject_h0 else "FAIL TO REJECT H₀ (Model Adequate)"

    return {
        "confidence_level": confidence_level,
        "total_observations": total_obs,
        "expected_failures": p_expected * total_obs,
        "observed_failures": failures,
        "expected_failure_rate": p_expected,
        "observed_failure_rate": p_observed,
        "lr_statistic": lr_stat,
        "p_value": p_value,
        "significance": significance,
        "reject_h0": reject_h0,
        "decision": decision,
    }


def rolling_var_backtest(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    lookback: int = DEFAULT_LOOKBACK,
    method: str = "historical"
) -> pd.DataFrame:
    """
    Perform rolling VaR estimation and identify exceedances.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Full portfolio return series.
    confidence_level : float
        Confidence level for VaR estimation.
    lookback : int
        Number of past observations to use for VaR estimation.
    method : str
        VaR estimation method: 'historical' or 'parametric'.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: 'Return', 'VaR', 'Exceedance'.
    """
    n = len(portfolio_returns)
    alpha = 1 - confidence_level

    var_series = pd.Series(index=portfolio_returns.index, dtype=float)
    exceedance_series = pd.Series(index=portfolio_returns.index, dtype=bool)

    for i in range(lookback, n):
        window = portfolio_returns.iloc[i - lookback:i]

        if method == "historical":
            var_value = np.percentile(window, alpha * 100)
        elif method == "parametric":
            mu = window.mean()
            sigma = window.std()
            z = stats.norm.ppf(alpha)
            var_value = mu + z * sigma
        else:
            raise ValueError(f"Unknown method: {method}")

        var_series.iloc[i] = var_value
        exceedance_series.iloc[i] = portfolio_returns.iloc[i] < var_value

    # Trim to backtesting period
    backtest_df = pd.DataFrame({
        "Return": portfolio_returns.iloc[lookback:],
        "VaR": var_series.iloc[lookback:],
        "Exceedance": exceedance_series.iloc[lookback:],
    })

    return backtest_df


def run_backtest(
    portfolio_returns: pd.Series,
    confidence_levels: list = CONFIDENCE_LEVELS,
    lookback: int = DEFAULT_LOOKBACK,
    method: str = "historical"
) -> dict:
    """
    Run full backtesting analysis for given confidence levels.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns.
    confidence_levels : list
        Confidence levels to test.
    lookback : int
        Lookback window for rolling VaR.
    method : str
        VaR method ('historical' or 'parametric').

    Returns
    -------
    dict
        Backtesting results including rolling VaR data and Kupiec test.
    """
    print(f"\n{'='*60}")
    print("  BACKTESTING VaR FORECASTS")
    print(f"{'='*60}")
    print(f"  Method       : {method.title()}")
    print(f"  Lookback     : {lookback} trading days")
    print(f"  Total data   : {len(portfolio_returns)} observations")
    print(f"  Backtest obs : {len(portfolio_returns) - lookback} observations")
    print(f"{'='*60}\n")

    results = {}

    for cl in confidence_levels:
        # Rolling VaR backtest
        backtest_df = rolling_var_backtest(
            portfolio_returns, cl, lookback, method
        )

        # Count exceedances
        failures = backtest_df["Exceedance"].sum()
        total_obs = len(backtest_df)

        # Kupiec's POF test
        kupiec = kupiec_pof_test(failures, total_obs, cl)

        print(f"  ── {cl:.0%} Confidence Level ──")
        print(f"  Expected failures : {kupiec['expected_failures']:.1f} "
              f"({kupiec['expected_failure_rate']:.2%})")
        print(f"  Observed failures : {kupiec['observed_failures']} "
              f"({kupiec['observed_failure_rate']:.2%})")
        print(f"  LR Statistic      : {kupiec['lr_statistic']:.4f}")
        print(f"  p-value           : {kupiec['p_value']:.4f}")
        print(f"  Decision          : {kupiec['decision']}")
        print()

        results[cl] = {
            "backtest_data": backtest_df,
            "kupiec_test": kupiec,
            "lookback": lookback,
            "method": method,
        }

    return results


def calibrate_windows(
    portfolio_returns: pd.Series,
    confidence_level: float = 0.95,
    windows: list = LOOKBACK_WINDOWS,
    method: str = "historical"
) -> pd.DataFrame:
    """
    Calibrate lookback window length by testing multiple windows.

    For each window, computes the Kupiec test and reports accuracy.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Portfolio returns.
    confidence_level : float
        Confidence level for VaR.
    windows : list
        List of lookback window sizes to test.
    method : str
        VaR method.

    Returns
    -------
    pd.DataFrame
        Summary table comparing window performance.
    """
    print(f"\n{'='*60}")
    print("  WINDOW LENGTH CALIBRATION")
    print(f"{'='*60}")
    print(f"  Confidence Level : {confidence_level:.0%}")
    print(f"  Windows tested   : {windows}")
    print(f"{'='*60}\n")

    calibration_results = []

    for window in windows:
        if window >= len(portfolio_returns):
            print(f"  ⚠ Window {window} exceeds data length, skipping.")
            continue

        backtest_df = rolling_var_backtest(
            portfolio_returns, confidence_level, window, method
        )

        failures = backtest_df["Exceedance"].sum()
        total_obs = len(backtest_df)
        kupiec = kupiec_pof_test(failures, total_obs, confidence_level)

        calibration_results.append({
            "Window": window,
            "Backtest Obs": total_obs,
            "Expected Failures": kupiec["expected_failures"],
            "Observed Failures": kupiec["observed_failures"],
            "Expected Rate": kupiec["expected_failure_rate"],
            "Observed Rate": kupiec["observed_failure_rate"],
            "LR Statistic": kupiec["lr_statistic"],
            "p-value": kupiec["p_value"],
            "Model Adequate": not kupiec["reject_h0"],
        })

    cal_df = pd.DataFrame(calibration_results)

    print(cal_df.to_string(index=False))
    print()

    return cal_df


if __name__ == "__main__":
    from data_loader import get_portfolio_data

    prices, returns, portfolio_returns = get_portfolio_data()

    # Run backtest
    results = run_backtest(portfolio_returns)

    # Calibrate windows
    cal_df = calibrate_windows(portfolio_returns, confidence_level=0.95)
