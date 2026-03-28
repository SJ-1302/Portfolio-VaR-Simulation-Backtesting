"""
VaR Models Module
=================
Implements three Value-at-Risk estimation methods:
  1. Historical Simulation
  2. Parametric (Variance-Covariance) Method
  3. Monte Carlo Simulation

All methods support 95% and 99% confidence levels.
"""

import numpy as np
import pandas as pd
from scipy import stats

from config import (
    CONFIDENCE_LEVELS, PORTFOLIO_VALUE,
    NUM_SIMULATIONS, RANDOM_SEED
)


# ──────────────────────────────────────────────────────────────
# 1. Historical Simulation VaR
# ──────────────────────────────────────────────────────────────

def historical_var(
    returns: pd.Series,
    confidence_levels: list = CONFIDENCE_LEVELS,
    portfolio_value: float = PORTFOLIO_VALUE
) -> dict:
    """
    Calculate VaR using Historical Simulation method.

    This method uses the actual empirical distribution of portfolio returns
    to estimate VaR. No distributional assumptions are made.

    Parameters
    ----------
    returns : pd.Series
        Historical portfolio returns.
    confidence_levels : list
        List of confidence levels (e.g., [0.95, 0.99]).
    portfolio_value : float
        Current portfolio value in currency units.

    Returns
    -------
    dict
        VaR results for each confidence level.
    """
    results = {}

    for cl in confidence_levels:
        alpha = 1 - cl
        var_pct = np.percentile(returns, alpha * 100)
        var_abs = abs(var_pct) * portfolio_value

        results[cl] = {
            "method": "Historical Simulation",
            "confidence_level": cl,
            "var_percentage": var_pct,
            "var_absolute": var_abs,
            "alpha": alpha,
        }

    return results


# ──────────────────────────────────────────────────────────────
# 2. Parametric (Variance-Covariance) VaR
# ──────────────────────────────────────────────────────────────

def parametric_var(
    returns: pd.Series,
    confidence_levels: list = CONFIDENCE_LEVELS,
    portfolio_value: float = PORTFOLIO_VALUE
) -> dict:
    """
    Calculate VaR using the Parametric (Variance-Covariance) method.

    Assumes returns follow a normal distribution. VaR is computed as:
        VaR = -(μ + z_α × σ)

    where z_α is the z-score corresponding to the confidence level.

    Parameters
    ----------
    returns : pd.Series
        Historical portfolio returns.
    confidence_levels : list
        List of confidence levels.
    portfolio_value : float
        Current portfolio value.

    Returns
    -------
    dict
        VaR results for each confidence level.
    """
    mu = returns.mean()
    sigma = returns.std()

    results = {}

    for cl in confidence_levels:
        alpha = 1 - cl
        z_score = stats.norm.ppf(alpha)
        var_pct = mu + z_score * sigma
        var_abs = abs(var_pct) * portfolio_value

        results[cl] = {
            "method": "Parametric (Variance-Covariance)",
            "confidence_level": cl,
            "var_percentage": var_pct,
            "var_absolute": var_abs,
            "alpha": alpha,
            "z_score": z_score,
            "mean": mu,
            "std": sigma,
        }

    return results


# ──────────────────────────────────────────────────────────────
# 3. Monte Carlo Simulation VaR
# ──────────────────────────────────────────────────────────────

def monte_carlo_var(
    returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: list = CONFIDENCE_LEVELS,
    portfolio_value: float = PORTFOLIO_VALUE,
    num_simulations: int = NUM_SIMULATIONS,
    seed: int = RANDOM_SEED
) -> dict:
    """
    Calculate VaR using Monte Carlo Simulation.

    Uses Cholesky decomposition to generate correlated random returns
    based on the historical covariance structure of the portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical returns of individual assets.
    weights : np.ndarray
        Portfolio weights.
    confidence_levels : list
        List of confidence levels.
    portfolio_value : float
        Current portfolio value.
    num_simulations : int
        Number of Monte Carlo simulations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict
        VaR results including simulated return distribution.
    """
    np.random.seed(seed)

    # Compute mean returns and covariance matrix
    mean_returns = returns.mean().values
    cov_matrix = returns.cov().values

    # Cholesky decomposition for correlated random draws
    L = np.linalg.cholesky(cov_matrix)

    # Generate random standard normal draws
    Z = np.random.standard_normal((num_simulations, len(weights)))

    # Correlated random returns
    simulated_returns = Z @ L.T + mean_returns

    # Portfolio simulated returns (weighted sum)
    portfolio_sim_returns = simulated_returns @ weights

    results = {}
    for cl in confidence_levels:
        alpha = 1 - cl
        var_pct = np.percentile(portfolio_sim_returns, alpha * 100)
        var_abs = abs(var_pct) * portfolio_value

        results[cl] = {
            "method": "Monte Carlo Simulation",
            "confidence_level": cl,
            "var_percentage": var_pct,
            "var_absolute": var_abs,
            "alpha": alpha,
            "num_simulations": num_simulations,
        }

    # Store the simulated returns for visualization
    results["simulated_returns"] = portfolio_sim_returns

    return results


# ──────────────────────────────────────────────────────────────
# Combined VaR Summary
# ──────────────────────────────────────────────────────────────

def compute_all_var(
    portfolio_returns: pd.Series,
    individual_returns: pd.DataFrame,
    weights: np.ndarray,
    confidence_levels: list = CONFIDENCE_LEVELS,
    portfolio_value: float = PORTFOLIO_VALUE
) -> dict:
    """
    Compute VaR using all three methods and return a consolidated summary.

    Parameters
    ----------
    portfolio_returns : pd.Series
        Weighted portfolio returns.
    individual_returns : pd.DataFrame
        Individual asset returns.
    weights : np.ndarray
        Portfolio weights.
    confidence_levels : list
        List of confidence levels.
    portfolio_value : float
        Portfolio value.

    Returns
    -------
    dict
        Dictionary with all VaR results.
    """
    print(f"\n{'='*60}")
    print("  VALUE-AT-RISK ESTIMATION")
    print(f"{'='*60}")
    print(f"  Portfolio Value: ₹{portfolio_value:,.0f}")
    print(f"  Confidence Levels: {confidence_levels}")
    print(f"{'='*60}\n")

    # 1. Historical VaR
    hist_var = historical_var(portfolio_returns, confidence_levels, portfolio_value)

    # 2. Parametric VaR
    param_var = parametric_var(portfolio_returns, confidence_levels, portfolio_value)

    # 3. Monte Carlo VaR
    mc_var = monte_carlo_var(individual_returns, weights, confidence_levels, portfolio_value)

    # Print summary table
    print(f"  {'Method':<35} {'CL':>5} {'VaR %':>10} {'VaR (₹)':>15}")
    print(f"  {'─'*35} {'─'*5} {'─'*10} {'─'*15}")

    for cl in confidence_levels:
        for var_result in [hist_var[cl], param_var[cl], mc_var[cl]]:
            print(
                f"  {var_result['method']:<35} "
                f"{var_result['confidence_level']:>5.0%} "
                f"{var_result['var_percentage']:>10.4%} "
                f"₹{var_result['var_absolute']:>13,.0f}"
            )
        print()

    return {
        "historical": hist_var,
        "parametric": param_var,
        "monte_carlo": mc_var,
    }


if __name__ == "__main__":
    from data_loader import get_portfolio_data
    from config import WEIGHTS

    prices, returns, portfolio_returns = get_portfolio_data()
    weights = np.array(WEIGHTS)
    all_var = compute_all_var(portfolio_returns, returns, weights)
