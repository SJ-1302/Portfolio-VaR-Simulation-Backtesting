"""
Portfolio VaR Simulation & Backtesting — Main Script
====================================================
Orchestrates the full pipeline:
  1. Load historical NIFTY sector ETF data
  2. Compute VaR using Historical, Parametric, and Monte Carlo methods
  3. Backtest VaR forecasts using Kupiec's POF Test
  4. Calibrate lookback window lengths
  5. Generate professional visualizations
"""

import sys
import warnings
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — plots save to files
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from config import (
    WEIGHTS, CONFIDENCE_LEVELS, PORTFOLIO_VALUE,
    DEFAULT_LOOKBACK, LOOKBACK_WINDOWS
)
from data_loader import get_portfolio_data
from var_models import compute_all_var
from backtesting import run_backtest, calibrate_windows
from visualization import generate_all_plots


def print_banner():
    """Print project banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════════╗
    ║                                                              ║
    ║     PORTFOLIO VaR SIMULATION & BACKTESTING                   ║
    ║     ─────────────────────────────────────────                 ║
    ║     Quantitative Risk Management Framework                   ║
    ║     NIFTY Sector ETF Portfolio                                ║
    ║                                                              ║
    ║     Methods: Historical | Parametric | Monte Carlo            ║
    ║     Backtesting: Kupiec's POF Test                            ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_final_summary(var_results, backtest_results, calibration_df):
    """Print a comprehensive final summary."""
    print(f"\n{'═'*60}")
    print("  FINAL SUMMARY")
    print(f"{'═'*60}")

    # VaR Summary Table
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  VALUE-AT-RISK ESTIMATES (Portfolio: ₹{PORTFOLIO_VALUE:,.0f})  │")
    print(f"  ├─────────────────────────────────────────────────────┤")

    for cl in CONFIDENCE_LEVELS:
        print(f"  │                                                     │")
        print(f"  │  {cl:.0%} Confidence Level:                             │")

        for method_key, method_name in [
            ("historical", "Historical"),
            ("parametric", "Parametric"),
            ("monte_carlo", "Monte Carlo")
        ]:
            var_abs = var_results[method_key][cl]["var_absolute"]
            var_pct = abs(var_results[method_key][cl]["var_percentage"]) * 100
            print(f"  │    {method_name:<15} VaR = ₹{var_abs:>10,.0f}  ({var_pct:.2f}%)   │")

    print(f"  │                                                     │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Backtesting Summary
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  KUPIEC'S POF BACKTESTING RESULTS                   │")
    print(f"  ├─────────────────────────────────────────────────────┤")

    for cl in CONFIDENCE_LEVELS:
        kupiec = backtest_results[cl]["kupiec_test"]
        status = "✓ ADEQUATE" if not kupiec["reject_h0"] else "✗ INADEQUATE"
        print(f"  │  {cl:.0%} CL: {status:<15} (p={kupiec['p_value']:.4f})         │")
        print(f"  │    Expected: {kupiec['expected_failures']:.1f} failures | "
              f"Observed: {kupiec['observed_failures']} failures     │")

    print(f"  │                                                     │")
    print(f"  └─────────────────────────────────────────────────────┘")

    # Window Calibration Summary
    print(f"\n  ┌─────────────────────────────────────────────────────┐")
    print(f"  │  WINDOW CALIBRATION SUMMARY                         │")
    print(f"  ├─────────────────────────────────────────────────────┤")

    for _, row in calibration_df.iterrows():
        status = "✓" if row["Model Adequate"] else "✗"
        print(f"  │  Window={int(row['Window']):>4}d  |  p-value={row['p-value']:.4f}  | {status}     │")

    best_window = calibration_df.loc[calibration_df["p-value"].idxmax()]
    print(f"  │                                                     │")
    print(f"  │  → Best window: {int(best_window['Window'])} days (p={best_window['p-value']:.4f})            │")
    print(f"  └─────────────────────────────────────────────────────┘")

    print(f"\n{'═'*60}")
    print("  Analysis Complete! Check the 'output/' folder for plots.")
    print(f"{'═'*60}\n")


def main():
    """Main execution pipeline."""
    print_banner()

    # ─── Step 1: Load Data ───
    print("\n[1/5] Loading portfolio data...")
    prices, returns, portfolio_returns = get_portfolio_data()
    weights = np.array(WEIGHTS)

    # ─── Step 2: Compute VaR ───
    print("\n[2/5] Computing Value-at-Risk...")
    var_results = compute_all_var(
        portfolio_returns, returns, weights,
        CONFIDENCE_LEVELS, PORTFOLIO_VALUE
    )

    # ─── Step 3: Backtest VaR ───
    print("\n[3/5] Backtesting VaR forecasts...")
    backtest_results = run_backtest(
        portfolio_returns,
        confidence_levels=CONFIDENCE_LEVELS,
        lookback=DEFAULT_LOOKBACK,
        method="historical"
    )

    # ─── Step 4: Calibrate Windows ───
    print("\n[4/5] Calibrating lookback windows...")
    calibration_df = calibrate_windows(
        portfolio_returns,
        confidence_level=0.95,
        windows=LOOKBACK_WINDOWS,
        method="historical"
    )

    # ─── Step 5: Generate Visualizations ───
    print("\n[5/5] Generating visualizations...")
    simulated_returns = var_results["monte_carlo"]["simulated_returns"]

    generate_all_plots(
        prices, returns, portfolio_returns,
        var_results, backtest_results, calibration_df,
        simulated_returns
    )

    # ─── Final Summary ───
    print_final_summary(var_results, backtest_results, calibration_df)


if __name__ == "__main__":
    main()
