"""
Visualization Module
====================
Publication-quality charts for portfolio analysis, VaR estimation,
Monte Carlo simulation, and backtesting results.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

from config import (
    CONFIDENCE_LEVELS, PORTFOLIO_VALUE, COLOR_PALETTE,
    FIGURE_DPI, OUTPUT_DIR, TICKER_NAMES, WEIGHTS
)


def setup_style():
    """Set up matplotlib style for professional plots."""
    plt.rcParams.update({
        "figure.facecolor": "#0F172A",
        "axes.facecolor": "#1E293B",
        "axes.edgecolor": "#334155",
        "axes.labelcolor": "#E2E8F0",
        "text.color": "#E2E8F0",
        "xtick.color": "#94A3B8",
        "ytick.color": "#94A3B8",
        "grid.color": "#334155",
        "grid.alpha": 0.5,
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "legend.facecolor": "#1E293B",
        "legend.edgecolor": "#334155",
        "legend.fontsize": 9,
    })


def ensure_output_dir():
    """Create output directory if it doesn't exist."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_portfolio_composition(prices: pd.DataFrame, weights: list = WEIGHTS):
    """
    Plot 1: Portfolio composition (pie chart) and cumulative returns.
    """
    setup_style()
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Pie Chart: Portfolio Weights ──
    ax1 = axes[0]
    colors = ["#3B82F6", "#8B5CF6", "#06B6D4", "#10B981", "#F59E0B", "#EF4444"]
    explode = [0.03] * len(weights)

    wedges, texts, autotexts = ax1.pie(
        weights,
        labels=prices.columns,
        autopct="%1.0f%%",
        colors=colors,
        explode=explode,
        startangle=140,
        pctdistance=0.80,
        wedgeprops=dict(edgecolor="#0F172A", linewidth=2),
    )
    for text in texts:
        text.set_fontsize(9)
        text.set_color("#E2E8F0")
    for autotext in autotexts:
        autotext.set_fontsize(9)
        autotext.set_fontweight("bold")
        autotext.set_color("#0F172A")
    ax1.set_title("Portfolio Allocation", fontsize=14, fontweight="bold", color="#E2E8F0")

    # ── Line Chart: Cumulative Returns ──
    ax2 = axes[1]
    cum_returns = (prices / prices.iloc[0] - 1) * 100

    for i, col in enumerate(cum_returns.columns):
        ax2.plot(cum_returns.index, cum_returns[col],
                 color=colors[i], linewidth=1.5, alpha=0.85, label=col)

    # Portfolio cumulative return
    portfolio_prices = (prices / prices.iloc[0]) @ np.array(weights)
    portfolio_cum = (portfolio_prices - 1) * 100
    ax2.plot(cum_returns.index, portfolio_cum,
             color="#FFFFFF", linewidth=2.5, label="Portfolio", linestyle="--")

    ax2.set_title("Cumulative Returns (%)", fontsize=14, fontweight="bold")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Return (%)")
    ax2.legend(loc="upper left", framealpha=0.9)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "01_portfolio_composition.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_return_distribution(
    portfolio_returns: pd.Series,
    var_results: dict
):
    """
    Plot 2: Return distribution histogram with VaR thresholds.
    """
    setup_style()
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(14, 7))

    # Histogram
    n, bins, patches = ax.hist(
        portfolio_returns * 100, bins=80, density=True,
        color="#3B82F6", alpha=0.6, edgecolor="#1E293B", linewidth=0.5
    )

    # Fit normal distribution
    mu = portfolio_returns.mean() * 100
    sigma = portfolio_returns.std() * 100
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 200)
    pdf = stats.norm.pdf(x, mu, sigma)
    ax.plot(x, pdf, color="#F8FAFC", linewidth=2, label="Normal Fit", linestyle="--")

    # KDE
    portfolio_returns_pct = portfolio_returns * 100
    kde_x = np.linspace(portfolio_returns_pct.min(), portfolio_returns_pct.max(), 200)
    kde = stats.gaussian_kde(portfolio_returns_pct)
    ax.plot(kde_x, kde(kde_x), color="#A78BFA", linewidth=2, label="KDE (Empirical)")

    # VaR lines
    var_colors = {0.95: "#F59E0B", 0.99: "#EF4444"}
    var_styles = {0.95: "--", 0.99: "-."}

    for cl in CONFIDENCE_LEVELS:
        hist_var = var_results["historical"][cl]["var_percentage"] * 100
        ax.axvline(
            hist_var, color=var_colors[cl], linewidth=2.5,
            linestyle=var_styles[cl],
            label=f"VaR {cl:.0%} = {hist_var:.2f}%"
        )

        # Shade the tail
        ax.fill_between(
            kde_x, kde(kde_x), where=(kde_x <= hist_var),
            color=var_colors[cl], alpha=0.25
        )

    ax.set_title("Portfolio Return Distribution with VaR Thresholds", fontsize=16, fontweight="bold")
    ax.set_xlabel("Daily Return (%)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=10)
    ax.grid(True, alpha=0.3)

    # Stats annotation
    stats_text = (
        f"Mean: {mu:.4f}%\n"
        f"Std: {sigma:.4f}%\n"
        f"Skew: {portfolio_returns.skew():.3f}\n"
        f"Kurt: {portfolio_returns.kurtosis():.3f}"
    )
    ax.text(0.02, 0.97, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#0F172A", edgecolor="#475569", alpha=0.9))

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "02_return_distribution.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_monte_carlo(
    simulated_returns: np.ndarray,
    var_results: dict,
    portfolio_value: float = PORTFOLIO_VALUE
):
    """
    Plot 3: Monte Carlo simulation results.
    """
    setup_style()
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    sim_pnl = simulated_returns * portfolio_value

    # ── Left: Histogram of simulated P&L ──
    ax1 = axes[0]
    ax1.hist(sim_pnl, bins=100, density=True, color="#8B5CF6", alpha=0.6,
             edgecolor="#1E293B", linewidth=0.3)

    var_colors = {0.95: "#F59E0B", 0.99: "#EF4444"}
    for cl in CONFIDENCE_LEVELS:
        var_abs = -var_results["monte_carlo"][cl]["var_absolute"]
        ax1.axvline(var_abs, color=var_colors[cl], linewidth=2.5, linestyle="--",
                    label=f"VaR {cl:.0%} = ₹{abs(var_abs):,.0f}")

    ax1.set_title("Monte Carlo Simulated P&L Distribution", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Portfolio P&L (₹)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.legend(loc="upper left", framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # ── Right: Simulated return paths (sample) ──
    ax2 = axes[1]
    n_show = min(500, len(simulated_returns))
    sample_returns = simulated_returns[:n_show]

    # Sort by return for color mapping
    sorted_idx = np.argsort(sample_returns)
    colors = plt.cm.RdYlGn(np.linspace(0, 1, n_show))

    for i, idx in enumerate(sorted_idx):
        ax2.barh(i, sample_returns[idx] * 100, color=colors[i], alpha=0.7, height=1.0)

    # VaR lines
    for cl in CONFIDENCE_LEVELS:
        var_val = var_results["monte_carlo"][cl]["var_percentage"] * 100
        ax2.axvline(var_val, color=var_colors[cl], linewidth=2, linestyle="--",
                    label=f"VaR {cl:.0%}")

    ax2.set_title(f"Simulated Returns (Sample of {n_show})", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Return (%)", fontsize=11)
    ax2.set_ylabel("Simulation #", fontsize=11)
    ax2.legend(loc="lower right", framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "03_monte_carlo.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_var_comparison(var_results: dict, portfolio_value: float = PORTFOLIO_VALUE):
    """
    Plot 4: Bar chart comparing VaR across all three methods.
    """
    setup_style()
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    methods = ["historical", "parametric", "monte_carlo"]
    method_labels = ["Historical\nSimulation", "Parametric\n(Var-Cov)", "Monte Carlo\nSimulation"]
    method_colors = ["#3B82F6", "#8B5CF6", "#06B6D4"]

    for idx, cl in enumerate(CONFIDENCE_LEVELS):
        ax = axes[idx]

        var_values = [
            var_results[m][cl]["var_absolute"] for m in methods
        ]
        var_pcts = [
            abs(var_results[m][cl]["var_percentage"]) * 100 for m in methods
        ]

        bars = ax.bar(
            method_labels, var_values,
            color=method_colors, edgecolor="#0F172A", linewidth=1.5,
            alpha=0.85, width=0.5
        )

        # Add value labels on bars
        for bar, val, pct in zip(bars, var_values, var_pcts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + portfolio_value * 0.001,
                f"₹{val:,.0f}\n({pct:.2f}%)",
                ha="center", va="bottom", fontsize=10, fontweight="bold",
                color="#E2E8F0"
            )

        ax.set_title(f"VaR Comparison — {cl:.0%} Confidence", fontsize=14, fontweight="bold")
        ax.set_ylabel("VaR (₹)", fontsize=11)
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_ylim(0, max(var_values) * 1.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "04_var_comparison.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_backtest_results(backtest_results: dict):
    """
    Plot 5: Backtesting — rolling VaR with exceedances highlighted.
    """
    setup_style()
    ensure_output_dir()

    n_plots = len(backtest_results)
    fig, axes = plt.subplots(n_plots, 1, figsize=(16, 6 * n_plots))
    if n_plots == 1:
        axes = [axes]

    var_colors = {0.95: "#F59E0B", 0.99: "#EF4444"}

    for ax, cl in zip(axes, sorted(backtest_results.keys())):
        bt = backtest_results[cl]
        data = bt["backtest_data"]
        kupiec = bt["kupiec_test"]

        # Plot returns
        ax.plot(data.index, data["Return"] * 100,
                color="#64748B", linewidth=0.5, alpha=0.7, label="Portfolio Return")

        # Plot VaR threshold
        ax.plot(data.index, data["VaR"] * 100,
                color=var_colors[cl], linewidth=1.5, alpha=0.9,
                label=f"VaR {cl:.0%}")

        # Highlight exceedances
        exceedances = data[data["Exceedance"]]
        ax.scatter(
            exceedances.index, exceedances["Return"] * 100,
            color="#EF4444", s=25, zorder=5, alpha=0.8,
            label=f"Exceedances ({len(exceedances)})",
            edgecolors="#0F172A", linewidths=0.5
        )

        # Kupiec test annotation
        status_color = "#10B981" if not kupiec["reject_h0"] else "#EF4444"
        status_icon = "✓" if not kupiec["reject_h0"] else "✗"

        annotation = (
            f"{status_icon} Kupiec POF Test\n"
            f"LR = {kupiec['lr_statistic']:.3f}, p = {kupiec['p_value']:.4f}\n"
            f"Expected: {kupiec['expected_failures']:.1f} | Observed: {kupiec['observed_failures']}\n"
            f"{'Model Adequate' if not kupiec['reject_h0'] else 'Model INADEQUATE'}"
        )
        ax.text(
            0.98, 0.97, annotation, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", facecolor=status_color, alpha=0.2,
                      edgecolor=status_color)
        )

        ax.set_title(f"VaR Backtesting — {cl:.0%} Confidence Level ({bt['method'].title()})",
                     fontsize=14, fontweight="bold")
        ax.set_xlabel("Date", fontsize=11)
        ax.set_ylabel("Return (%)", fontsize=11)
        ax.legend(loc="lower left", framealpha=0.9)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "05_backtest_results.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_window_calibration(calibration_df: pd.DataFrame, confidence_level: float = 0.95):
    """
    Plot 6: Window calibration results.
    """
    setup_style()
    ensure_output_dir()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # ── Left: Failure rates comparison ──
    ax1 = axes[0]
    x = np.arange(len(calibration_df))
    width = 0.3

    bars1 = ax1.bar(x - width/2, calibration_df["Expected Rate"] * 100,
                    width, label="Expected Rate", color="#3B82F6", alpha=0.8)
    bars2 = ax1.bar(x + width/2, calibration_df["Observed Rate"] * 100,
                    width, label="Observed Rate", color="#F59E0B", alpha=0.8)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{w}d" for w in calibration_df["Window"]])
    ax1.set_xlabel("Lookback Window", fontsize=11)
    ax1.set_ylabel("Failure Rate (%)", fontsize=11)
    ax1.set_title(f"Expected vs. Observed Failure Rates ({confidence_level:.0%} CL)",
                  fontsize=13, fontweight="bold")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis="y")

    # ── Right: p-values with significance threshold ──
    ax2 = axes[1]

    colors = ["#10B981" if adequate else "#EF4444"
              for adequate in calibration_df["Model Adequate"]]

    ax2.bar(x, calibration_df["p-value"], color=colors, alpha=0.8, width=0.5)
    ax2.axhline(y=0.05, color="#F59E0B", linewidth=2, linestyle="--",
                label="Significance Level (α=0.05)")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"{w}d" for w in calibration_df["Window"]])
    ax2.set_xlabel("Lookback Window", fontsize=11)
    ax2.set_ylabel("p-value", fontsize=11)
    ax2.set_title("Kupiec POF Test p-values by Window", fontsize=13, fontweight="bold")
    ax2.legend(framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis="y")

    # Add pass/fail labels
    for i, (pval, adequate) in enumerate(
        zip(calibration_df["p-value"], calibration_df["Model Adequate"])
    ):
        label = "✓ Pass" if adequate else "✗ Fail"
        color = "#10B981" if adequate else "#EF4444"
        ax2.text(i, pval + 0.02, label, ha="center", fontsize=10,
                 fontweight="bold", color=color)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "06_window_calibration.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def plot_correlation_heatmap(returns: pd.DataFrame):
    """
    Plot 7: Correlation heatmap of ETF returns.
    """
    setup_style()
    ensure_output_dir()

    fig, ax = plt.subplots(figsize=(10, 8))

    corr = returns.corr()

    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

    cmap = sns.diverging_palette(250, 15, s=75, l=40, n=9, center="light", as_cmap=True)

    sns.heatmap(
        corr, mask=mask, cmap=cmap, vmin=-1, vmax=1,
        annot=True, fmt=".2f", linewidths=1, linecolor="#0F172A",
        square=True, ax=ax,
        cbar_kws={"shrink": 0.8, "label": "Correlation"},
        annot_kws={"size": 11, "weight": "bold"}
    )

    ax.set_title("ETF Return Correlations", fontsize=16, fontweight="bold", pad=20)
    ax.tick_params(axis="both", labelsize=10)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "07_correlation_heatmap.png"), dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()


def generate_all_plots(
    prices, returns, portfolio_returns,
    var_results, backtest_results, calibration_df,
    simulated_returns
):
    """Generate all visualization plots."""
    print(f"\n{'='*60}")
    print("  GENERATING VISUALIZATIONS")
    print(f"{'='*60}\n")

    plot_portfolio_composition(prices)
    print("  ✓ Plot 1: Portfolio Composition")

    plot_return_distribution(portfolio_returns, var_results)
    print("  ✓ Plot 2: Return Distribution with VaR")

    plot_monte_carlo(simulated_returns, var_results)
    print("  ✓ Plot 3: Monte Carlo Simulation")

    plot_var_comparison(var_results)
    print("  ✓ Plot 4: VaR Method Comparison")

    plot_backtest_results(backtest_results)
    print("  ✓ Plot 5: Backtesting Results")

    plot_window_calibration(calibration_df)
    print("  ✓ Plot 6: Window Calibration")

    plot_correlation_heatmap(returns)
    print("  ✓ Plot 7: Correlation Heatmap")

    print(f"\n  All plots saved to '{OUTPUT_DIR}/' directory.\n")
