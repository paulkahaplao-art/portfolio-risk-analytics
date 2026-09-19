from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.historical_data import load_historical_data
from src.portfolio_dynamics import calculate_rebalanced_returns
from src.historical_analysis import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_maximum_drawdown,
    calculate_rolling_volatility,
    calculate_rolling_var,
)
from src.stress_testing import (
    run_stress_scenarios,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/historical/market_data.csv"
OUTPUT_DIR = Path("Output")
PORTFOLIO_VALUE = 100_000

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


# ============================================================
# Helper functions
# ============================================================

def save_figure(filename):
    """
    Save the current matplotlib figure to the Output directory.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUT_DIR / filename

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"Created: {output_path}")


def load_portfolio_data():
    """
    Load historical asset data and calculate portfolio returns.
    """
    historical_data = load_historical_data(DATA_PATH)

    asset_returns = historical_data.pct_change().dropna()

    portfolio_returns = calculate_rebalanced_returns(
        asset_returns,
        WEIGHTS,
    )

    return historical_data, asset_returns, portfolio_returns


# ============================================================
# 1. Portfolio Performance
# ============================================================

def create_portfolio_performance(
    historical_data,
    portfolio_returns,
):
    """
    Create cumulative portfolio and asset performance chart.
    """

    portfolio_growth = (1 + portfolio_returns).cumprod() * 100

    asset_growth = (
        historical_data
        / historical_data.iloc[0]
        * 100
    )

    plt.figure(figsize=(12, 7))

    plt.plot(
        portfolio_growth.index,
        portfolio_growth,
        linewidth=2.5,
        label="Portfolio",
    )

    for column in asset_growth.columns:
        plt.plot(
            asset_growth.index,
            asset_growth[column],
            linewidth=1.5,
            label=column.replace("_", " "),
        )

    plt.title(
        "Portfolio Performance",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Date")
    plt.ylabel("Growth of $100")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    plt.tight_layout()

    save_figure("portfolio_performance.png")

    plt.close()


# ============================================================
# 2. Portfolio Drawdown
# ============================================================

def create_portfolio_drawdown(
    portfolio_returns,
):
    """
    Create portfolio drawdown chart.
    """

    cumulative_growth = (
        1 + portfolio_returns
    ).cumprod()

    running_peak = cumulative_growth.cummax()

    drawdown = (
        cumulative_growth / running_peak
    ) - 1

    maximum_drawdown = drawdown.min()

    plt.figure(figsize=(12, 7))

    plt.plot(
        drawdown.index,
        drawdown * 100,
        linewidth=2,
        label="Portfolio Drawdown",
    )

    # Illustrative drawdown limit from Lesson 34.
    drawdown_limit = -0.10

    plt.axhline(
        drawdown_limit * 100,
        linestyle="--",
        linewidth=1.5,
        label="10% Drawdown Limit",
    )

    plt.title(
        "Portfolio Drawdown",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Date")
    plt.ylabel("Drawdown (%)")

    plt.legend()

    plt.grid(
        True,
        alpha=0.3,
    )

    # Display maximum drawdown on the chart.
    plt.text(
        0.02,
        0.05,
        f"Maximum Drawdown: {maximum_drawdown:.2%}",
        transform=plt.gca().transAxes,
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            alpha=0.1,
        ),
    )

    plt.tight_layout()

    save_figure("portfolio_drawdown.png")

    plt.close()


# ============================================================
# 3. Risk Metrics
# ============================================================

def create_risk_metrics(
    portfolio_returns,
):
    """
    Create rolling risk metrics chart.
    """

    rolling_volatility = calculate_rolling_volatility(
        portfolio_returns,
        window=63,
    )

    rolling_var = calculate_rolling_var(
        portfolio_returns,
        window=252,
        confidence=0.95,
    )

    cumulative_growth = (
        1 + portfolio_returns
    ).cumprod()

    running_peak = cumulative_growth.cummax()

    drawdown = (
        cumulative_growth / running_peak
    ) - 1

    # Create three-panel figure.
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(12, 10),
        sharex=True,
    )

    # --------------------------------------------------------
    # Volatility
    # --------------------------------------------------------

    axes[0].plot(
        rolling_volatility.index,
        rolling_volatility * 100,
        linewidth=2,
    )

    axes[0].axhline(
        8,
        linestyle="--",
        linewidth=1.5,
        label="8% Risk Limit",
    )

    axes[0].set_title(
        "Rolling Annualised Volatility"
    )

    axes[0].set_ylabel("Volatility (%)")

    axes[0].legend()

    axes[0].grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # VaR
    # --------------------------------------------------------

    axes[1].plot(
        rolling_var.index,
        rolling_var * 100,
        linewidth=2,
    )

    axes[1].axhline(
        -2,
        linestyle="--",
        linewidth=1.5,
        label="2% VaR Limit",
    )

    axes[1].set_title(
        "Rolling Historical VaR (95%)"
    )

    axes[1].set_ylabel("VaR (%)")

    axes[1].legend()

    axes[1].grid(
        True,
        alpha=0.3,
    )

    # --------------------------------------------------------
    # Drawdown
    # --------------------------------------------------------

    axes[2].plot(
        drawdown.index,
        drawdown * 100,
        linewidth=2,
    )

    axes[2].axhline(
        -10,
        linestyle="--",
        linewidth=1.5,
        label="10% Drawdown Limit",
    )

    axes[2].set_title(
        "Portfolio Drawdown"
    )

    axes[2].set_ylabel("Drawdown (%)")
    axes[2].set_xlabel("Date")

    axes[2].legend()

    axes[2].grid(
        True,
        alpha=0.3,
    )

    fig.suptitle(
        "Portfolio Risk Metrics",
        fontsize=16,
        fontweight="bold",
    )

    plt.tight_layout()

    save_figure("risk_metrics.png")

    plt.close()


# ============================================================
# 4. Stress Testing
# ============================================================

def create_stress_test():
    """
    Create portfolio stress-test chart.
    """

    results = run_stress_scenarios(
        PORTFOLIO_VALUE,
        WEIGHTS,
    )

    results = results.sort_values(
        "Portfolio P&L"
    )

    plt.figure(figsize=(12, 7))

    bars = plt.barh(
        results["Scenario"],
        results["Portfolio P&L"],
    )

    plt.axvline(
        0,
        linewidth=1,
    )

    plt.title(
        "Portfolio Stress Test Results",
        fontsize=16,
        fontweight="bold",
    )

    plt.xlabel("Portfolio P&L ($)")
    plt.ylabel("Scenario")

    plt.grid(
        axis="x",
        alpha=0.3,
    )

    # Add values to each bar.
    for bar, value in zip(
        bars,
        results["Portfolio P&L"],
    ):
        if value >= 0:
            x_position = value
            alignment = "left"
        else:
            x_position = value
            alignment = "right"

        plt.text(
            x_position,
            bar.get_y() + bar.get_height() / 2,
            f"${value:,.0f}",
            va="center",
            ha=alignment,
            fontsize=10,
        )

    plt.tight_layout()

    save_figure("stress_test.png")

    plt.close()

    return results


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Generating portfolio analytics outputs")
    print("=" * 60)

    historical_data, asset_returns, portfolio_returns = (
        load_portfolio_data()
    )

    print("\nCreating portfolio performance...")
    create_portfolio_performance(
        historical_data,
        portfolio_returns,
    )

    print("\nCreating portfolio drawdown...")
    create_portfolio_drawdown(
        portfolio_returns,
    )

    print("\nCreating risk metrics...")
    create_risk_metrics(
        portfolio_returns,
    )

    print("\nCreating stress test...")
    stress_results = create_stress_test()

    print("\nStress test results:")
    print(stress_results)

    print("\n" + "=" * 60)
    print("Output generation complete")
    print("=" * 60)

    print(f"\nFiles saved to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()