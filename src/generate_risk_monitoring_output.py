from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.historical_data import load_historical_data
from src.portfolio_dynamics import calculate_rebalanced_returns
from src.risk_monitoring_history import (
    calculate_rolling_risk_monitor,
    calculate_historical_limit_utilisation,
    add_historical_risk_status,
    create_historical_monitoring_summary,
)


DATA_PATH = "data/historical/market_data.csv"
OUTPUT_DIR = Path("Output")

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})

VOLATILITY_LIMIT = 0.08
VAR_LIMIT = -0.02
DRAWDOWN_LIMIT = -0.10


def load_portfolio_returns():
    historical_data = load_historical_data(
        DATA_PATH
    )

    asset_returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    portfolio_returns = (
        calculate_rebalanced_returns(
            asset_returns,
            WEIGHTS,
        )
    )

    return portfolio_returns


def build_monitoring_history(
    portfolio_returns
):
    risk_history = calculate_rolling_risk_monitor(
        portfolio_returns,
        volatility_window=63,
        var_window=252,
        var_confidence=0.95,
    )

    risk_history = (
        calculate_historical_limit_utilisation(
            risk_history,
            volatility_limit=VOLATILITY_LIMIT,
            var_limit=VAR_LIMIT,
            drawdown_limit=DRAWDOWN_LIMIT,
        )
    )

    risk_history = add_historical_risk_status(
        risk_history,
        volatility_limit=VOLATILITY_LIMIT,
        var_limit=VAR_LIMIT,
        drawdown_limit=DRAWDOWN_LIMIT,
    )

    return risk_history


def save_monitoring_history(
    risk_history
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "risk_monitoring_history.csv"
    )

    risk_history.to_csv(
        output_path,
        index=True,
    )

    return output_path


def create_monitoring_chart(
    risk_history
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(14, 11),
        sharex=True,
    )

    # ---------------------------------------------
    # Volatility
    # ---------------------------------------------

    axes[0].plot(
        risk_history.index,
        risk_history["Rolling Volatility"] * 100,
        linewidth=1.8,
        label="Rolling Volatility",
    )

    axes[0].axhline(
        VOLATILITY_LIMIT * 100,
        linestyle="--",
        linewidth=1.5,
        label="8% Limit",
    )

    axes[0].set_title(
        "Rolling Annualised Volatility"
    )

    axes[0].set_ylabel(
        "Volatility (%)"
    )

    axes[0].legend()
    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    # ---------------------------------------------
    # VaR
    # ---------------------------------------------

    axes[1].plot(
        risk_history.index,
        risk_history["Rolling VaR"] * 100,
        linewidth=1.8,
        label="Rolling VaR 95%",
    )

    axes[1].axhline(
        VAR_LIMIT * 100,
        linestyle="--",
        linewidth=1.5,
        label="2% VaR Limit",
    )

    axes[1].set_title(
        "Rolling Historical VaR"
    )

    axes[1].set_ylabel(
        "VaR (%)"
    )

    axes[1].legend()
    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    # ---------------------------------------------
    # Drawdown
    # ---------------------------------------------

    axes[2].plot(
        risk_history.index,
        risk_history["Drawdown"] * 100,
        linewidth=1.8,
        label="Portfolio Drawdown",
    )

    axes[2].axhline(
        DRAWDOWN_LIMIT * 100,
        linestyle="--",
        linewidth=1.5,
        label="10% Drawdown Limit",
    )

    axes[2].set_title(
        "Portfolio Drawdown"
    )

    axes[2].set_ylabel(
        "Drawdown (%)"
    )

    axes[2].set_xlabel(
        "Date"
    )

    axes[2].legend()
    axes[2].grid(
        axis="y",
        alpha=0.3,
    )

    fig.suptitle(
        "Historical Portfolio Risk Monitoring",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "risk_monitoring_history.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    return output_path


def main():
    print("=" * 60)
    print("Historical Risk Monitoring")
    print("=" * 60)

    portfolio_returns = (
        load_portfolio_returns()
    )

    risk_history = (
        build_monitoring_history(
            portfolio_returns
        )
    )

    print("\nMonitoring History")
    print("-" * 60)

    print(
        risk_history.tail(10).to_string()
    )

    print("\nMonitoring Summary")
    print("-" * 60)

    summary = (
        create_historical_monitoring_summary(
            risk_history
        )
    )

    for key, value in summary.items():
        if isinstance(value, float):
            print(
                f"{key}: {value:.2%}"
            )
        else:
            print(
                f"{key}: {value}"
            )

    print("\nStatus Counts")
    print("-" * 60)

    print(
        risk_history["Status"]
        .value_counts()
    )

    csv_path = (
        save_monitoring_history(
            risk_history
        )
    )

    chart_path = (
        create_monitoring_chart(
            risk_history
        )
    )

    print("\nCreated:")
    print(csv_path)
    print(chart_path)

    print("\nComplete.")


if __name__ == "__main__":
    main()