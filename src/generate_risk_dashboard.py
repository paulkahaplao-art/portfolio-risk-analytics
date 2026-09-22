from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.historical_data import load_historical_data
from src.portfolio_dynamics import calculate_rebalanced_returns
from src.historical_analysis import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_maximum_drawdown,
)
from src.risk_attribution import (
    calculate_covariance_matrix,
    create_risk_attribution_table,
)
from src.factor_risk import (
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
)
from src.stress_testing import run_stress_scenarios


DATA_PATH = "data/historical/market_data.csv"
FACTOR_PATH = "data/historical/factor_data.csv"
OUTPUT_DIR = Path("Output")
PORTFOLIO_VALUE = 100_000

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})

VOLATILITY_LIMIT = 0.08
VAR_LIMIT = -0.02
ES_LIMIT = -0.03
DRAWDOWN_LIMIT = -0.10


def load_data():
    historical_data = load_historical_data(DATA_PATH)

    factor_data = pd.read_csv(
        FACTOR_PATH,
        parse_dates=["Date"],
    ).set_index("Date")

    asset_returns = historical_data.pct_change().dropna()

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    portfolio_returns = calculate_rebalanced_returns(
        asset_returns,
        WEIGHTS,
    )

    return (
        historical_data,
        asset_returns,
        factor_returns,
        portfolio_returns,
    )


def calculate_risk_metrics(portfolio_returns):
    volatility = (
        portfolio_returns.std()
        * (252 ** 0.5)
    )

    var_95 = calculate_historical_var(
        portfolio_returns,
        confidence=0.95,
    )

    es_95 = calculate_expected_shortfall(
        portfolio_returns,
        confidence=0.95,
    )

    maximum_drawdown = calculate_maximum_drawdown(
        portfolio_returns
    )

    return {
        "Volatility": volatility,
        "VaR 95%": var_95,
        "Expected Shortfall 95%": es_95,
        "Maximum Drawdown": maximum_drawdown,
    }


def calculate_risk_status(
    actual,
    limit,
    warning_ratio=0.875,
):
    utilisation = abs(actual) / abs(limit)

    if utilisation >= 1.0:
        return "BREACH"

    if utilisation >= warning_ratio:
        return "WARNING"

    return "PASS"


def create_risk_status_table(risk_metrics):
    rows = []

    metric_limits = {
        "Volatility": VOLATILITY_LIMIT,
        "VaR 95%": VAR_LIMIT,
        "Expected Shortfall 95%": ES_LIMIT,
        "Maximum Drawdown": DRAWDOWN_LIMIT,
    }

    for metric, actual in risk_metrics.items():
        limit = metric_limits[metric]

        status = calculate_risk_status(
            actual,
            limit,
        )

        utilisation = (
            abs(actual)
            / abs(limit)
        )

        rows.append({
            "Metric": metric,
            "Actual": actual,
            "Limit": limit,
            "Utilisation": utilisation,
            "Status": status,
        })

    return pd.DataFrame(rows)


def calculate_asset_attribution(asset_returns):
    covariance = calculate_covariance_matrix(
        asset_returns
    )

    return create_risk_attribution_table(
        WEIGHTS,
        covariance,
    )


def calculate_factor_exposures(
    asset_returns,
    factor_returns,
):
    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    portfolio_exposure = (
        calculate_portfolio_factor_exposure(
            exposures,
            WEIGHTS,
        )
    )

    return portfolio_exposure


def calculate_stress_results():
    results = run_stress_scenarios(
        PORTFOLIO_VALUE,
        WEIGHTS,
    )

    return results.sort_values(
        "Portfolio P&L"
    )


def create_dashboard(
    risk_metrics,
    risk_status,
    asset_attribution,
    factor_exposure,
    stress_results,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig = plt.figure(
        figsize=(16, 12)
    )

    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[1, 1.3, 1.3],
    )

    # -------------------------------------------------
    # 1. Risk Metrics
    # -------------------------------------------------

    ax1 = fig.add_subplot(
        grid[0, 0]
    )

    metric_names = [
        "Volatility",
        "VaR 95%",
        "ES 95%",
        "Max Drawdown",
    ]

    metric_values = [
        risk_metrics["Volatility"] * 100,
        risk_metrics["VaR 95%"] * 100,
        risk_metrics["Expected Shortfall 95%"] * 100,
        risk_metrics["Maximum Drawdown"] * 100,
    ]

    ax1.bar(
        metric_names,
        metric_values,
    )

    ax1.axhline(
        0,
        linewidth=1,
    )

    ax1.set_title(
        "Portfolio Risk Metrics",
        fontsize=13,
        fontweight="bold",
    )

    ax1.set_ylabel(
        "Percentage (%)"
    )

    ax1.tick_params(
        axis="x",
        rotation=25,
    )

    ax1.grid(
        axis="y",
        alpha=0.3,
    )

    # -------------------------------------------------
    # 2. Risk Limit Status
    # -------------------------------------------------

    ax2 = fig.add_subplot(
        grid[0, 1]
    )

    status_values = (
        risk_status["Utilisation"] * 100
    )

    ax2.barh(
        risk_status["Metric"],
        status_values,
    )

    ax2.axvline(
        100,
        linestyle="--",
        linewidth=1.5,
        label="Limit",
    )

    ax2.axvline(
        87.5,
        linestyle=":",
        linewidth=1.5,
        label="Warning",
    )

    ax2.set_title(
        "Risk Limit Utilisation",
        fontsize=13,
        fontweight="bold",
    )

    ax2.set_xlabel(
        "Utilisation (%)"
    )

    ax2.legend()

    ax2.grid(
        axis="x",
        alpha=0.3,
    )

    # -------------------------------------------------
    # 3. Asset Risk Attribution
    # -------------------------------------------------

    ax3 = fig.add_subplot(
        grid[1, 0]
    )

    labels = [
        label.replace("_", " ")
        for label in asset_attribution.index
    ]

    contributions = (
        asset_attribution[
            "Percentage Contribution"
        ] * 100
    )

    ax3.bar(
        labels,
        contributions,
    )

    ax3.set_title(
        "Asset-Level Risk Contribution",
        fontsize=13,
        fontweight="bold",
    )

    ax3.set_ylabel(
        "Risk Contribution (%)"
    )

    ax3.tick_params(
        axis="x",
        rotation=25,
    )

    ax3.grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(
        contributions
    ):
        ax3.text(
            index,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    # -------------------------------------------------
    # 4. Factor Exposure
    # -------------------------------------------------

    ax4 = fig.add_subplot(
        grid[1, 1]
    )

    factor_labels = [
        label.replace("_", " ")
        for label in factor_exposure.index
    ]

    factor_values = factor_exposure.values

    ax4.bar(
        factor_labels,
        factor_values,
    )

    ax4.axhline(
        0,
        linewidth=1,
    )

    ax4.set_title(
        "Portfolio Factor Exposure",
        fontsize=13,
        fontweight="bold",
    )

    ax4.set_ylabel(
        "Exposure"
    )

    ax4.grid(
        axis="y",
        alpha=0.3,
    )

    # -------------------------------------------------
    # 5. Stress Testing
    # -------------------------------------------------

    ax5 = fig.add_subplot(
        grid[2, 0]
    )

    stress_labels = (
        stress_results["Scenario"]
    )

    stress_pnl = (
        stress_results["Portfolio P&L"]
    )

    ax5.barh(
        stress_labels,
        stress_pnl,
    )

    ax5.axvline(
        0,
        linewidth=1,
    )

    ax5.set_title(
        "Stress-Test Portfolio P&L",
        fontsize=13,
        fontweight="bold",
    )

    ax5.set_xlabel(
        "Portfolio P&L ($)"
    )

    ax5.grid(
        axis="x",
        alpha=0.3,
    )

    for index, value in enumerate(
        stress_pnl
    ):
        ax5.text(
            value,
            index,
            f"${value:,.0f}",
            va="center",
            ha="right" if value < 0 else "left",
            fontsize=9,
        )

    # -------------------------------------------------
    # 6. Dashboard Summary
    # -------------------------------------------------

    ax6 = fig.add_subplot(
        grid[2, 1]
    )

    ax6.axis("off")

    overall_status = "PASS"

    if (
        risk_status["Status"]
        == "BREACH"
    ).any():
        overall_status = "BREACH"

    elif (
        risk_status["Status"]
        == "WARNING"
    ).any():
        overall_status = "WARNING"

    summary_text = (
        "PORTFOLIO RISK DASHBOARD\n\n"
        f"Portfolio Value: "
        f"${PORTFOLIO_VALUE:,.0f}\n\n"
        f"Volatility: "
        f"{risk_metrics['Volatility']:.2%}\n"
        f"VaR 95%: "
        f"{risk_metrics['VaR 95%']:.2%}\n"
        f"ES 95%: "
        f"{risk_metrics['Expected Shortfall 95%']:.2%}\n"
        f"Maximum Drawdown: "
        f"{risk_metrics['Maximum Drawdown']:.2%}\n\n"
        f"Overall Risk Status: "
        f"{overall_status}"
    )

    ax6.text(
        0.05,
        0.95,
        summary_text,
        transform=ax6.transAxes,
        fontsize=12,
        verticalalignment="top",
        bbox=dict(
            boxstyle="round",
            alpha=0.1,
        ),
    )

    fig.suptitle(
        "Integrated Portfolio Risk Dashboard",
        fontsize=20,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "portfolio_risk_dashboard.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Created: {output_path}"
    )


def main():
    print("=" * 60)
    print("Generating Portfolio Risk Dashboard")
    print("=" * 60)

    (
        historical_data,
        asset_returns,
        factor_returns,
        portfolio_returns,
    ) = load_data()

    risk_metrics = calculate_risk_metrics(
        portfolio_returns
    )

    risk_status = create_risk_status_table(
        risk_metrics
    )

    asset_attribution = (
        calculate_asset_attribution(
            asset_returns
        )
    )

    factor_exposure = (
        calculate_factor_exposures(
            asset_returns,
            factor_returns,
        )
    )

    stress_results = (
        calculate_stress_results()
    )

    print("\nRisk Metrics")
    print("-" * 60)

    for metric, value in risk_metrics.items():
        print(
            f"{metric}: {value:.2%}"
        )

    print("\nRisk Limit Status")
    print("-" * 60)
    print(risk_status)

    print("\nAsset Risk Attribution")
    print("-" * 60)
    print(asset_attribution)

    print("\nPortfolio Factor Exposure")
    print("-" * 60)
    print(factor_exposure)

    print("\nStress Test Results")
    print("-" * 60)
    print(stress_results)

    create_dashboard(
        risk_metrics,
        risk_status,
        asset_attribution,
        factor_exposure,
        stress_results,
    )

    print("\nDashboard generation complete.")
    print(
        f"Output directory: "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()