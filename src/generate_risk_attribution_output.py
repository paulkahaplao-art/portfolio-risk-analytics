from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.historical_data import load_historical_data
from src.risk_attribution import (
    calculate_covariance_matrix,
    create_risk_attribution_table,
)
from src.factor_risk import (
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
    calculate_factor_covariance,
    calculate_factor_risk_contribution,
)


# ============================================================
# Configuration
# ============================================================

DATA_PATH = "data/historical/market_data.csv"
FACTOR_PATH = "data/historical/factor_data.csv"
OUTPUT_DIR = Path("Output")

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


# ============================================================
# Load data
# ============================================================

def load_data():
    historical_data = load_historical_data(
        DATA_PATH
    )

    factor_data = pd.read_csv(
        FACTOR_PATH,
        parse_dates=["Date"],
    ).set_index("Date")

    asset_returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    return asset_returns, factor_returns


# ============================================================
# Calculate asset-level risk attribution
# ============================================================

def calculate_asset_risk_attribution(
    asset_returns,
):
    covariance = calculate_covariance_matrix(
        asset_returns
    )

    attribution = create_risk_attribution_table(
        WEIGHTS,
        covariance,
    )

    return attribution


# ============================================================
# Calculate factor-level risk attribution
# ============================================================

def calculate_factor_risk_attribution(
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

    factor_covariance = (
        calculate_factor_covariance(
            factor_returns
        )
    )

    factor_risk = (
        calculate_factor_risk_contribution(
            portfolio_exposure,
            factor_covariance,
        )
    )

    return factor_risk


# ============================================================
# Create visual
# ============================================================

def create_risk_attribution_chart(
    asset_attribution,
    factor_attribution,
):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(12, 10),
    )

    # --------------------------------------------------------
    # Asset-level risk contribution
    # --------------------------------------------------------

    asset_labels = [
        label.replace("_", " ")
        for label in asset_attribution.index
    ]

    asset_values = (
        asset_attribution[
            "Percentage Contribution"
        ] * 100
    )

    axes[0].bar(
        asset_labels,
        asset_values,
    )

    axes[0].set_title(
        "Asset-Level Risk Contribution",
        fontsize=14,
        fontweight="bold",
    )

    axes[0].set_ylabel(
        "Risk Contribution (%)"
    )

    axes[0].grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(
        asset_values
    ):
        axes[0].text(
            index,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    # --------------------------------------------------------
    # Factor-level risk contribution
    # --------------------------------------------------------

    factor_labels = [
        label.replace("_", " ")
        for label in factor_attribution.index
    ]

    factor_values = (
        factor_attribution[
            "Percentage Risk"
        ] * 100
    )

    axes[1].bar(
        factor_labels,
        factor_values,
    )

    axes[1].set_title(
        "Factor-Level Risk Contribution",
        fontsize=14,
        fontweight="bold",
    )

    axes[1].set_ylabel(
        "Risk Contribution (%)"
    )

    axes[1].set_xlabel(
        "Risk Factor"
    )

    axes[1].grid(
        axis="y",
        alpha=0.3,
    )

    for index, value in enumerate(
        factor_values
    ):
        axes[1].text(
            index,
            value,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
        )

    fig.suptitle(
        "Portfolio Risk Attribution",
        fontsize=17,
        fontweight="bold",
    )

    plt.tight_layout()

    output_path = (
        OUTPUT_DIR
        / "risk_attribution.png"
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


# ============================================================
# Main
# ============================================================

def main():

    print("=" * 60)
    print("Generating risk attribution output")
    print("=" * 60)

    asset_returns, factor_returns = (
        load_data()
    )

    asset_attribution = (
        calculate_asset_risk_attribution(
            asset_returns
        )
    )

    factor_attribution = (
        calculate_factor_risk_attribution(
            asset_returns,
            factor_returns,
        )
    )

    print("\nAsset-Level Risk Attribution")
    print("-" * 60)
    print(asset_attribution)

    print("\nFactor-Level Risk Attribution")
    print("-" * 60)
    print(factor_attribution)

    create_risk_attribution_chart(
        asset_attribution,
        factor_attribution,
    )

    print("\nComplete.")


if __name__ == "__main__":
    main()