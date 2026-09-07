import pandas as pd

from src.historical_data import load_historical_data
from src.risk_attribution import (
    calculate_covariance_matrix,
    calculate_portfolio_volatility,
    create_risk_attribution_table,
)
from src.risk_budget import (
    calculate_risk_concentration,
    identify_risk_concentration,
    create_risk_budget_table,
)


WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


def load_portfolio_returns(filepath):
    """Load historical prices and calculate daily returns."""

    historical_data = load_historical_data(filepath)

    returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    return returns


def calculate_actual_risk_attribution(
    returns,
    weights,
):
    """Calculate actual portfolio risk attribution."""

    covariance = calculate_covariance_matrix(
        returns
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            weights,
            covariance,
        )
    )

    attribution = create_risk_attribution_table(
        weights,
        covariance,
    )

    return (
        covariance,
        portfolio_volatility,
        attribution,
    )


def create_actual_risk_budget(
    attribution,
    target_risk,
):
    """Create risk budget using actual attribution."""

    actual_risk = attribution[
        "Percentage Contribution"
    ]

    return create_risk_budget_table(
        attribution["Weight"],
        actual_risk,
        target_risk,
    )

def validate_risk_attribution(attribution):
    """Validate that risk contributions reconcile to 100%."""

    total = attribution[
        "Percentage Contribution"
    ].sum()

    if abs(total - 1.0) > 1e-10:
        raise ValueError(
            f"Risk contribution does not reconcile: {total:.10f}"
        )

    return True


if __name__ == "__main__":

    filepath = (
        "data/historical/market_data.csv"
    )

    returns = load_portfolio_returns(
        filepath
    )

    covariance, portfolio_volatility, attribution = (
        calculate_actual_risk_attribution(
            returns,
            WEIGHTS,
        )
    )

    covariance, portfolio_volatility, attribution = (
    calculate_actual_risk_attribution(
        returns,
        WEIGHTS,
    )
)

    validate_risk_attribution(attribution)

    target_risk = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    risk_concentration = calculate_risk_concentration(
        attribution["Percentage Contribution"],
        WEIGHTS,
    )

    risk_budget = create_actual_risk_budget(
        attribution,
        target_risk,
    )

    concentrated_assets = identify_risk_concentration(
        attribution["Percentage Contribution"],
        WEIGHTS,
        threshold=1.5,
    )

    print("\nActual Portfolio Risk Attribution")
    print("=" * 90)

    print(
        attribution.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Marginal Contribution": "{:.6f}".format,
                "Component Contribution": "{:.2%}".format,
                "Percentage Contribution": "{:.2%}".format,
            }
        )
    )

    print("\nPortfolio Volatility")
    print("=" * 90)

    print(
        f"{portfolio_volatility:.2%}"
    )

    print("\nRisk Concentration")
    print("=" * 90)

    print(
        risk_concentration.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Risk Contribution": "{:.2%}".format,
                "Risk / Capital Ratio": "{:.2f}x".format,
            }
        )
    )

    print("\nRisk Budget")
    print("=" * 90)

    print(
        risk_budget.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Actual Risk Contribution": "{:.2%}".format,
                "Target Risk Contribution": "{:.2%}".format,
                "Risk Budget Deviation": "{:.2%}".format,
                "Risk / Capital Ratio": "{:.2f}x".format,
            }
        )
    )

    print("\nPotential Risk Concentrations")
    print("=" * 90)

    if concentrated_assets.empty:
        print("No assets exceed the concentration threshold.")
    else:
        print(
            concentrated_assets.to_string(
                formatters={
                    "Weight": "{:.2%}".format,
                    "Risk Contribution": "{:.2%}".format,
                    "Risk / Capital Ratio": "{:.2f}x".format,
                }
            )
        )