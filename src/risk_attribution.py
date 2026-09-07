import numpy as np
import pandas as pd


def calculate_covariance_matrix(asset_returns):
    """Calculate annualised covariance matrix."""

    return asset_returns.cov() * 252


def calculate_portfolio_volatility(weights, covariance):
    """Calculate annualised portfolio volatility."""

    portfolio_variance = (
        weights.T
        @ covariance
        @ weights
    )

    return np.sqrt(portfolio_variance)


def calculate_marginal_contribution_to_risk(
    weights,
    covariance,
):
    """Calculate marginal contribution to portfolio volatility."""

    portfolio_volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    if portfolio_volatility == 0:
        raise ValueError(
            "Portfolio volatility is zero."
        )

    marginal_contribution = (
        covariance @ weights
    ) / portfolio_volatility

    return marginal_contribution


def calculate_component_contribution_to_risk(
    weights,
    covariance,
):
    """Calculate each asset's component contribution to risk."""

    marginal_contribution = (
        calculate_marginal_contribution_to_risk(
            weights,
            covariance,
        )
    )

    component_contribution = (
        weights * marginal_contribution
    )

    return component_contribution


def calculate_percentage_contribution_to_risk(
    weights,
    covariance,
):
    """Calculate percentage contribution to portfolio risk."""

    component_contribution = (
        calculate_component_contribution_to_risk(
            weights,
            covariance,
        )
    )

    portfolio_volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    return component_contribution / portfolio_volatility


def calculate_diversification_benefit(
    weights,
    covariance,
):
    """Calculate the volatility reduction from diversification."""

    individual_volatilities = np.sqrt(
        np.diag(covariance)
    )

    weighted_individual_volatility = (
        weights * individual_volatilities
    ).sum()

    portfolio_volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    return (
        weighted_individual_volatility
        - portfolio_volatility
    )


def create_risk_attribution_table(
    weights,
    covariance,
):
    """Create an asset-level risk attribution table."""

    marginal = calculate_marginal_contribution_to_risk(
        weights,
        covariance,
    )

    component = calculate_component_contribution_to_risk(
        weights,
        covariance,
    )

    percentage = calculate_percentage_contribution_to_risk(
        weights,
        covariance,
    )

    table = pd.DataFrame({
        "Weight": weights,
        "Marginal Contribution": marginal,
        "Component Contribution": component,
        "Percentage Contribution": percentage,
    })

    return table


if __name__ == "__main__":

    from src.historical_data import load_historical_data

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    historical_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    asset_returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    covariance = calculate_covariance_matrix(
        asset_returns
    )

    portfolio_volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    attribution = create_risk_attribution_table(
        weights,
        covariance,
    )

    diversification_benefit = (
        calculate_diversification_benefit(
            weights,
            covariance,
        )
    )

    print("\nRisk Attribution")
    print("=" * 80)

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
    print("-" * 80)
    print(f"{portfolio_volatility:.2%}")

    print("\nDiversification Benefit")
    print("-" * 80)
    print(f"{diversification_benefit:.2%}")

    print("\nEuler Reconciliation")
    print("-" * 80)
    print(
        f"Sum of component contributions: "
        f"{attribution['Component Contribution'].sum():.2%}"
    )
    print(
        f"Portfolio volatility: "
        f"{portfolio_volatility:.2%}"
    )