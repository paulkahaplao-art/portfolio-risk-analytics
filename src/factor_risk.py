import numpy as np
import pandas as pd


def calculate_factor_exposure(
    asset_returns,
    factor_returns,
):
    """Estimate asset exposure to factors using OLS."""

    aligned = asset_returns.join(
        factor_returns,
        how="inner",
    )

    factors = factor_returns.columns

    exposures = {}

    X = aligned[factors].values

    X = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    for asset in asset_returns.columns:

        y = aligned[asset].values

        coefficients = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        exposures[asset] = coefficients[1:]

    return pd.DataFrame(
        exposures,
        index=factors,
    ).T


def calculate_portfolio_factor_exposure(
    asset_exposures,
    weights,
):
    """Calculate portfolio exposure to each factor."""

    aligned_weights = weights.reindex(
        asset_exposures.index
    )

    return (
        asset_exposures
        .mul(aligned_weights, axis=0)
        .sum(axis=0)
    )


def calculate_factor_covariance(
    factor_returns,
):
    """Calculate annualised factor covariance."""

    return factor_returns.cov() * 252


def calculate_factor_risk_contribution(
    portfolio_factor_exposure,
    factor_covariance,
):
    """Calculate factor-level risk contribution."""

    factor_exposure = (
        portfolio_factor_exposure
    )

    factor_variance = (
        factor_covariance
        @ factor_exposure
    )

    portfolio_factor_variance = (
        factor_exposure.T
        @ factor_covariance
        @ factor_exposure
    )

    portfolio_factor_volatility = np.sqrt(
        portfolio_factor_variance
    )

    if portfolio_factor_volatility == 0:
        raise ValueError(
            "Portfolio factor volatility is zero."
        )

    marginal_factor_risk = (
        factor_variance
        / portfolio_factor_volatility
    )

    component_factor_risk = (
        factor_exposure
        * marginal_factor_risk
    )

    percentage_factor_risk = (
        component_factor_risk
        / portfolio_factor_volatility
    )

    return pd.DataFrame({
        "Factor Exposure": factor_exposure,
        "Marginal Risk": marginal_factor_risk,
        "Component Risk": component_factor_risk,
        "Percentage Risk": percentage_factor_risk,
    })


def calculate_factor_diversification_benefit(
    portfolio_factor_exposure,
    factor_covariance,
):
    """Calculate diversification benefit across factors."""

    factor_volatility = np.sqrt(
        np.diag(factor_covariance)
    )

    weighted_factor_volatility = (
        portfolio_factor_exposure
        .abs()
        * factor_volatility
    ).sum()

    portfolio_volatility = np.sqrt(
        portfolio_factor_exposure.T
        @ factor_covariance
        @ portfolio_factor_exposure
    )

    return (
        weighted_factor_volatility
        - portfolio_volatility
    )

def calculate_factor_fitted_returns(
    asset_returns,
    factor_returns,
):
    """Calculate returns explained by the factor model."""

    aligned = asset_returns.join(
        factor_returns,
        how="inner",
    )

    factors = factor_returns.columns

    X = aligned[factors].values

    X = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    fitted_returns = {}

    for asset in asset_returns.columns:

        y = aligned[asset].values

        coefficients = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        fitted = X @ coefficients

        fitted_returns[asset] = fitted

    return pd.DataFrame(
        fitted_returns,
        index=aligned.index,
    )

def calculate_residual_returns(
    asset_returns,
    factor_returns,
):
    """Calculate asset returns not explained by the factor model."""

    fitted_returns = calculate_factor_fitted_returns(
        asset_returns,
        factor_returns,
    )

    aligned_returns = asset_returns.loc[
        fitted_returns.index,
        fitted_returns.columns,
    ]

    return aligned_returns - fitted_returns

def calculate_residual_risk(
    residual_returns,
):
    """Calculate annualised residual volatility."""

    return residual_returns.std() * np.sqrt(252)

def calculate_factor_r_squared(
    asset_returns,
    factor_returns,
):
    """Calculate R-squared for each asset factor regression."""

    aligned = asset_returns.join(
        factor_returns,
        how="inner",
    )

    factors = factor_returns.columns

    X = aligned[factors].values

    X = np.column_stack([
        np.ones(len(X)),
        X,
    ])

    r_squared = {}

    for asset in asset_returns.columns:

        y = aligned[asset].values

        coefficients = np.linalg.lstsq(
            X,
            y,
            rcond=None,
        )[0]

        fitted = X @ coefficients

        residual = y - fitted

        ss_residual = np.sum(
            residual ** 2
        )

        ss_total = np.sum(
            (y - y.mean()) ** 2
        )

        if ss_total == 0:
            r_squared[asset] = 0.0
        else:
            r_squared[asset] = (
                1
                - ss_residual / ss_total
            )

    return pd.Series(r_squared)

def create_factor_model_summary(
    asset_returns,
    factor_returns,
):
    """Create a summary of factor model quality."""

    r_squared = calculate_factor_r_squared(
        asset_returns,
        factor_returns,
    )

    residual_returns = calculate_residual_returns(
        asset_returns,
        factor_returns,
    )

    residual_risk = calculate_residual_risk(
        residual_returns,
    )

    total_risk = (
        asset_returns.std()
        * np.sqrt(252)
    )

    summary = pd.DataFrame({
        "R Squared": r_squared,
        "Residual Volatility": residual_risk,
        "Total Volatility": total_risk,
    })

    summary["Explained Risk Indicator"] = (
        1
        - (
            summary["Residual Volatility"]
            / summary["Total Volatility"]
        )
    )

    return summary

if __name__ == "__main__":

    from src.historical_data import (
        load_historical_data,
    )

    asset_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    factor_data = pd.read_csv(
        "data/historical/factor_data.csv",
        parse_dates=["Date"],
    )

    factor_data = factor_data.set_index(
        "Date"
    )

    asset_returns = (
        asset_data
        .pct_change()
        .dropna()
    )

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    residual_returns = calculate_residual_returns(
    asset_returns,
    factor_returns,
    )

    residual_risk = calculate_residual_risk(
    residual_returns,
    )

    model_summary = create_factor_model_summary(
    asset_returns,
    factor_returns,
)

    portfolio_exposure = (
        calculate_portfolio_factor_exposure(
            exposures,
            weights,
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

    diversification_benefit = (
        calculate_factor_diversification_benefit(
            portfolio_exposure,
            factor_covariance,
        )
    )

    print("\nAsset Factor Exposures")
    print("=" * 80)

    print(
        exposures.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nPortfolio Factor Exposure")
    print("=" * 80)

    print(
        portfolio_exposure.to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFactor Risk Attribution")
    print("=" * 80)

    print(
        factor_risk.to_string(
            formatters={
                "Factor Exposure": "{:.4f}".format,
                "Marginal Risk": "{:.6f}".format,
                "Component Risk": "{:.2%}".format,
                "Percentage Risk": "{:.2%}".format,
            }
        )
    )

    print("\nFactor Diversification Benefit")
    print("=" * 80)

    print(
        f"{diversification_benefit:.2%}"
    )

    print("\nResidual Risk")
    print("=" * 80)

    print(
        residual_risk.to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print("\nFactor Model Quality")
    print("=" * 80)

    print(
        model_summary.to_string(
            formatters={
                "R Squared": "{:.2%}".format,
                "Residual Volatility": "{:.2%}".format,
                "Total Volatility": "{:.2%}".format,
                "Explained Risk Indicator": "{:.2%}".format,
            }
        )
    )

def calculate_factor_stress_return(
    portfolio_factor_exposure,
    factor_shock,
):
    """Calculate portfolio return under a factor shock."""

    shock = pd.Series(
        factor_shock
    ).reindex(
        portfolio_factor_exposure.index
    )

    if shock.isna().any():
        raise ValueError(
            "Factor shock is missing one or more factors."
        )

    return (
        portfolio_factor_exposure
        * shock
    ).sum()

def calculate_factor_stress_loss(
    portfolio_value,
    portfolio_return,
):
    """Calculate portfolio P&L under a factor stress."""

    return (
        portfolio_value
        * portfolio_return
    )

def run_factor_stress_scenarios(
    portfolio_factor_exposure,
    portfolio_value,
):
    """Run predefined factor stress scenarios."""

    scenarios = {
        "Equity Shock": {
            "Equity": -0.20,
            "Rates": 0.00,
            "Credit": 0.00,
        },
        "Rates Shock": {
            "Equity": 0.00,
            "Rates": -0.10,
            "Credit": 0.00,
        },
        "Credit Shock": {
            "Equity": 0.00,
            "Rates": 0.00,
            "Credit": -0.15,
        },
        "Multi-Factor Crisis": {
            "Equity": -0.20,
            "Rates": -0.10,
            "Credit": -0.15,
        },
    }

    results = []

    for scenario_name, scenario in scenarios.items():

        portfolio_return = (
            calculate_factor_stress_return(
                portfolio_factor_exposure,
                scenario,
            )
        )

        pnl = calculate_factor_stress_loss(
            portfolio_value,
            portfolio_return,
        )

        results.append({
            "Scenario": scenario_name,
            "Portfolio Return": portfolio_return,
            "Portfolio P&L": pnl,
        })

    return pd.DataFrame(results)

if __name__ == "__main__": 

    portfolio_exposure = (
    calculate_portfolio_factor_exposure(
        exposures,
        weights,
    )
    )

    factor_stress_results = (
    run_factor_stress_scenarios(
        portfolio_exposure,
        100_000,
    )
    )   

    print("\nFactor Stress Testing")
    print("=" * 80)

    print(
    factor_stress_results.to_string(
        index=False,
        formatters={
            "Portfolio Return": "{:.2%}".format,
            "Portfolio P&L": "${:,.2f}".format,
        }
    )
    )   