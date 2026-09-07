import numpy as np
import pandas as pd


def validate_weights(weights):
    """
    Validate portfolio weights.
    """

    if weights.isna().any():
        raise ValueError("Portfolio weights contain missing values.")

    if (weights < 0).any():
        raise ValueError("Portfolio weights cannot be negative.")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("Portfolio weights must sum to 1.")

    return True


def validate_portfolio_returns(portfolio_returns):
    """
    Validate portfolio return series.
    """

    if portfolio_returns.empty:
        raise ValueError("Portfolio returns are empty.")

    if portfolio_returns.isna().any():
        raise ValueError("Portfolio returns contain missing values.")

    if not np.isfinite(portfolio_returns).all():
        raise ValueError(
            "Portfolio returns contain infinite values."
        )

    return True


def validate_risk_attribution(attribution):
    """
    Validate Euler risk attribution.
    """

    required_columns = [
        "Weight",
        "Marginal Contribution",
        "Component Contribution",
        "Percentage Contribution",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in attribution.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Risk attribution is missing columns: {missing_columns}"
        )

    percentage_total = (
        attribution["Percentage Contribution"].sum()
    )

    if not np.isclose(
        percentage_total,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Risk attribution does not reconcile to 100%."
        )

    return True


def validate_factor_risk(factor_risk):
    """
    Validate factor risk attribution.
    """

    required_columns = [
        "Factor Exposure",
        "Marginal Risk",
        "Component Risk",
        "Percentage Risk",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in factor_risk.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Factor risk is missing columns: {missing_columns}"
        )

    percentage_total = (
        factor_risk["Percentage Risk"].sum()
    )

    if not np.isclose(
        percentage_total,
        1.0,
        atol=1e-10,
    ):
        raise ValueError(
            "Factor risk does not reconcile to 100%."
        )

    return True


def validate_risk_engine_results(results):
    """
    Run all major validation checks on integrated
    risk-engine results.
    """

    required_sections = [
        "Asset Returns",
        "Portfolio Returns",
        "Historical Risk",
        "Portfolio Volatility",
        "Risk Attribution",
        "Risk Concentration",
        "Factor Exposures",
        "Portfolio Factor Exposure",
        "Factor Risk",
        "Residual Risk",
        "Factor Stress",
    ]

    missing_sections = [
        section
        for section in required_sections
        if section not in results
    ]

    if missing_sections:
        raise ValueError(
            f"Risk engine is missing sections: {missing_sections}"
        )

    validate_portfolio_returns(
        results["Portfolio Returns"]
    )

    validate_risk_attribution(
        results["Risk Attribution"]
    )

    validate_factor_risk(
        results["Factor Risk"]
    )

    if results["Portfolio Volatility"] < 0:
        raise ValueError(
            "Portfolio volatility cannot be negative."
        )

    return True