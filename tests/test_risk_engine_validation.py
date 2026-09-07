import numpy as np
import pandas as pd
import pytest

from src.risk_engine_validation import (
    validate_weights,
    validate_portfolio_returns,
    validate_risk_attribution,
    validate_factor_risk,
)


def test_valid_weights():

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    assert validate_weights(weights) is True


def test_weights_must_sum_to_one():

    weights = pd.Series({
        "Asset_A": 0.60,
        "Asset_B": 0.50,
    })

    with pytest.raises(ValueError):
        validate_weights(weights)


def test_negative_weights_rejected():

    weights = pd.Series({
        "Asset_A": 1.10,
        "Asset_B": -0.10,
    })

    with pytest.raises(ValueError):
        validate_weights(weights)


def test_valid_portfolio_returns():

    returns = pd.Series([
        0.01,
        -0.02,
        0.005,
    ])

    assert validate_portfolio_returns(returns) is True


def test_missing_returns_rejected():

    returns = pd.Series([
        0.01,
        np.nan,
        0.005,
    ])

    with pytest.raises(ValueError):
        validate_portfolio_returns(returns)


def test_risk_attribution_reconciles():

    attribution = pd.DataFrame({
        "Weight": [0.50, 0.50],
        "Marginal Contribution": [0.10, 0.10],
        "Component Contribution": [0.05, 0.05],
        "Percentage Contribution": [0.50, 0.50],
    })

    assert validate_risk_attribution(
        attribution
    ) is True


def test_risk_attribution_failure():

    attribution = pd.DataFrame({
        "Weight": [0.50, 0.50],
        "Marginal Contribution": [0.10, 0.10],
        "Component Contribution": [0.05, 0.04],
        "Percentage Contribution": [0.50, 0.40],
    })

    with pytest.raises(ValueError):
        validate_risk_attribution(attribution)


def test_factor_risk_reconciles():

    factor_risk = pd.DataFrame({
        "Factor Exposure": [0.5, 0.3],
        "Marginal Risk": [0.10, 0.20],
        "Component Risk": [0.05, 0.05],
        "Percentage Risk": [0.50, 0.50],
    })

    assert validate_factor_risk(
        factor_risk
    ) is True


def test_factor_risk_failure():

    factor_risk = pd.DataFrame({
        "Factor Exposure": [0.5, 0.3],
        "Marginal Risk": [0.10, 0.20],
        "Component Risk": [0.05, 0.03],
        "Percentage Risk": [0.50, 0.30],
    })

    with pytest.raises(ValueError):
        validate_factor_risk(factor_risk)