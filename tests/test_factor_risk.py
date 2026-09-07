import numpy as np
import pandas as pd
import pytest

from src.factor_risk import (
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
    calculate_factor_covariance,
    calculate_factor_risk_contribution,
    calculate_factor_fitted_returns,
    calculate_residual_returns,
    calculate_factor_r_squared,
    calculate_factor_stress_return,
    calculate_factor_stress_loss,
)

def test_factor_exposure():

    factor_returns = pd.DataFrame({
        "Factor_A": [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.01,
        ],
        "Factor_B": [
            0.00,
            0.01,
            -0.01,
            0.02,
            -0.02,
        ],
    })

    asset_returns = pd.DataFrame({
        "Asset_A": (
            2 * factor_returns["Factor_A"]
            + factor_returns["Factor_B"]
        )
    })

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    assert exposures.loc[
        "Asset_A",
        "Factor_A"
    ] == pytest.approx(2.0)

    assert exposures.loc[
        "Asset_A",
        "Factor_B"
    ] == pytest.approx(1.0)


def test_portfolio_factor_exposure():

    exposures = pd.DataFrame({
        "Factor_A": [1.0, 2.0],
        "Factor_B": [0.5, 1.0],
    }, index=["Asset_A", "Asset_B"])

    weights = pd.Series({
        "Asset_A": 0.60,
        "Asset_B": 0.40,
    })

    result = calculate_portfolio_factor_exposure(
        exposures,
        weights,
    )

    assert result["Factor_A"] == pytest.approx(1.4)
    assert result["Factor_B"] == pytest.approx(0.7)


def test_factor_covariance():

    factor_returns = pd.DataFrame({
        "Factor_A": [0.01, -0.01, 0.02],
        "Factor_B": [0.02, 0.00, -0.01],
    })

    covariance = calculate_factor_covariance(
        factor_returns
    )

    assert covariance.shape == (2, 2)
    assert np.all(
        np.diag(covariance) >= 0
    )


def test_factor_risk_reconciles():

    exposure = pd.Series({
        "Factor_A": 1.0,
        "Factor_B": 0.5,
    })

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=["Factor_A", "Factor_B"],
        columns=["Factor_A", "Factor_B"],
    )

    result = calculate_factor_risk_contribution(
        exposure,
        covariance,
    )

    assert result[
        "Percentage Risk"
    ].sum() == pytest.approx(1.0)

def test_factor_risk_components_reconcile():

    exposure = pd.Series({
        "Factor_A": 1.0,
        "Factor_B": 0.5,
    })

    covariance = pd.DataFrame(
        [
            [0.04, 0.01],
            [0.01, 0.09],
        ],
        index=["Factor_A", "Factor_B"],
        columns=["Factor_A", "Factor_B"],
    )

    result = calculate_factor_risk_contribution(
        exposure,
        covariance,
    )

    portfolio_volatility = np.sqrt(
        exposure.T
        @ covariance
        @ exposure
    )

    assert result[
        "Component Risk"
    ].sum() == pytest.approx(
        portfolio_volatility
    )

def test_factor_fitted_returns():

    factor_returns = pd.DataFrame({
        "Factor_A": [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.01,
        ],
    })

    asset_returns = pd.DataFrame({
        "Asset_A": (
            2 * factor_returns["Factor_A"]
        )
    })

    fitted = calculate_factor_fitted_returns(
        asset_returns,
        factor_returns,
    )

    assert fitted["Asset_A"].tolist() == pytest.approx(
        asset_returns["Asset_A"].tolist()
    )

def test_residual_returns():

    factor_returns = pd.DataFrame({
        "Factor_A": [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.01,
        ],
    })

    residual_component = pd.Series(
        [0.001, -0.001, 0.002, -0.002, 0.001]
    )

    asset_returns = pd.DataFrame({
        "Asset_A": (
            2 * factor_returns["Factor_A"]
            + residual_component
        )
    })

    residuals = calculate_residual_returns(
        asset_returns,
        factor_returns,
    )

    assert residuals["Asset_A"].std() > 0

def test_factor_r_squared():

    factor_returns = pd.DataFrame({
        "Factor_A": [
            0.01,
            -0.01,
            0.02,
            -0.02,
            0.01,
        ],
    })

    asset_returns = pd.DataFrame({
        "Asset_A": (
            2 * factor_returns["Factor_A"]
        )
    })

    r_squared = calculate_factor_r_squared(
        asset_returns,
        factor_returns,
    )

    assert r_squared["Asset_A"] == pytest.approx(
        1.0
    )

def test_factor_stress_return():

    exposure = pd.Series({
        "Equity": 1.0,
        "Rates": -0.5,
        "Credit": 0.25,
    })

    shock = {
        "Equity": -0.20,
        "Rates": -0.10,
        "Credit": -0.20,
    }

    result = calculate_factor_stress_return(
        exposure,
        shock,
    )

    expected = (
        1.0 * -0.20
        + (-0.5) * -0.10
        + 0.25 * -0.20
    )

    assert result == pytest.approx(
        expected
    )

def test_factor_stress_loss():

    result = calculate_factor_stress_loss(
        100_000,
        -0.25,
    )

    assert result == pytest.approx(
        -25_000
    )

def test_factor_stress_missing_factor():

    exposure = pd.Series({
        "Equity": 1.0,
        "Rates": 0.5,
        "Credit": 0.25,
    })

    incomplete_shock = {
        "Equity": -0.20,
        "Rates": -0.10,
    }

    with pytest.raises(ValueError):

        calculate_factor_stress_return(
            exposure,
            incomplete_shock,
        )