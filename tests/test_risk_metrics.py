import numpy as np
import pandas as pd

from src.risk_metrics import (
    maximum_drawdown,
    historical_var,
    expected_shortfall,
)

from src.portfolio_risk import (
    calculate_risk_contributions,
    calculate_benchmark_returns,
    calculate_active_returns,
    calculate_tracking_error,
    calculate_information_ratio,
    calculate_rolling_volatility,
    calculate_rolling_tracking_error,
    calculate_rolling_active_return,
    create_risk_report,
)

def test_maximum_drawdown():
    growth = pd.Series([
        1.00,
        1.10,
        1.05,
        0.90,
        1.00,
    ])

    result = maximum_drawdown(growth)

    expected = 0.90 / 1.10 - 1

    assert abs(result - expected) < 1e-10


def test_historical_var():
    returns = pd.Series([
        -0.10,
        -0.05,
        -0.02,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
    ])

    result = historical_var(
        returns,
        confidence_level=0.90
    )

    assert result > 0


def test_expected_shortfall():
    returns = pd.Series([
        -0.10,
        -0.05,
        -0.02,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
        0.06,
        0.07,
    ])

    result = expected_shortfall(
        returns,
        confidence_level=0.90
    )

    var = historical_var(
        returns,
        confidence_level=0.90
    )

    assert result > 0
    assert result >= var

def test_risk_contributions_reconcile():

    asset_returns = pd.DataFrame({
        "Australian_Equity": [
            0.01,
            -0.005,
            0.015,
            0.01,
        ],
        "International_Equity": [
            0.005,
            0.005,
            0.008,
            0.004,
        ],
        "Bonds": [
            0.001,
            0.001,
            -0.0005,
            0.0015,
        ],
        "Cash": [
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
    })

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    result = calculate_risk_contributions(
        asset_returns,
        weights
    )

    total_contribution = (
        result["Component Contribution"].sum()
    )

    covariance_matrix = asset_returns.cov()

    portfolio_variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = (
    np.sqrt(portfolio_variance)
    * np.sqrt(252)
    )

    assert abs(
        total_contribution - portfolio_volatility
    ) < 1e-10

def test_benchmark_returns():

    asset_returns = pd.DataFrame({
        "Australian_Equity": [0.01, 0.02],
        "International_Equity": [0.03, 0.01],
        "Bonds": [0.001, 0.002],
        "Cash": [0.0001, 0.0001],
    })

    result = calculate_benchmark_returns(
        asset_returns
    )

    expected = pd.Series([
        0.02,
        0.015,
    ])

    pd.testing.assert_series_equal(
        result,
        expected
    )

def test_active_returns():

    portfolio = pd.Series([
        0.010,
        0.020,
        0.015,
    ])

    benchmark = pd.Series([
        0.005,
        0.010,
        0.020,
    ])

    result = calculate_active_returns(
        portfolio,
        benchmark
    )

    expected = pd.Series([
        0.005,
        0.010,
        -0.005,
    ])

    pd.testing.assert_series_equal(
        result,
        expected
    )

def test_tracking_error():

    portfolio = pd.Series([
        0.010,
        0.020,
        0.015,
        0.005,
    ])

    benchmark = pd.Series([
        0.005,
        0.010,
        0.020,
        0.010,
    ])

    result = calculate_tracking_error(
        portfolio,
        benchmark
    )

    active = portfolio - benchmark

    expected = active.std() * (252 ** 0.5)

    assert abs(result - expected) < 1e-10

def test_information_ratio():

    portfolio = pd.Series([
        0.010,
        0.020,
        0.015,
        0.005,
    ])

    benchmark = pd.Series([
        0.005,
        0.010,
        0.020,
        0.010,
    ])

    result = calculate_information_ratio(
        portfolio,
        benchmark
    )

    active = portfolio - benchmark

    annualised_active_return = active.mean() * 252
    tracking_error = active.std() * (252 ** 0.5)

    expected = annualised_active_return / tracking_error if tracking_error != 0 else float("nan")

    assert abs(result - expected) < 1e-10

def test_rolling_volatility():

    returns = pd.Series([
        0.01,
        0.02,
        0.015,
        0.005,
        0.01,
        0.02,
    ])

    result = calculate_rolling_volatility(
        returns,
        window=3
    )

    expected = (
        returns
        .rolling(3)
        .std()
        * (252 ** 0.5)
    )

    pd.testing.assert_series_equal(
        result,
        expected
    )

def test_rolling_tracking_error():

    portfolio = pd.Series([
        0.010,
        0.020,
        0.015,
        0.005,
        0.010,
    ])

    benchmark = pd.Series([
        0.005,
        0.010,
        0.020,
        0.010,
        0.005,
    ])

    result = calculate_rolling_tracking_error(
        portfolio,
        benchmark,
        window=3
    )

    active = portfolio - benchmark

    expected = (
        active
        .rolling(3)
        .std()
        * (252 ** 0.5)
    )

    pd.testing.assert_series_equal(
        result,
        expected
    )

def test_rolling_active_return():

    portfolio = pd.Series([
        0.010,
        0.020,
        0.015,
        0.005,
        0.010,
    ])

    benchmark = pd.Series([
        0.005,
        0.010,
        0.020,
        0.010,
        0.005,
    ])

    result = calculate_rolling_active_return(
        portfolio,
        benchmark,
        window=3
    )

    active = portfolio - benchmark

    expected = (
        active
        .rolling(3)
        .mean()
        * 252
    )

    pd.testing.assert_series_equal(
        result,
        expected
    )


def test_create_risk_report():

    asset_returns = pd.DataFrame({
        "Australian_Equity": [
            0.010,
            0.020,
            0.015,
            0.005,
            0.010,
        ],
        "International_Equity": [
            0.005,
            0.010,
            0.020,
            0.010,
            0.005,
        ],
        "Bonds": [
            0.001,
            0.001,
            0.001,
            0.001,
            0.001,
        ],
        "Cash": [
            0.0001,
            0.0001,
            0.0001,
            0.0001,
            0.0001,
        ],
    })

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    portfolio_returns = (
        asset_returns * weights
    ).sum(axis=1)

    benchmark_returns = pd.Series([
        0.005,
        0.010,
        0.020,
        0.010,
        0.005,
    ])

    report = create_risk_report(
        asset_returns,
        weights,
        portfolio_returns,
        benchmark_returns,
        1_000_000,
    )

    expected_columns = [
        "Annualised Volatility",
        "Maximum Drawdown",
        "95% VaR",
        "99% VaR",
        "95% Expected Shortfall",
        "95% VaR ($)",
        "99% VaR ($)",
        "95% Expected Shortfall ($)",
        "Tracking Error",
        "Information Ratio",
    ]

    for column in expected_columns:
        assert column in report.index