import numpy as np
import pandas as pd

import pytest

from src.portfolio_dynamics import (
    calculate_rebalanced_returns,
    calculate_buy_and_hold_values,
    calculate_buy_and_hold_returns,
    calculate_buy_and_hold_weights,
)


def test_rebalanced_returns():

    returns = pd.DataFrame({
        "Asset_A": [0.10, 0.10],
        "Asset_B": [0.00, 0.00],
    })

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    portfolio_returns = calculate_rebalanced_returns(
        returns,
        weights,
    )

    expected = pd.Series(
        [0.05, 0.05],
        index=returns.index,
    )

    pd.testing.assert_series_equal(
        portfolio_returns,
        expected,
        check_names=False,
    )


def test_buy_and_hold_value():

    returns = pd.DataFrame({
        "Asset_A": [0.10],
        "Asset_B": [0.00],
    })

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    values = calculate_buy_and_hold_values(
        returns,
        weights,
        initial_value=100_000,
    )

    assert values.iloc[0] == 105_000


def test_buy_and_hold_weights_drift():

    returns = pd.DataFrame({
        "Asset_A": [0.10],
        "Asset_B": [0.00],
    })

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    result = calculate_buy_and_hold_weights(
        returns,
        weights,
        initial_value=100_000,
    )

    assert result.iloc[0]["Asset_A"] == pytest.approx(
        55 / 105
    )

    assert result.iloc[0]["Asset_B"] == pytest.approx(
        50 / 105
    )


def test_buy_and_hold_weights_sum_to_one():

    returns = pd.DataFrame({
        "Asset_A": [0.10, -0.05],
        "Asset_B": [0.00, 0.02],
    })

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    result = calculate_buy_and_hold_weights(
        returns,
        weights,
    )

    assert np.allclose(
        result.sum(axis=1),
        1.0,
    )


def test_buy_and_hold_returns():

    returns = pd.DataFrame({
        "Asset_A": [0.10, 0.10],
        "Asset_B": [0.00, 0.00],
    })

    weights = pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.50,
    })

    result = calculate_buy_and_hold_returns(
        returns,
        weights,
        initial_value=100_000,
    )

    assert result.iloc[0] == pytest.approx(
        105_000 / 100_000 - 1
    )

    assert result.iloc[1] == pytest.approx(
        110_500 / 105_000 - 1
    )