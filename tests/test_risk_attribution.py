import numpy as np
import pandas as pd
import pytest

from src.risk_attribution import (
    calculate_portfolio_volatility,
    calculate_marginal_contribution_to_risk,
    calculate_component_contribution_to_risk,
    calculate_percentage_contribution_to_risk,
    calculate_diversification_benefit,
)

from src.portfolio_risk_attribution import (
    validate_risk_attribution,
)

@pytest.fixture
def weights():
    return pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.30,
        "Asset_C": 0.20,
    })


@pytest.fixture
def covariance():
    return pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.01],
            [0.00, 0.01, 0.01],
        ],
        index=["Asset_A", "Asset_B", "Asset_C"],
        columns=["Asset_A", "Asset_B", "Asset_C"],
    )


def test_portfolio_volatility(weights, covariance):

    volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    assert volatility > 0


def test_marginal_contribution(weights, covariance):

    marginal = calculate_marginal_contribution_to_risk(
        weights,
        covariance,
    )

    assert len(marginal) == 3
    assert all(np.isfinite(marginal))


def test_component_contribution_reconciles(
    weights,
    covariance,
):

    components = calculate_component_contribution_to_risk(
        weights,
        covariance,
    )

    portfolio_volatility = calculate_portfolio_volatility(
        weights,
        covariance,
    )

    assert components.sum() == pytest.approx(
        portfolio_volatility
    )


def test_percentage_contribution_reconciles(
    weights,
    covariance,
):

    percentages = calculate_percentage_contribution_to_risk(
        weights,
        covariance,
    )

    assert percentages.sum() == pytest.approx(1.0)


def test_diversification_benefit(
    weights,
    covariance,
):

    benefit = calculate_diversification_benefit(
        weights,
        covariance,
    )

    assert benefit >= 0

def test_validate_risk_attribution(
    weights,
    covariance,
):

    from src.risk_attribution import (
        create_risk_attribution_table,
    )

    attribution = create_risk_attribution_table(
        weights,
        covariance,
    )

    assert validate_risk_attribution(
        attribution
    ) is True