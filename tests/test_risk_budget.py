import pandas as pd
import pytest

from src.risk_budget import (
    calculate_risk_concentration,
    identify_risk_concentration,
    calculate_risk_budget_deviation,
    create_risk_budget_table,
)


@pytest.fixture
def weights():
    return pd.Series({
        "Asset_A": 0.50,
        "Asset_B": 0.30,
        "Asset_C": 0.20,
    })


@pytest.fixture
def risk_contributions():
    return pd.Series({
        "Asset_A": 0.40,
        "Asset_B": 0.50,
        "Asset_C": 0.10,
    })


def test_risk_concentration(
    weights,
    risk_contributions,
):

    result = calculate_risk_concentration(
        risk_contributions,
        weights,
    )

    assert result.loc[
        "Asset_A",
        "Risk / Capital Ratio"
    ] == pytest.approx(0.8)

    assert result.loc[
        "Asset_B",
        "Risk / Capital Ratio"
    ] == pytest.approx(5 / 3)


def test_identify_risk_concentration(
    weights,
    risk_contributions,
):

    result = identify_risk_concentration(
        risk_contributions,
        weights,
        threshold=1.5,
    )

    assert "Asset_B" in result.index
    assert "Asset_A" not in result.index


def test_risk_budget_deviation():

    result = calculate_risk_budget_deviation(
        0.60,
        0.40,
    )

    assert result == pytest.approx(0.20)


def test_risk_budget_table(
    weights,
    risk_contributions,
):

    target = pd.Series({
        "Asset_A": 0.40,
        "Asset_B": 0.40,
        "Asset_C": 0.20,
    })

    result = create_risk_budget_table(
        weights,
        risk_contributions,
        target,
    )

    assert result.loc[
        "Asset_A",
        "Risk Budget Deviation"
    ] == pytest.approx(0.00)

    assert result.loc[
        "Asset_B",
        "Risk Budget Deviation"
    ] == pytest.approx(0.10)

    assert result.loc[
        "Asset_C",
        "Risk Budget Deviation"
    ] == pytest.approx(-0.10)