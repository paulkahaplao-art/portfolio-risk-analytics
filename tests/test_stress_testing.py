import pandas as pd
import pytest

from src.stress_testing import (
    calculate_scenario_return,
    calculate_scenario_loss,
    classify_scenario_severity,
    identify_worst_days,
    calculate_day_contributions,
)


@pytest.fixture
def weights():
    return pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })


def test_calculate_scenario_return(weights):
    scenario = {
        "Australian_Equity": -0.10,
        "International_Equity": -0.20,
        "Bonds": -0.05,
        "Cash": 0.00,
    }

    result = calculate_scenario_return(
        weights,
        scenario,
    )

    expected = (
        0.30 * -0.10
        + 0.40 * -0.20
        + 0.20 * -0.05
        + 0.10 * 0.00
    )

    assert result == pytest.approx(expected)


def test_calculate_scenario_loss():
    result = calculate_scenario_loss(
        100_000,
        -0.20,
    )

    assert result == pytest.approx(-20_000)


def test_classify_scenario_severity():
    assert classify_scenario_severity(-0.02) == "Low"
    assert classify_scenario_severity(-0.10) == "Moderate"
    assert classify_scenario_severity(-0.20) == "High"
    assert classify_scenario_severity(-0.30) == "Severe"


def test_identify_worst_days(weights):
    returns = pd.DataFrame(
        {
            "Australian_Equity": [
                0.01,
                -0.05,
                0.02,
            ],
            "International_Equity": [
                0.02,
                -0.10,
                0.01,
            ],
            "Bonds": [
                0.00,
                -0.02,
                0.01,
            ],
            "Cash": [
                0.001,
                0.001,
                0.001,
            ],
        },
        index=pd.to_datetime([
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
        ]),
    )

    result = identify_worst_days(
        returns,
        weights,
        n=1,
    )

    assert len(result) == 1
    assert result.index[0] == pd.Timestamp("2025-01-02")


def test_day_contributions(weights):
    returns = pd.DataFrame(
        {
            "Australian_Equity": [-0.10],
            "International_Equity": [-0.20],
            "Bonds": [-0.05],
            "Cash": [0.00],
        },
        index=pd.to_datetime(["2025-01-01"]),
    )

    contributions = calculate_day_contributions(
        returns,
        weights,
        pd.Timestamp("2025-01-01"),
    )

    assert contributions["Australian_Equity"] == pytest.approx(-0.03)
    assert contributions["International_Equity"] == pytest.approx(-0.08)
    assert contributions["Bonds"] == pytest.approx(-0.01)
    assert contributions["Cash"] == pytest.approx(0.00)

    assert contributions.sum() == pytest.approx(-0.12)