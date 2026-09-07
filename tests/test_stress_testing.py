import pandas as pd
import pytest

from src.stress_testing import (
    calculate_scenario_return,
    calculate_scenario_loss,
    classify_scenario_severity,
    run_stress_scenarios,
    identify_worst_days,
    calculate_day_contributions,
    create_historical_stress_scenario,
    rank_stress_scenarios,
    create_stress_summary,
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

def test_missing_scenario_asset_raises_error(weights):
    incomplete_scenario = {
        "Australian_Equity": -0.10,
        "International_Equity": -0.20,
        "Bonds": -0.05,
    }

    with pytest.raises(ValueError):
        calculate_scenario_return(
            weights,
            incomplete_scenario,
        )

def test_rank_stress_scenarios():

    results = pd.DataFrame({
        "Scenario": [
            "Scenario A",
            "Scenario B",
            "Scenario C",
        ],
        "Portfolio Return": [
            -0.05,
            -0.20,
            -0.10,
        ],
        "Portfolio P&L": [
            -5_000,
            -20_000,
            -10_000,
        ],
        "Severity": [
            "Moderate",
            "High",
            "Moderate",
        ],
    })

    ranked = rank_stress_scenarios(results)

    assert ranked.iloc[0]["Scenario"] == "Scenario B"
    assert ranked.iloc[0]["Portfolio P&L"] == -20_000
    assert ranked.iloc[-1]["Scenario"] == "Scenario A"


def test_create_stress_summary():

    results = pd.DataFrame({
        "Scenario": [
            "Scenario A",
            "Scenario B",
        ],
        "Portfolio Return": [
            -0.05,
            -0.20,
        ],
        "Portfolio P&L": [
            -5_000,
            -20_000,
        ],
        "Severity": [
            "Moderate",
            "High",
        ],
    })

    summary = create_stress_summary(results)

    assert summary["Worst Scenario"] == "Scenario B"
    assert summary["Worst Return"] == -0.20
    assert summary["Worst P&L"] == -20_000
    assert summary["Worst Severity"] == "High"
    assert summary["Scenario Count"] == 2