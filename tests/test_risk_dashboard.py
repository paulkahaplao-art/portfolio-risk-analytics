import pandas as pd
import pytest

from src.risk_dashboard import (
    create_risk_dashboard,
    calculate_limit_utilisation,
    create_breach_report,
    create_warning_report,
    create_factor_dashboard,
    create_stress_dashboard,
)


def create_test_results():
    return {
        "Historical Risk": {
            "Volatility": 0.06,
            "VaR 95%": -0.01,
            "VaR 99%": -0.015,
            "ES 95%": -0.012,
            "ES 99%": -0.018,
            "Maximum Drawdown": -0.05,
        },
        "Risk Monitoring": {
            "Limit Results": pd.DataFrame({
                "Metric": [
                    "Volatility",
                    "VaR 95%",
                    "Maximum Drawdown",
                ],
                "Actual": [
                    0.06,
                    -0.01,
                    -0.05,
                ],
                "Warning": [
                    0.07,
                    -0.015,
                    -0.08,
                ],
                "Limit": [
                    0.08,
                    -0.02,
                    -0.10,
                ],
                "Status": [
                    "PASS",
                    "PASS",
                    "PASS",
                ],
            }),
            "Summary": {
                "Overall Status": "PASS",
                "Number of Breaches": 0,
                "Number of Warnings": 0,
            },
        },
        "Factor Risk": pd.DataFrame({
            "Factor Exposure": [0.30, 0.10, 0.05],
            "Marginal Risk": [0.20, 0.05, 0.02],
            "Component Risk": [0.06, 0.005, 0.001],
            "Percentage Risk": [0.90, 0.08, 0.02],
        }, index=["Equity", "Rates", "Credit"]),
        "Factor Stress": pd.DataFrame({
            "Scenario": [
                "Equity Shock",
                "Rates Shock",
            ],
            "Portfolio Return": [
                -0.06,
                -0.01,
            ],
            "Portfolio P&L": [
                -6000,
                -1000,
            ],
        }),
    }


def test_create_risk_dashboard():
    results = create_test_results()

    dashboard = create_risk_dashboard(results)

    assert len(dashboard) == 9
    assert "Metric" in dashboard.columns
    assert "Value" in dashboard.columns


def test_calculate_limit_utilisation():
    limit_results = pd.DataFrame({
        "Metric": ["Volatility", "VaR 95%"],
        "Actual": [0.06, -0.01],
        "Warning": [0.07, -0.015],
        "Limit": [0.08, -0.02],
        "Status": ["PASS", "PASS"],
    })

    result = calculate_limit_utilisation(limit_results)

    assert result.loc[0, "Limit Utilisation"] == pytest.approx(0.75)
    assert result.loc[1, "Limit Utilisation"] == pytest.approx(0.50)


def test_create_breach_report():
    results = create_test_results()

    report = create_breach_report(
        results["Risk Monitoring"]["Limit Results"]
    )

    assert len(report) == 0


def test_create_warning_report():
    results = create_test_results()

    report = create_warning_report(
        results["Risk Monitoring"]["Limit Results"]
    )

    assert len(report) == 0


def test_create_factor_dashboard():
    results = create_test_results()

    dashboard = create_factor_dashboard(results)

    assert len(dashboard) == 3
    assert "Factor" in dashboard.columns
    assert "Factor Exposure" in dashboard.columns


def test_create_stress_dashboard():
    results = create_test_results()

    dashboard = create_stress_dashboard(results)

    assert len(dashboard) == 2
    assert "Scenario" in dashboard.columns
    assert "Portfolio P&L" in dashboard.columns