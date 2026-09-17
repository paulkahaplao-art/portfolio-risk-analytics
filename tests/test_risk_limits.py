import pandas as pd
import pytest

from src.risk_limits import (
    check_max_limit,
    check_min_limit,
    check_risk_limit,
    check_all_risk_limits,
    identify_breaches,
    identify_warnings,
    create_limit_summary,
    build_risk_limit_metrics,
    build_risk_monitoring_report,
)

def test_max_limit_pass():
    assert check_max_limit(
        0.05,
        0.07,
        0.08,
    ) == "PASS"


def test_max_limit_warning():
    assert check_max_limit(
        0.075,
        0.07,
        0.08,
    ) == "WARNING"


def test_max_limit_breach():
    assert check_max_limit(
        0.09,
        0.07,
        0.08,
    ) == "BREACH"


def test_min_limit_pass():
    assert check_min_limit(
        -0.01,
        -0.015,
        -0.02,
    ) == "PASS"


def test_min_limit_warning():
    assert check_min_limit(
        -0.017,
        -0.015,
        -0.02,
    ) == "WARNING"


def test_min_limit_breach():
    assert check_min_limit(
        -0.025,
        -0.015,
        -0.02,
    ) == "BREACH"


def test_invalid_direction():
    with pytest.raises(ValueError):
        check_risk_limit(
            0.05,
            0.07,
            0.08,
            "invalid",
        )


def test_check_all_risk_limits():
    metrics = {
        "Volatility": 0.06,
        "VaR 95%": -0.01,
        "Maximum Drawdown": -0.05,
        "International Equity Weight": 0.40,
        "Equity Factor Exposure": 0.30,
    }

    results = check_all_risk_limits(metrics)

    assert len(results) == 5
    assert (results["Status"] == "PASS").all()


def test_identify_breaches():
    results = pd.DataFrame({
        "Metric": ["A", "B", "C"],
        "Actual": [0.05, 0.10, 0.06],
        "Warning": [0.07, 0.07, 0.07],
        "Limit": [0.08, 0.08, 0.08],
        "Status": ["PASS", "BREACH", "WARNING"],
    })

    breaches = identify_breaches(results)

    assert len(breaches) == 1
    assert breaches.iloc[0]["Metric"] == "B"


def test_identify_warnings():
    results = pd.DataFrame({
        "Metric": ["A", "B", "C"],
        "Actual": [0.05, 0.10, 0.06],
        "Warning": [0.07, 0.07, 0.07],
        "Limit": [0.08, 0.08, 0.08],
        "Status": ["PASS", "BREACH", "WARNING"],
    })

    warnings = identify_warnings(results)

    assert len(warnings) == 1
    assert warnings.iloc[0]["Metric"] == "C"


def test_limit_summary():
    results = pd.DataFrame({
        "Metric": ["A", "B", "C"],
        "Actual": [0.05, 0.10, 0.06],
        "Warning": [0.07, 0.07, 0.07],
        "Limit": [0.08, 0.08, 0.08],
        "Status": ["PASS", "BREACH", "WARNING"],
    })

    summary = create_limit_summary(results)

    assert summary["Overall Status"] == "BREACH"
    assert summary["Number of Breaches"] == 1
    assert summary["Number of Warnings"] == 1
    assert summary["Number of Metrics"] == 3

def test_build_risk_limit_metrics():
    risk_engine_results = {
        "Historical Risk": {
            "Volatility": 0.06,
            "VaR 95%": -0.01,
            "Maximum Drawdown": -0.05,
        },
        "Portfolio Factor Exposure": pd.Series({
            "Equity": 0.30,
            "Rates": 0.10,
            "Credit": 0.05,
        }),
    }

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    metrics = build_risk_limit_metrics(
        risk_engine_results,
        weights,
    )

    assert metrics["Volatility"] == pytest.approx(0.06)
    assert metrics["VaR 95%"] == pytest.approx(-0.01)
    assert metrics["Maximum Drawdown"] == pytest.approx(-0.05)
    assert metrics["International Equity Weight"] == pytest.approx(0.40)
    assert metrics["Equity Factor Exposure"] == pytest.approx(0.30)


def test_build_risk_monitoring_report():
    risk_engine_results = {
        "Historical Risk": {
            "Volatility": 0.06,
            "VaR 95%": -0.01,
            "Maximum Drawdown": -0.05,
        },
        "Portfolio Factor Exposure": pd.Series({
            "Equity": 0.30,
            "Rates": 0.10,
            "Credit": 0.05,
        }),
    }

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    report = build_risk_monitoring_report(
        risk_engine_results,
        weights,
    )

    assert "Metrics" in report
    assert "Limit Results" in report
    assert "Summary" in report

    assert report["Summary"]["Overall Status"] == "PASS"
    assert report["Summary"]["Number of Breaches"] == 0