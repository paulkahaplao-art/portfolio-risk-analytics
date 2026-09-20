import pandas as pd
import pytest

from src.risk_alerts import (
    create_alert_message,
    create_risk_alerts,
    filter_active_alerts,
    create_alert_summary,
    create_alert_report,
)


def test_create_alert_message_breach():
    message = create_alert_message(
        "Volatility",
        0.09,
        0.07,
        0.08,
        "BREACH",
    )

    assert "breached" in message
    assert "Volatility" in message


def test_create_alert_message_warning():
    message = create_alert_message(
        "Volatility",
        0.075,
        0.07,
        0.08,
        "WARNING",
    )

    assert "approaching" in message


def test_create_alert_message_pass():
    message = create_alert_message(
        "Volatility",
        0.05,
        0.07,
        0.08,
        "PASS",
    )

    assert "within" in message


def test_create_risk_alerts():
    limit_results = pd.DataFrame({
        "Metric": [
            "Volatility",
            "VaR 95%",
            "Drawdown",
        ],
        "Actual": [
            0.09,
            -0.015,
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
            "BREACH",
            "WARNING",
            "PASS",
        ],
    })

    alerts = create_risk_alerts(
        limit_results
    )

    assert len(alerts) == 3
    assert "Message" in alerts.columns
    assert "Limit Utilisation" in alerts.columns

    assert alerts.loc[
        0,
        "Limit Utilisation",
    ] == pytest.approx(1.125)


def test_filter_active_alerts():
    alerts = pd.DataFrame({
        "Metric": [
            "A",
            "B",
            "C",
        ],
        "Status": [
            "PASS",
            "WARNING",
            "BREACH",
        ],
    })

    active = filter_active_alerts(
        alerts
    )

    assert len(active) == 2


def test_create_alert_summary():
    alerts = pd.DataFrame({
        "Metric": [
            "A",
            "B",
            "C",
        ],
        "Status": [
            "PASS",
            "WARNING",
            "BREACH",
        ],
    })

    summary = create_alert_summary(
        alerts
    )

    assert summary["Overall Status"] == "BREACH"
    assert summary["Breaches"] == 1
    assert summary["Warnings"] == 1
    assert summary["Active Alerts"] == 2


def test_create_alert_report():
    limit_results = pd.DataFrame({
        "Metric": [
            "Volatility",
            "VaR 95%",
        ],
        "Actual": [
            0.09,
            -0.01,
        ],
        "Warning": [
            0.07,
            -0.015,
        ],
        "Limit": [
            0.08,
            -0.02,
        ],
        "Status": [
            "BREACH",
            "PASS",
        ],
    })

    report = create_alert_report(
        limit_results
    )

    assert "Alerts" in report
    assert "Active Alerts" in report
    assert "Summary" in report

    assert len(
        report["Active Alerts"]
    ) == 1