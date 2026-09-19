import pandas as pd
import pytest

from src.risk_monitoring_history import (
    calculate_rolling_drawdown,
    calculate_rolling_risk_monitor,
    calculate_historical_limit_utilisation,
    classify_historical_risk_status,
    add_historical_risk_status,
    identify_historical_breaches,
    identify_historical_warnings,
    create_historical_monitoring_summary,
)


def test_calculate_rolling_drawdown():
    returns = pd.Series(
        [0.10, -0.05, -0.10],
        index=pd.date_range(
            "2025-01-01",
            periods=3,
        ),
    )

    drawdown = calculate_rolling_drawdown(
        returns
    )

    assert drawdown.iloc[0] == pytest.approx(0.0)
    assert drawdown.iloc[1] < 0
    assert drawdown.iloc[2] < drawdown.iloc[1]


def test_calculate_rolling_risk_monitor():
    returns = pd.Series(
        [0.01, -0.01] * 150,
        index=pd.date_range(
            "2025-01-01",
            periods=300,
        ),
    )

    result = calculate_rolling_risk_monitor(
        returns,
        volatility_window=20,
        var_window=50,
    )

    assert "Rolling Volatility" in result.columns
    assert "Rolling VaR" in result.columns
    assert "Drawdown" in result.columns

    assert len(result) == 300


def test_limit_utilisation():
    risk_history = pd.DataFrame({
        "Rolling Volatility": [0.04],
        "Rolling VaR": [-0.01],
        "Drawdown": [-0.05],
    })

    result = calculate_historical_limit_utilisation(
        risk_history
    )

    assert result.loc[
        0,
        "Volatility Utilisation",
    ] == pytest.approx(0.50)

    assert result.loc[
        0,
        "VaR Utilisation",
    ] == pytest.approx(0.50)

    assert result.loc[
        0,
        "Drawdown Utilisation",
    ] == pytest.approx(0.50)


def test_classify_pass():
    status = classify_historical_risk_status(
        0.04,
        -0.01,
        -0.05,
    )

    assert status == "PASS"


def test_classify_warning():
    status = classify_historical_risk_status(
        0.075,
        -0.01,
        -0.05,
    )

    assert status == "WARNING"


def test_classify_breach():
    status = classify_historical_risk_status(
        0.09,
        -0.01,
        -0.05,
    )

    assert status == "BREACH"


def test_add_historical_risk_status():
    risk_history = pd.DataFrame({
        "Rolling Volatility": [
            0.04,
            0.075,
            0.09,
        ],
        "Rolling VaR": [
            -0.01,
            -0.01,
            -0.01,
        ],
        "Drawdown": [
            -0.05,
            -0.05,
            -0.05,
        ],
    })

    result = add_historical_risk_status(
        risk_history
    )

    assert list(result["Status"]) == [
        "PASS",
        "WARNING",
        "BREACH",
    ]


def test_identify_breaches():
    risk_history = pd.DataFrame({
        "Rolling Volatility": [0.04, 0.09],
        "Rolling VaR": [-0.01, -0.01],
        "Drawdown": [-0.05, -0.05],
        "Status": ["PASS", "BREACH"],
    })

    breaches = identify_historical_breaches(
        risk_history
    )

    assert len(breaches) == 1


def test_identify_warnings():
    risk_history = pd.DataFrame({
        "Rolling Volatility": [0.04, 0.075],
        "Rolling VaR": [-0.01, -0.01],
        "Drawdown": [-0.05, -0.05],
        "Status": ["PASS", "WARNING"],
    })

    warnings = identify_historical_warnings(
        risk_history
    )

    assert len(warnings) == 1


def test_historical_monitoring_summary():
    risk_history = pd.DataFrame({
        "Rolling Volatility": [0.04, 0.09],
        "Rolling VaR": [-0.01, -0.03],
        "Drawdown": [-0.05, -0.12],
        "Status": ["PASS", "BREACH"],
    })

    summary = create_historical_monitoring_summary(
        risk_history
    )

    assert summary["Number of Breaches"] == 1
    assert summary["Observations"] == 2
    assert summary["Maximum Volatility"] == pytest.approx(
        0.09
    )
    assert summary["Worst VaR"] == pytest.approx(
        -0.03
    )
    assert summary["Maximum Drawdown"] == pytest.approx(
        -0.12
    )