import pandas as pd

from src.historical_analysis import (
    calculate_rolling_volatility,
    calculate_rolling_var,
    calculate_maximum_drawdown,
    classify_risk_regime,
)

def add_risk_regime(
    risk_history,
    low_threshold=0.08,
    high_threshold=0.15,
):
    result = risk_history.copy()

    result["Risk Regime"] = classify_risk_regime(
        result["Rolling Volatility"],
        low_threshold=low_threshold,
        high_threshold=high_threshold,
    )

    return result

def calculate_rolling_drawdown(portfolio_returns):
    wealth = (1 + portfolio_returns).cumprod()

    running_max = wealth.cummax()

    drawdown = wealth / running_max - 1

    return drawdown


def calculate_rolling_risk_monitor(
    portfolio_returns,
    volatility_window=63,
    var_window=252,
    var_confidence=0.95,
):
    rolling_volatility = calculate_rolling_volatility(
        portfolio_returns,
        window=volatility_window,
    )

    rolling_var = calculate_rolling_var(
        portfolio_returns,
        window=var_window,
        confidence=var_confidence,
    )

    rolling_drawdown = calculate_rolling_drawdown(
        portfolio_returns,
    )

    result = pd.DataFrame({
        "Rolling Volatility": rolling_volatility,
        "Rolling VaR": rolling_var,
        "Drawdown": rolling_drawdown,
    })

    return result


def calculate_historical_limit_utilisation(
    risk_history,
    volatility_limit=0.08,
    var_limit=-0.02,
    drawdown_limit=-0.10,
):
    result = risk_history.copy()

    result["Volatility Utilisation"] = (
        result["Rolling Volatility"].abs()
        / abs(volatility_limit)
    )

    result["VaR Utilisation"] = (
        result["Rolling VaR"].abs()
        / abs(var_limit)
    )

    result["Drawdown Utilisation"] = (
        result["Drawdown"].abs()
        / abs(drawdown_limit)
    )

    return result


def classify_historical_risk_status(
    volatility,
    var,
    drawdown,
    volatility_limit=0.08,
    var_limit=-0.02,
    drawdown_limit=-0.10,
):
    if (
        volatility >= volatility_limit
        or var <= var_limit
        or drawdown <= drawdown_limit
    ):
        return "BREACH"

    if (
        volatility >= volatility_limit * 0.875
        or var <= var_limit * 0.875
        or drawdown <= drawdown_limit * 0.875
    ):
        return "WARNING"

    return "PASS"


def add_historical_risk_status(
    risk_history,
    volatility_limit=0.08,
    var_limit=-0.02,
    drawdown_limit=-0.10,
):
    result = risk_history.copy()

    result["Status"] = result.apply(
        lambda row: classify_historical_risk_status(
            row["Rolling Volatility"],
            row["Rolling VaR"],
            row["Drawdown"],
            volatility_limit,
            var_limit,
            drawdown_limit,
        ),
        axis=1,
    )

    return result


def identify_historical_breaches(risk_history):
    return risk_history[
        risk_history["Status"] == "BREACH"
    ].copy()


def identify_historical_warnings(risk_history):
    return risk_history[
        risk_history["Status"] == "WARNING"
    ].copy()


def create_historical_monitoring_summary(risk_history):
    breaches = identify_historical_breaches(
        risk_history
    )

    warnings = identify_historical_warnings(
        risk_history
    )

    return {
        "Number of Breaches": len(breaches),
        "Number of Warnings": len(warnings),
        "Observations": len(risk_history),
        "Maximum Volatility": risk_history[
            "Rolling Volatility"
        ].max(),
        "Worst VaR": risk_history[
            "Rolling VaR"
        ].min(),
        "Maximum Drawdown": risk_history[
            "Drawdown"
        ].min(),
    }