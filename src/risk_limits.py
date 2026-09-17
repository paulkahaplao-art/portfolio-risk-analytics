import pandas as pd


RISK_LIMITS = {
    "Volatility": {
        "warning": 0.07,
        "limit": 0.08,
        "direction": "max",
    },
    "VaR 95%": {
        "warning": -0.015,
        "limit": -0.02,
        "direction": "min",
    },
    "Maximum Drawdown": {
        "warning": -0.08,
        "limit": -0.10,
        "direction": "min",
    },
    "International Equity Weight": {
        "warning": 0.45,
        "limit": 0.50,
        "direction": "max",
    },
    "Equity Factor Exposure": {
        "warning": 0.45,
        "limit": 0.50,
        "direction": "max",
    },
}


def check_max_limit(value, warning, limit):
    if value >= limit:
        return "BREACH"

    if value >= warning:
        return "WARNING"

    return "PASS"


def check_min_limit(value, warning, limit):
    if value <= limit:
        return "BREACH"

    if value <= warning:
        return "WARNING"

    return "PASS"


def check_risk_limit(value, warning, limit, direction):
    if direction == "max":
        return check_max_limit(value, warning, limit)

    if direction == "min":
        return check_min_limit(value, warning, limit)

    raise ValueError(
        f"Unknown limit direction: {direction}"
    )


def check_all_risk_limits(metrics, limits=None):
    if limits is None:
        limits = RISK_LIMITS

    results = []

    for metric, configuration in limits.items():
        if metric not in metrics:
            raise ValueError(
                f"Risk metric is missing: {metric}"
            )

        value = metrics[metric]

        status = check_risk_limit(
            value,
            configuration["warning"],
            configuration["limit"],
            configuration["direction"],
        )

        results.append({
            "Metric": metric,
            "Actual": value,
            "Warning": configuration["warning"],
            "Limit": configuration["limit"],
            "Status": status,
        })

    return pd.DataFrame(results)


def identify_breaches(limit_results):
    return limit_results[
        limit_results["Status"] == "BREACH"
    ].copy()


def identify_warnings(limit_results):
    return limit_results[
        limit_results["Status"] == "WARNING"
    ].copy()


def create_limit_summary(limit_results):
    breaches = identify_breaches(limit_results)
    warnings = identify_warnings(limit_results)

    if len(breaches) > 0:
        overall_status = "BREACH"
    elif len(warnings) > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    return {
        "Overall Status": overall_status,
        "Number of Breaches": len(breaches),
        "Number of Warnings": len(warnings),
        "Number of Metrics": len(limit_results),
    }

def build_risk_limit_metrics(
    risk_engine_results,
    weights,
):
    historical_risk = risk_engine_results["Historical Risk"]
    factor_exposure = risk_engine_results["Portfolio Factor Exposure"]

    metrics = {
        "Volatility": historical_risk["Volatility"],
        "VaR 95%": historical_risk["VaR 95%"],
        "Maximum Drawdown": historical_risk["Maximum Drawdown"],
        "International Equity Weight": weights["International_Equity"],
        "Equity Factor Exposure": factor_exposure["Equity"],
    }

    return metrics

def build_risk_monitoring_report(
    risk_engine_results,
    weights,
):
    metrics = build_risk_limit_metrics(
        risk_engine_results,
        weights,
    )

    limit_results = check_all_risk_limits(metrics)

    summary = create_limit_summary(limit_results)

    return {
        "Metrics": metrics,
        "Limit Results": limit_results,
        "Summary": summary,
    }