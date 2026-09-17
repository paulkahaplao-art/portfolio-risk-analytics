import pandas as pd


def create_risk_dashboard(risk_engine_results):
    historical_risk = risk_engine_results["Historical Risk"]
    monitoring = risk_engine_results["Risk Monitoring"]

    limit_results = monitoring["Limit Results"]
    summary = monitoring["Summary"]

    dashboard = pd.DataFrame({
        "Metric": [
            "Overall Risk Status",
            "Annualised Volatility",
            "95% VaR",
            "99% VaR",
            "95% Expected Shortfall",
            "99% Expected Shortfall",
            "Maximum Drawdown",
            "Risk Limit Breaches",
            "Risk Limit Warnings",
        ],
        "Value": [
            summary["Overall Status"],
            historical_risk["Volatility"],
            historical_risk["VaR 95%"],
            historical_risk["VaR 99%"],
            historical_risk["ES 95%"],
            historical_risk["ES 99%"],
            historical_risk["Maximum Drawdown"],
            summary["Number of Breaches"],
            summary["Number of Warnings"],
        ],
    })

    return dashboard


def calculate_limit_utilisation(limit_results):
    utilisation = limit_results.copy()

    def calculate_row(row):
        actual = row["Actual"]
        warning = row["Warning"]
        limit = row["Limit"]

        if limit == 0:
            return float("nan")

        if limit > 0:
            return actual / limit

        return abs(actual / limit)

    utilisation["Limit Utilisation"] = utilisation.apply(
        calculate_row,
        axis=1,
    )

    return utilisation


def create_breach_report(limit_results):
    breaches = limit_results[
        limit_results["Status"] == "BREACH"
    ].copy()

    return breaches[
        [
            "Metric",
            "Actual",
            "Warning",
            "Limit",
            "Status",
        ]
    ]


def create_warning_report(limit_results):
    warnings = limit_results[
        limit_results["Status"] == "WARNING"
    ].copy()

    return warnings[
        [
            "Metric",
            "Actual",
            "Warning",
            "Limit",
            "Status",
        ]
    ]


def create_factor_dashboard(risk_engine_results):
    factor_risk = risk_engine_results["Factor Risk"]

    return factor_risk.reset_index().rename(
        columns={"index": "Factor"}
    )


def create_stress_dashboard(risk_engine_results):
    factor_stress = risk_engine_results["Factor Stress"]

    return factor_stress[
        [
            "Scenario",
            "Portfolio Return",
            "Portfolio P&L",
        ]
    ].copy()


def save_dashboard_report(
    dashboard,
    output_path="data/risk_dashboard.csv",
):
    dashboard.to_csv(
        output_path,
        index=False,
    )

    return output_path