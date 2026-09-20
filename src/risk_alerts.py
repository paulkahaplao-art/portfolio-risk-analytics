import pandas as pd


def create_alert_message(
    metric,
    actual,
    warning,
    limit,
    status,
):
    if status == "BREACH":
        return (
            f"{metric} has breached its risk limit. "
            f"Actual: {actual:.2%}; "
            f"Limit: {limit:.2%}."
        )

    if status == "WARNING":
        return (
            f"{metric} is approaching its risk limit. "
            f"Actual: {actual:.2%}; "
            f"Warning level: {warning:.2%}; "
            f"Limit: {limit:.2%}."
        )

    return (
        f"{metric} remains within its risk limit. "
        f"Actual: {actual:.2%}; "
        f"Limit: {limit:.2%}."
    )


def create_risk_alerts(limit_results):
    alerts = []

    for _, row in limit_results.iterrows():
        actual = row["Actual"]
        warning = row["Warning"]
        limit = row["Limit"]
        status = row["Status"]

        if limit == 0:
            utilisation = float("nan")
        else:
            utilisation = abs(actual) / abs(limit)

        message = create_alert_message(
            row["Metric"],
            actual,
            warning,
            limit,
            status,
        )

        alerts.append({
            "Metric": row["Metric"],
            "Status": status,
            "Actual": actual,
            "Warning": warning,
            "Limit": limit,
            "Limit Utilisation": utilisation,
            "Message": message,
        })

    return pd.DataFrame(alerts)


def filter_active_alerts(alerts):
    return alerts[
        alerts["Status"].isin(
            ["WARNING", "BREACH"]
        )
    ].copy()


def create_alert_summary(alerts):
    breaches = (
        alerts["Status"] == "BREACH"
    ).sum()

    warnings = (
        alerts["Status"] == "WARNING"
    ).sum()

    if breaches > 0:
        overall_status = "BREACH"
    elif warnings > 0:
        overall_status = "WARNING"
    else:
        overall_status = "PASS"

    return {
        "Overall Status": overall_status,
        "Breaches": int(breaches),
        "Warnings": int(warnings),
        "Active Alerts": int(breaches + warnings),
        "Total Metrics": len(alerts),
    }


def create_alert_report(limit_results):
    alerts = create_risk_alerts(
        limit_results
    )

    active_alerts = filter_active_alerts(
        alerts
    )

    summary = create_alert_summary(
        alerts
    )

    return {
        "Alerts": alerts,
        "Active Alerts": active_alerts,
        "Summary": summary,
    }


def save_alert_report(
    alerts,
    output_path="Output/risk_alerts.csv",
):
    alerts.to_csv(
        output_path,
        index=False,
    )

    return output_path