import pandas as pd


def calculate_risk_concentration(
    risk_contributions,
    weights,
):
    """Calculate risk contribution relative to capital allocation."""

    result = pd.DataFrame({
        "Weight": weights,
        "Risk Contribution": risk_contributions,
    })

    result["Risk / Capital Ratio"] = (
        result["Risk Contribution"]
        / result["Weight"]
    )

    return result


def identify_risk_concentration(
    risk_contributions,
    weights,
    threshold=1.5,
):
    """Identify assets whose risk contribution is high relative to weight."""

    concentration = calculate_risk_concentration(
        risk_contributions,
        weights,
    )

    return concentration[
        concentration["Risk / Capital Ratio"] >= threshold
    ]


def calculate_risk_budget_deviation(
    actual_risk_contribution,
    target_risk_contribution,
):
    """Calculate deviation between actual and target risk contribution."""

    return (
        actual_risk_contribution
        - target_risk_contribution
    )


def create_risk_budget_table(
    weights,
    actual_risk_contributions,
    target_risk_contributions,
):
    """Create a portfolio risk-budget table."""

    table = pd.DataFrame({
        "Weight": weights,
        "Actual Risk Contribution": actual_risk_contributions,
        "Target Risk Contribution": target_risk_contributions,
    })

    table["Risk Budget Deviation"] = (
        table["Actual Risk Contribution"]
        - table["Target Risk Contribution"]
    )

    table["Risk / Capital Ratio"] = (
        table["Actual Risk Contribution"]
        / table["Weight"]
    )

    return table


if __name__ == "__main__":

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    # Example values.
    # These will later be replaced by the actual
    # risk attribution results.
    risk_contributions = pd.Series({
        "Australian_Equity": 0.25,
        "International_Equity": 0.60,
        "Bonds": 0.10,
        "Cash": 0.05,
    })

    target_risk = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    concentration = calculate_risk_concentration(
        risk_contributions,
        weights,
    )

    risk_budget = create_risk_budget_table(
        weights,
        risk_contributions,
        target_risk,
    )

    print("\nRisk Concentration")
    print("=" * 80)

    print(
        concentration.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Risk Contribution": "{:.2%}".format,
                "Risk / Capital Ratio": "{:.2f}x".format,
            }
        )
    )

    print("\nRisk Budget")
    print("=" * 80)

    print(
        risk_budget.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Actual Risk Contribution": "{:.2%}".format,
                "Target Risk Contribution": "{:.2%}".format,
                "Risk Budget Deviation": "{:.2%}".format,
                "Risk / Capital Ratio": "{:.2f}x".format,
            }
        )
    )

    print("\nPotential Risk Concentrations")
    print("=" * 80)

    concentrations = identify_risk_concentration(
        risk_contributions,
        weights,
    )

    print(
        concentrations.to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Risk Contribution": "{:.2%}".format,
                "Risk / Capital Ratio": "{:.2f}x".format,
            }
        )
    )