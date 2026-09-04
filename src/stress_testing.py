import pandas as pd

from src.historical_data import load_historical_data


WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


def calculate_scenario_return(weights, scenario):
    """Calculate portfolio return under a specified scenario."""

    scenario_returns = pd.Series(scenario)

    scenario_returns = scenario_returns.reindex(
        weights.index
    )

    if scenario_returns.isna().any():
        raise ValueError(
            "Scenario is missing one or more portfolio assets."
        )

    return (weights * scenario_returns).sum()


def calculate_scenario_loss(
    portfolio_value,
    portfolio_return,
):
    """Calculate dollar profit/loss under a scenario."""

    return portfolio_value * portfolio_return


def classify_scenario_severity(portfolio_return):
    """Classify scenario severity based on portfolio loss."""

    if portfolio_return > -0.05:
        return "Low"

    if portfolio_return > -0.15:
        return "Moderate"

    if portfolio_return > -0.25:
        return "High"

    return "Severe"


def run_stress_scenarios(
    portfolio_value,
    weights,
):
    """Run predefined portfolio stress scenarios."""

    scenarios = {
        "Moderate Equity Selloff": {
            "Australian_Equity": -0.10,
            "International_Equity": -0.15,
            "Bonds": -0.02,
            "Cash": 0.00,
        },
        "Severe Equity Selloff": {
            "Australian_Equity": -0.25,
            "International_Equity": -0.35,
            "Bonds": -0.05,
            "Cash": 0.00,
        },
        "Equity + Bond Shock": {
            "Australian_Equity": -0.20,
            "International_Equity": -0.25,
            "Bonds": -0.10,
            "Cash": 0.00,
        },
        "Diversified Stress": {
            "Australian_Equity": -0.15,
            "International_Equity": -0.20,
            "Bonds": 0.03,
            "Cash": 0.00,
        },
    }

    results = []

    for scenario_name, scenario in scenarios.items():

        portfolio_return = calculate_scenario_return(
            weights,
            scenario,
        )

        loss = calculate_scenario_loss(
            portfolio_value,
            portfolio_return,
        )

        severity = classify_scenario_severity(
            portfolio_return
        )

        results.append({
            "Scenario": scenario_name,
            "Portfolio Return": portfolio_return,
            "Portfolio P&L": loss,
            "Severity": severity,
        })

    return pd.DataFrame(results)


def identify_worst_days(
    asset_returns,
    weights,
    n=10,
):
    """Identify the worst historical portfolio days."""

    portfolio_returns = (
        asset_returns[weights.index]
        .mul(weights)
        .sum(axis=1)
    )

    return portfolio_returns.nsmallest(n)


def calculate_day_contributions(
    asset_returns,
    weights,
    date,
):
    """Calculate asset contributions to portfolio return."""

    day_returns = asset_returns.loc[
        date,
        weights.index,
    ]

    return day_returns * weights


def create_historical_stress_scenario(
    asset_returns,
    weights,
):
    """Create a stress scenario from the worst historical day."""

    portfolio_returns = (
        asset_returns[weights.index]
        .mul(weights)
        .sum(axis=1)
    )

    worst_date = portfolio_returns.idxmin()

    scenario = asset_returns.loc[
        worst_date,
        weights.index,
    ].to_dict()

    return worst_date, scenario


if __name__ == "__main__":

    portfolio_value = 100_000

    weights = WEIGHTS.copy()

    print("\nPortfolio Stress Test")
    print("=" * 70)

    results = run_stress_scenarios(
        portfolio_value,
        weights,
    )

    print(
        results.to_string(
            index=False,
            formatters={
                "Portfolio Return": "{:.2%}".format,
                "Portfolio P&L": "${:,.2f}".format,
            },
        )
    )

    historical_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    historical_returns = (
        historical_data.pct_change().dropna()
    )

    worst_days = identify_worst_days(
        historical_returns,
        weights,
        n=10,
    )

    print("\nWorst 10 Historical Portfolio Days")
    print("=" * 70)

    print(
        worst_days.to_string(
            float_format=lambda x: f"{x:.4%}"
        )
    )

    worst_date = worst_days.index[0]

    contributions = calculate_day_contributions(
        historical_returns,
        weights,
        worst_date,
    )

    print("\nWorst Historical Day")
    print("=" * 70)
    print(f"Date: {worst_date.date()}")
    print(
        f"Portfolio Return: "
        f"{worst_days.iloc[0]:.4%}"
    )

    print("\nAsset Contributions")
    print("-" * 70)

    for asset, contribution in contributions.items():
        print(
            f"{asset}: "
            f"{contribution:.4%}"
        )

    stress_date, historical_scenario = (
        create_historical_stress_scenario(
            historical_returns,
            weights,
        )
    )

    historical_stress_return = (
        weights
        * pd.Series(historical_scenario)
    ).sum()

    historical_stress_loss = (
        portfolio_value
        * historical_stress_return
    )

    print("\nHistorical Stress Scenario")
    print("=" * 70)
    print(f"Date: {stress_date.date()}")
    print(
        f"Portfolio Return: "
        f"{historical_stress_return:.4%}"
    )
    print(
        f"Portfolio P&L: "
        f"${historical_stress_loss:,.2f}"
    )