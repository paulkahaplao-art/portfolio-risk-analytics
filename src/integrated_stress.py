import pandas as pd

from src.historical_data import load_historical_data
from src.stress_testing import (
    calculate_scenario_return,
    calculate_scenario_loss,
    classify_scenario_severity,
    create_historical_stress_scenario,
    run_stress_scenarios,
    rank_stress_scenarios,
)
from src.factor_risk import (
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
    run_factor_stress_scenarios,
)


WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


def run_historical_stress(
    asset_returns,
    weights,
    portfolio_value=100_000,
):
    """
    Calculate the worst observed historical portfolio day.
    """

    worst_date, scenario = create_historical_stress_scenario(
        asset_returns,
        weights,
    )

    portfolio_return = calculate_scenario_return(
        weights,
        scenario,
    )

    portfolio_pnl = calculate_scenario_loss(
        portfolio_value,
        portfolio_return,
    )

    severity = classify_scenario_severity(
        portfolio_return,
    )

    return {
        "Scenario": "Historical Worst Day",
        "Date": worst_date,
        "Portfolio Return": portfolio_return,
        "Portfolio P&L": portfolio_pnl,
        "Severity": severity,
    }


def run_asset_stress(
    portfolio_value,
    weights,
):
    """
    Run predefined asset-level stress scenarios.
    """

    results = run_stress_scenarios(
        portfolio_value,
        weights,
    )

    results["Date"] = pd.NaT

    return results[
        [
            "Scenario",
            "Date",
            "Portfolio Return",
            "Portfolio P&L",
            "Severity",
        ]
    ]


def run_factor_stress(
    asset_returns,
    factor_returns,
    weights,
    portfolio_value=100_000,
):
    """
    Run factor-based stress scenarios.
    """

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    portfolio_exposure = calculate_portfolio_factor_exposure(
        exposures,
        weights,
    )

    results = run_factor_stress_scenarios(
        portfolio_exposure,
        portfolio_value,
    )

    results["Date"] = pd.NaT
    results["Severity"] = results["Portfolio Return"].apply(
        classify_scenario_severity
    )

    results["Scenario"] = (
        "Factor: "
        + results["Scenario"]
    )

    return results[
        [
            "Scenario",
            "Date",
            "Portfolio Return",
            "Portfolio P&L",
            "Severity",
        ]
    ]


def build_integrated_stress_report(
    filepath,
    factor_returns,
    weights,
    portfolio_value=100_000,
):
    """
    Combine historical, asset-level and factor stress tests.
    """

    historical_data = load_historical_data(filepath)

    asset_returns = historical_data.pct_change().dropna()

    historical_result = pd.DataFrame([
        run_historical_stress(
            asset_returns,
            weights,
            portfolio_value,
        )
    ])

    asset_results = run_asset_stress(
        portfolio_value,
        weights,
    )

    factor_results = run_factor_stress(
        asset_returns,
        factor_returns,
        weights,
        portfolio_value,
    )

    combined = pd.concat(
        [
            historical_result,
            asset_results,
            factor_results,
        ],
        ignore_index=True,
    )

    combined = combined[
        [
            "Scenario",
            "Date",
            "Portfolio Return",
            "Portfolio P&L",
            "Severity",
        ]
    ]

    return combined


def identify_worst_stress_scenario(results):
    """
    Identify the scenario producing the largest portfolio loss.
    """

    ranked = rank_stress_scenarios(results)

    return ranked.iloc[0]


if __name__ == "__main__":

    factor_data = pd.read_csv(
        "data/historical/factor_data.csv",
        parse_dates=["Date"],
    ).set_index("Date")

    historical_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    asset_returns = historical_data.pct_change().dropna()

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    results = build_integrated_stress_report(
        "data/historical/market_data.csv",
        factor_returns,
        WEIGHTS,
        portfolio_value=100_000,
    )

    worst = identify_worst_stress_scenario(
        results
    )

    print("\nINTEGRATED STRESS TESTING")
    print("=" * 100)

    print("\nAll Stress Scenarios")
    print("-" * 100)

    print(
        results.to_string(
            index=False,
            formatters={
                "Portfolio Return": "{:.2%}".format,
                "Portfolio P&L": "${:,.2f}".format,
            },
        )
    )

    print("\nWorst Stress Scenario")
    print("-" * 100)

    print(
        f"Scenario: {worst['Scenario']}"
    )

    print(
        f"Portfolio Return: "
        f"{worst['Portfolio Return']:.2%}"
    )

    print(
        f"Portfolio P&L: "
        f"${worst['Portfolio P&L']:,.2f}"
    )

    print(
        f"Severity: {worst['Severity']}"
    )