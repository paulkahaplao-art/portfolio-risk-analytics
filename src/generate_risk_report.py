from pathlib import Path

import pandas as pd

from src.historical_data import load_historical_data
from src.portfolio_dynamics import calculate_rebalanced_returns
from src.historical_analysis import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_maximum_drawdown,
)

from src.risk_attribution import(
    calculate_covariance_matrix,
    create_risk_attribution_table,
)

from src.factor_risk import(
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
)

from src.stress_testing import run_stress_scenarios

DATA_PATH = "data/historical/market_data.csv"
FACTOR_PATH = "data/historical/factor_data.csv"
OUTPUT_DIR = Path("Output")

PORTFOLIO_VALUE = 100_000

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})

VOLATILITY_LIMIT = 0.08
VAR_LIMIT = -0.02
ES_LIMIT = -0.03
DRAWDOWN_LIMIT = -0.1

def load_data():
    historical_data = load_historical_data(DATA_PATH)

    factor_data = pd.read_csv(
        FACTOR_PATH,
        parse_dates=["Date"],
    ).set_index("Date")

    asset_returns = historical_data.pct_change().dropna()

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    portfolio_returns = calculate_rebalanced_returns(
        asset_returns,
        WEIGHTS,
    )

    return (
        asset_returns, 
        factor_returns,
        portfolio_returns,
    )

def calculate_risk_metrics(portfolio_returns):
    volatility = (
        portfolio_returns.std()
        * (252 ** 0.5)
    )


    var_95 = calculate_historical_var(
        portfolio_returns,
        confidence=0.95
    )

    es_95 = calculate_expected_shortfall(
        portfolio_returns,
        confidence=0.95,
    )

    maximum_drawdown = calculate_maximum_drawdown(
        portfolio_returns
    )

    return {
        "Volatility": volatility,
        "VaR 95%": var_95,
        "Expected Shortfall 95%": es_95,
        "Maximum Drawdown": maximum_drawdown,
    }

def calculate_status(actual, limit):
    utilisation = abs(actual)/abs(limit)

    if utilisation >=1.0:
        return "BREACH"

    if utilisation >=0.875:
        return "WARNING"

    return "PASS"

def create_risk_summary(risk_metrics):
    limits = {
        "Volatility": VOLATILITY_LIMIT,
        "VaR 95%": VAR_LIMIT,
        "Expected Shortfall 95%": ES_LIMIT,
        "Maximum Drawdown": DRAWDOWN_LIMIT,
    }

    rows = []

    for metric, actual in risk_metrics.items():
        limit = limits[metric]

        utilisation = (
            abs(actual) / abs(limit)
        )

        status = calculate_status(
            actual, 
            limit,
        )

        rows.append({
            "Report Selection": "Risk Metrics",
            "Metric": metric,
            "Actual": actual, 
            "Limit": limit,
            "Utilisation": utilisation, 
            "Status": status
        })

        return pd.DataFrame(rows)

def create_attribution_summary(asset_returns):
    covariance = calculate_covariance_matrix(
        asset_returns
    )

    attribution = create_risk_attribution_table(
        WEIGHTS,
        covariance,
    )

    rows = []

    for asset in attribution.index:
        rows.append({
            "Report Section": "Asset Risk Attribution",
            "Metric": asset,
            "Actual": attribution.loc[
                asset,
                "Percentage Contribution"
            ],
            "Limit": None,
            "Utilisation": None,
            "Status": "INFO",
        })

    return pd.DataFrame(rows)

def create_factor_summary(
        asset_returns,
        factor_returns,
):
    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    portfolio_exposure = (
        calculate_portfolio_factor_exposure(
            exposures,
            WEIGHTS,
        )
    )

    rows = []

    for factor in portfolio_exposure.index:
        rows.append({
            "Report Section": "Factor Exposure",
            "Metric": factor,
            "Actual": portfolio_exposure[factor],
            "Limit": None,
            "Utilisation": None,
            "Status": "INFO",
        })

    return pd.DataFrame(rows)


def create_stress_summary():
    stress_results = run_stress_scenarios(
        PORTFOLIO_VALUE,
        WEIGHTS,
    )

    rows = []

    for _, row in stress_results.iterrows():
        rows.append({
            "Report Section": "Stress Test",
            "Metric": row["Scenario"],
            "Actual": row["Portfolio Return"],
            "Limit": None,
            "Utilisation": None,
            "Status": row["Severity"],
        })

    return pd.DataFrame(rows)

def create_overall_summary(risk_summary):
    if (
        risk_summary["Status"] == "BREACH"
    ).any():
        overall_status = "BREACH"

    elif (
        risk_summary["Status"] == "WARNING"
    ).any():
        overall_status = "WARNING"

    else:
        overall_status = "PASS"

    return pd.DataFrame([{
        "Report Section": "Overall",
        "Metric": "Overall Risk Status",
        "Actual": None,
        "Limit": None,
        "Utilisation": None,
        "Status": overall_status,
    }])

def build_report():
    (
        asset_returns,
        factor_returns,
        portfolio_returns,
    ) = load_data()

    risk_metrics = calculate_risk_metrics(
        portfolio_returns
    )

    risk_summary = create_risk_summary(
        risk_metrics
    )

    attribution_summary = (
        create_attribution_summary(
            asset_returns
        )
    )

    factor_summary = create_factor_summary(
        asset_returns,
        factor_returns,
    )

    stress_summary = create_stress_summary()

    overall_summary = create_overall_summary (
        risk_summary
    )

    report = pd.concat(
        [
            overall_summary,
            risk_summary,
            attribution_summary,
            factor_summary,
            stress_summary,
        ],
        ignore_index=True,
    )

    return report

def save_report(report):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        OUTPUT_DIR
        / "portfolio_risk_report.csv"
    )

    report.to_csv(
        output_path,
        index=False,
    )

    return output_path

def main():
    print("=" * 60)
    print("Generating Portfolio Risk Report")
    print("=" * 60)

    report = build_report()

    print("\nPortfolio Risk Report")
    print("-" * 60)
    print(report.to_string(index=False))

    output_path = save_report(report)

    print("\nReport created:")
    print(output_path)

    print("\nComplete.")

if __name__ == "__main__":
    main()

