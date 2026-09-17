import pandas as pd

from src.risk_engine_validation import (
    validate_weights,
    validate_risk_engine_results,
)

from src.historical_data import load_historical_data

from src.historical_analysis import (
    calculate_historical_var,
    calculate_expected_shortfall,
    calculate_maximum_drawdown,
)

from src.risk_attribution import (
    calculate_covariance_matrix,
    calculate_portfolio_volatility,
    create_risk_attribution_table,
)

from src.risk_budget import (
    calculate_risk_concentration,
    create_risk_budget_table,
)

from src.risk_limits import (
    build_risk_monitoring_report,
)

from src.factor_risk import (
    calculate_factor_exposure,
    calculate_portfolio_factor_exposure,
    calculate_factor_covariance,
    calculate_factor_risk_contribution,
    calculate_residual_returns,
    calculate_residual_risk,
    run_factor_stress_scenarios,
)

from src.risk_dashboard import (
    create_risk_dashboard,
    calculate_limit_utilisation,
    create_breach_report,
    create_warning_report,
    create_factor_dashboard,
    create_stress_dashboard,
)

from src.risk_monitoring_excel import (
    create_risk_monitoring_excel,
)

WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})

from src.risk_dashboard import save_dashboard_report




def calculate_portfolio_returns(
    asset_returns,
    weights,
):
    """Calculate weighted portfolio returns."""

    aligned_weights = weights.reindex(
        asset_returns.columns
    )

    return (
        asset_returns
        .mul(aligned_weights, axis=1)
        .sum(axis=1)
    )


def calculate_historical_risk(
    portfolio_returns,
):
    """Calculate historical portfolio risk metrics."""

    return {
        "Volatility": (
            portfolio_returns.std()
            * (252 ** 0.5)
        ),
        "VaR 95%": calculate_historical_var(
            portfolio_returns,
            confidence=0.95,
        ),
        "VaR 99%": calculate_historical_var(
            portfolio_returns,
            confidence=0.99,
        ),
        "ES 95%": calculate_expected_shortfall(
            portfolio_returns,
            confidence=0.95,
        ),
        "ES 99%": calculate_expected_shortfall(
            portfolio_returns,
            confidence=0.99,
        ),
        "Maximum Drawdown": (
            calculate_maximum_drawdown(
                portfolio_returns
            )
        ),
    }


def build_risk_engine(
    filepath,
    weights,
    factor_returns,
    portfolio_value=100_000,
):
    """Build an integrated portfolio risk model."""

    validate_weights(weights)

    historical_data = load_historical_data(
        filepath
    )

    asset_returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    portfolio_returns = (
        calculate_portfolio_returns(
            asset_returns,
            weights,
        )
    )

    historical_risk = (
        calculate_historical_risk(
            portfolio_returns
        )
    )

    covariance = (
        calculate_covariance_matrix(
            asset_returns
        )
    )

    portfolio_volatility = (
        calculate_portfolio_volatility(
            weights,
            covariance,
        )
    )

    attribution = (
        create_risk_attribution_table(
            weights,
            covariance,
        )
    )

    risk_concentration = (
        calculate_risk_concentration(
            attribution[
                "Percentage Contribution"
            ],
            weights,
        )
    )

    exposures = calculate_factor_exposure(
        asset_returns,
        factor_returns,
    )

    portfolio_factor_exposure = (
        calculate_portfolio_factor_exposure(
            exposures,
            weights,
        )
    )

    factor_covariance = (
        calculate_factor_covariance(
            factor_returns
        )
    )

    factor_risk = (
        calculate_factor_risk_contribution(
            portfolio_factor_exposure,
            factor_covariance,
        )
    )

    residual_returns = (
        calculate_residual_returns(
            asset_returns,
            factor_returns,
        )
    )

    residual_risk = calculate_residual_risk(
        residual_returns
    )

    factor_stress = (
        run_factor_stress_scenarios(
            portfolio_factor_exposure,
            portfolio_value,
        )
    )

    results = {
        "Asset Returns": asset_returns,
        "Portfolio Returns": portfolio_returns,
        "Historical Risk": historical_risk,
        "Portfolio Volatility": portfolio_volatility,
        "Risk Attribution": attribution,
        "Risk Concentration": risk_concentration,
        "Factor Exposures": exposures,
        "Portfolio Factor Exposure": portfolio_factor_exposure,
        "Factor Risk": factor_risk,
        "Residual Risk": residual_risk,
        "Factor Stress": factor_stress,
    }

    risk_monitoring = build_risk_monitoring_report(
        results,
        weights,
    )

    results["Risk Monitoring"] = risk_monitoring

    validate_risk_engine_results(results)
    return results


if __name__ == "__main__":

    historical_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    factor_data = pd.read_csv(
        "data/historical/factor_data.csv",
        parse_dates=["Date"],
    )

    factor_data = factor_data.set_index(
        "Date"
    )

    asset_returns = (
        historical_data
        .pct_change()
        .dropna()
    )

    factor_returns = factor_data.loc[
        asset_returns.index
    ]

    results = build_risk_engine(
        "data/historical/market_data.csv",
        WEIGHTS,
        factor_returns,
        portfolio_value=100_000,
    )

    dashboard = create_risk_dashboard(results)

    limit_utilisation = calculate_limit_utilisation(
        results["Risk Monitoring"]["Limit Results"]
    )

    breach_report = create_breach_report(
        results["Risk Monitoring"]["Limit Results"]
    )

    warning_report = create_warning_report(
        results["Risk Monitoring"]["Limit Results"]
    )

    factor_dashboard = create_factor_dashboard(results)

    stress_dashboard = create_stress_dashboard(results)

    excel_path = create_risk_monitoring_excel(
    dashboard,
    limit_utilisation,
    factor_dashboard,
    stress_dashboard,
    results["Risk Monitoring"]["Summary"],
    )

    print(f"\nExcel risk monitoring report saved to: {excel_path}")

    print("\nRisk Dashboard")
    print("=" * 60)
    print(dashboard)

    print("\nLimit Utilisation")
    print("-" * 60)
    print(limit_utilisation)

    print("\nBreaches")
    print("-" * 60)
    print(breach_report)

    print("\nWarnings")
    print("-" * 60)
    print(warning_report)

    print("\nFactor Dashboard")
    print("-" * 60)
    print(factor_dashboard)

    print("\nStress Dashboard")
    print("-" * 60)
    print(stress_dashboard)

    print("\nINTEGRATED PORTFOLIO RISK ENGINE")
    print("=" * 90)

    print("\nHistorical Risk")
    print("-" * 90)

    for metric, value in results[
        "Historical Risk"
    ].items():

        print(
            f"{metric}: {value:.2%}"
        )

    print("\nPortfolio Volatility")
    print("-" * 90)

    print(
        f"{results['Portfolio Volatility']:.2%}"
    )

    print("\nRisk Attribution")
    print("-" * 90)

    print(
        results[
            "Risk Attribution"
        ].to_string(
            formatters={
                "Weight": "{:.2%}".format,
                "Marginal Contribution": "{:.6f}".format,
                "Component Contribution": "{:.2%}".format,
                "Percentage Contribution": "{:.2%}".format,
            }
        )
    )

    print("\nPortfolio Factor Exposure")
    print("-" * 90)

    print(
        results[
            "Portfolio Factor Exposure"
        ].to_string(
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\nFactor Risk")
    print("-" * 90)

    print(
        results[
            "Factor Risk"
        ].to_string(
            formatters={
                "Factor Exposure": "{:.4f}".format,
                "Marginal Risk": "{:.6f}".format,
                "Component Risk": "{:.2%}".format,
                "Percentage Risk": "{:.2%}".format,
            }
        )
    )

    print("\nResidual Risk")
    print("-" * 90)

    print(
        results[
            "Residual Risk"
        ].to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print("\nFactor Stress Tests")
    print("-" * 90)

    print(
        results[
            "Factor Stress"
        ].to_string(
            index=False,
            formatters={
                "Portfolio Return": "{:.2%}".format,
                "Portfolio P&L": "${:,.2f}".format,
            }
        )
    )

    print("\nRisk Monitoring")
    print("-" * 50)

    monitoring = results["Risk Monitoring"]

    print("\nRisk Limit Results:")
    print(monitoring["Limit Results"])

    print("\nRisk Monitoring Summary:")
    print(monitoring["Summary"])

    save_dashboard_report(
    dashboard,
    "data/risk_dashboard.csv",
    )