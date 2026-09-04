import pandas as pd

from src.risk_metrics import (
    annualised_volatility,
    maximum_drawdown,
    historical_var,
    expected_shortfall,
)

from src.visualisation import (
    plot_risk_summary,
    plot_risk_contributions,
    plot_rolling_volatility,
    plot_rolling_tracking_error,
    plot_cumulative_performance,
)

from src.excel_report import create_excel_report

def load_portfolio_data(filepath):
    """
    Load portfolio data from CSV.
    """
    return pd.read_csv(filepath)


def validate_portfolio_data(df):
    """
    Perform basic data-quality checks.
    """
    print("Rows:", len(df))
    print("Columns:", df.columns.tolist())

    print()
    print("Missing values:")
    print(df.isna().sum())

    print()
    print("Duplicate rows:", df.duplicated().sum())


def calculate_asset_returns(df):
    """
    Calculate daily returns for each asset.
    """
    prices = df.drop(columns=["Date"]).copy()

    returns = prices.pct_change()

    return returns


def validate_weights(weights):
    """
    Validate portfolio weights.
    """
    if abs(weights.sum() - 1.0) > 1e-10:
        raise ValueError(
            f"Portfolio weights must sum to 1. "
            f"Current total: {weights.sum()}"
        )

    if (weights < 0).any():
        raise ValueError(
            "Negative portfolio weights are not allowed."
        )


def calculate_portfolio_returns(asset_returns, weights):
    """
    Calculate daily portfolio returns using fixed asset weights.
    """
    portfolio_returns = asset_returns.mul(weights).sum(axis=1)

    return portfolio_returns

def calculate_portfolio_volatility(asset_returns, weights):
    """
    Calculate annualised portfolio volatility
    using the covariance matrix.
    """
    covariance_matrix = asset_returns.cov()

    portfolio_variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    portfolio_volatility = portfolio_variance ** 0.5

    annualised_volatility = (
        portfolio_volatility * (252 ** 0.5)
    )

    return annualised_volatility

def calculate_risk_contributions(asset_returns, weights):
    """
    Calculate marginal, component and percentage
    contribution to annualised portfolio volatility.
    """

    covariance_matrix = asset_returns.cov()

    portfolio_variance = (
        weights.T
        @ covariance_matrix
        @ weights
    )

    daily_volatility = portfolio_variance ** 0.5

    annualised_volatility = (
        daily_volatility * (252 ** 0.5)
    )

    marginal_contribution = (
        covariance_matrix @ weights
    ) / daily_volatility

    component_contribution = (
        weights * marginal_contribution
    )

    annualised_component_contribution = (
        component_contribution * (252 ** 0.5)
    )

    percentage_contribution = (
        annualised_component_contribution
        / annualised_volatility
    )

    result = pd.DataFrame({
        "Weight": weights,
        "Marginal Contribution": marginal_contribution,
        "Component Contribution":
            annualised_component_contribution,
        "Risk Contribution %":
            percentage_contribution,
    })

    return result

def calculate_benchmark_returns(asset_returns):
    """
    Calculate benchmark returns using a 50/50
    Australian and International Equity allocation.
    """

    benchmark_weights = pd.Series({
        "Australian_Equity": 0.50,
        "International_Equity": 0.50,
    })

    benchmark_returns = (
        asset_returns[
            [
                "Australian_Equity",
                "International_Equity",
            ]
        ]
        .mul(benchmark_weights)
        .sum(axis=1)
    )

    return benchmark_returns

def calculate_active_returns(
    portfolio_returns,
    benchmark_returns
):
    """
    Calculate portfolio returns relative
    to the benchmark.
    """

    return portfolio_returns - benchmark_returns

def calculate_tracking_error(
    portfolio_returns,
    benchmark_returns
):
    """
    Calculate annualised tracking error.
    """

    active_returns = (
        portfolio_returns
        - benchmark_returns
    )

    tracking_error = (
        active_returns.std()
        * (252 ** 0.5)
    )

    return tracking_error

def calculate_information_ratio(
    portfolio_returns,
    benchmark_returns
):
    """
    Calculate the Information Ratio.
    """

    active_returns = (
        portfolio_returns
        - benchmark_returns
    )

    annualised_active_return = (
        active_returns.mean() * 252
    )

    tracking_error = (
        active_returns.std()
        * (252 ** 0.5)
    )

    if tracking_error == 0:
        return float("nan")

    return (
        annualised_active_return
        / tracking_error
    )

def calculate_risk_metrics(
    returns,
    portfolio_value=1_000_000
):
    """
    Calculate key portfolio risk metrics.
    """

    volatility = annualised_volatility(returns)

    growth = (1 + returns).cumprod()

    drawdown = maximum_drawdown(growth)

    var_95 = historical_var(
        returns,
        confidence_level=0.95
    )

    var_99 = historical_var(
        returns,
        confidence_level=0.99
    )

    es_95 = expected_shortfall(
        returns,
        confidence_level=0.95
    )

    return {
        "Annualised Volatility": volatility,
        "Maximum Drawdown": drawdown,
        "95% VaR": var_95,
        "99% VaR": var_99,
        "95% Expected Shortfall": es_95,
        "95% VaR ($)": var_95 * portfolio_value,
        "99% VaR ($)": var_99 * portfolio_value,
        "95% Expected Shortfall ($)": es_95 * portfolio_value,
    }

def calculate_rolling_volatility(
    portfolio_returns,
    window=5
):
    """
    Calculate annualised rolling portfolio volatility.
    """

    return (
        portfolio_returns
        .rolling(window)
        .std()
        * (252 ** 0.5)
    )

def calculate_rolling_tracking_error(
    portfolio_returns,
    benchmark_returns,
    window=5
):
    """
    Calculate annualised rolling tracking error.
    """

    active_returns = (
        portfolio_returns - benchmark_returns
    )

    return (
        active_returns
        .rolling(window)
        .std()
        * (252 ** 0.5)
    )

def calculate_rolling_active_return(
    portfolio_returns,
    benchmark_returns,
    window=5
):
    """
    Calculate annualised rolling active return.
    """

    active_returns = (
        portfolio_returns - benchmark_returns
    )

    return (
        active_returns
        .rolling(window)
        .mean()
        * 252
    )

def create_risk_report(
    asset_returns,
    weights,
    portfolio_returns,
    benchmark_returns,
    portfolio_value,
):
    """
    Create a consolidated portfolio risk report.
    """

    volatility = calculate_portfolio_volatility(
        asset_returns,
        weights,
    )   

    drawdown = maximum_drawdown(
        (1 + portfolio_returns).cumprod()
    )

    var_95 = historical_var(
        portfolio_returns,
        0.95
    )

    var_99 = historical_var(
        portfolio_returns,
        0.99
    )

    es_95 = expected_shortfall(
        portfolio_returns,
        0.95
    )

    tracking_error = calculate_tracking_error(
        portfolio_returns,
        benchmark_returns
    )

    information_ratio = calculate_information_ratio(
        portfolio_returns,
        benchmark_returns
    )

    report = {
        "Annualised Volatility": volatility,
        "Maximum Drawdown": drawdown,
        "95% VaR": var_95,
        "99% VaR": var_99,
        "95% Expected Shortfall": es_95,
        "95% VaR ($)": var_95 * portfolio_value,
        "99% VaR ($)": var_99 * portfolio_value,
        "95% Expected Shortfall ($)": es_95 * portfolio_value,
        "Tracking Error": tracking_error,
        "Information Ratio": information_ratio,
    }

    return pd.Series(report)

if __name__ == "__main__":

    filepath = "data/portfolio.csv"

    # Load data
    df = load_portfolio_data(filepath)

    # Validate data
    validate_portfolio_data(df)

    # Define portfolio weights
    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    # Calculate asset returns
    asset_returns = calculate_asset_returns(df)
    asset_returns = asset_returns.dropna()

    # Calculate portfolio returns
    portfolio_returns = calculate_portfolio_returns(
        asset_returns,
        weights,
    )

    # Calculate benchmark
    benchmark_returns = calculate_benchmark_returns(
        asset_returns,
    )

    # Calculate active returns
    active_returns = calculate_active_returns(
        portfolio_returns,
        benchmark_returns,
    )

    # Calculate risk contributions
    risk_contributions = calculate_risk_contributions(
        asset_returns,
        weights,
    )

    # Calculate rolling volatility
    rolling_volatility = calculate_rolling_volatility(
        portfolio_returns,
        window=5,
    )

    # Calculate rolling tracking error
    rolling_tracking_error = calculate_rolling_tracking_error(
        portfolio_returns,
        benchmark_returns,
        window=5,
    )

    # Create consolidated risk report
    risk_report = create_risk_report(
        asset_returns,
        weights,
        portfolio_returns,
        benchmark_returns,
        portfolio_value=1_000_000,
    )

    # ONLY NOW create charts
    plot_risk_summary(
        risk_report,
        "reports/charts/risk_summary.png",
    )

    plot_risk_contributions(
        risk_contributions,
        "reports/charts/risk_contributions.png",
    )

    plot_rolling_volatility(
        rolling_volatility,
        "reports/charts/rolling_volatility.png",
    )

    plot_rolling_tracking_error(
        rolling_tracking_error,
        "reports/charts/rolling_tracking_error.png",
    )

    plot_cumulative_performance(
    portfolio_returns,
    benchmark_returns,
    "reports/charts/cumulative_performance.png",
    )

        # Create Excel risk report
    create_excel_report(
        risk_report=risk_report,
        risk_contributions=risk_contributions,
        rolling_volatility=rolling_volatility,
        rolling_tracking_error=rolling_tracking_error,
        portfolio_returns=portfolio_returns,
        benchmark_returns=benchmark_returns,
        output_path="reports/portfolio_risk_report.xlsx",
    )