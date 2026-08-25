import numpy as np


def annualised_volatility(returns, periods_per_year=252):
    """
    Calculate annualised volatility from periodic returns.
    """
    return returns.std() * np.sqrt(periods_per_year)

def sharpe_ratio(
    returns,
    annual_risk_free_rate=0.04,
    periods_per_year=252
):
    """
    Calculate annualised Sharpe ratio.
    """

    daily_risk_free_rate = (
        (1 + annual_risk_free_rate)
        ** (1 / periods_per_year)
        - 1
    )

    excess_returns = returns - daily_risk_free_rate

    return (
        excess_returns.mean()
        / returns.std()
        * np.sqrt(periods_per_year)
    )

def maximum_drawdown(growth):
    """
    Calculate maximum drawdown from a growth series.
    """

    running_peak = growth.cummax()

    drawdown = (
        growth / running_peak - 1
    )

    return drawdown.min()

def tracking_error(
    portfolio_returns,
    benchmark_returns,
    periods_per_year=252
):
    """
    Calculate annualised tracking error.
    """

    active_returns = (
        portfolio_returns - benchmark_returns
    )

    return (
        active_returns.std()
        * np.sqrt(periods_per_year)
    )

def information_ratio(
    portfolio_returns,
    benchmark_returns,
    periods_per_year=252
):
    """
    Calculate annualised Information Ratio.
    """

    active_returns = (
        portfolio_returns - benchmark_returns
    )

    annualised_active_return = (
        active_returns.mean()
        * periods_per_year
    )

    annualised_tracking_error = (
        active_returns.std()
        * np.sqrt(periods_per_year)
    )

    return (
        annualised_active_return
        / annualised_tracking_error
    )


