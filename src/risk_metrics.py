import numpy as np
import pandas as pd


def annualised_volatility(returns, periods_per_year=252):
    """
    Calculate annualised volatility from periodic returns.
    """
    return returns.std() * np.sqrt(periods_per_year)


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualised Sharpe ratio.
    """
    excess_returns = returns - risk_free_rate / periods_per_year

    return (
        excess_returns.mean()
        / excess_returns.std()
        * np.sqrt(periods_per_year)
    )

def maximum_drawdown(growth):
    """
    Calculate maximum drawdown from a growth/wealth series.
    """
    running_max = growth.cummax()

    drawdown = growth / running_max - 1

    return drawdown.min()


def historical_var(returns, confidence_level=0.95):
    """
    Calculate historical Value at Risk.

    Returns VaR as a positive number representing the potential loss.
    """
    percentile = returns.quantile(1 - confidence_level)

    return -percentile


def var_dollar_value(
    returns,
    portfolio_value,
    confidence_level=0.95
):
    """
    Calculate historical VaR in dollar terms.
    """
    var = historical_var(
        returns,
        confidence_level=confidence_level
    )

    return portfolio_value * var


def expected_shortfall(returns, confidence_level=0.95):
    """
    Calculate historical Expected Shortfall.

    Returns the average loss beyond the VaR threshold.
    """
    var_threshold = returns.quantile(1 - confidence_level)

    tail_returns = returns[returns <= var_threshold]

    return -tail_returns.mean()


def expected_shortfall_dollar_value(
    returns,
    portfolio_value,
    confidence_level=0.95
):
    """
    Calculate Expected Shortfall in dollar terms.
    """
    es = expected_shortfall(
        returns,
        confidence_level=confidence_level
    )

    return portfolio_value * es