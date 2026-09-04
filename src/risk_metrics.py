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
    Calculate historical Value at Risk as a positive loss magnitude.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    loss_quantile = returns.quantile(1 - confidence_level)

    return -loss_quantile


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
    Calculate historical Expected Shortfall as a positive loss magnitude.
    """
    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )

    var_threshold = returns.quantile(1 - confidence_level)

    tail_losses = returns[returns <= var_threshold]

    return -tail_losses.mean()


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