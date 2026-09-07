import pandas as pd

from src.historical_data import load_historical_data

def calculate_portfolio_returns(returns, weights):
    """Calculate daily portfolio returns from asset returns."""
    return returns[weights.index].mul(weights).sum(axis=1)


def calculate_historical_returns(df):
    """Calculate daily percentage returns from historical index levels."""
    returns = df.pct_change().dropna()
    return returns


def calculate_summary_statistics(returns):
    """Calculate annualised return and volatility."""
    annualised_return = (1 + returns.mean()) ** 252 - 1
    annualised_volatility = returns.std() * (252 ** 0.5)

    summary = pd.DataFrame({
        "Annualised Return": annualised_return,
        "Annualised Volatility": annualised_volatility,
    })

    return summary

def identify_extreme_returns(returns, threshold=0.05):
    """Identify daily returns exceeding the specified absolute threshold."""
    extreme = returns[returns.abs().max(axis=1) > threshold]
    return extreme

def calculate_correlation_matrix(returns):
    """Calculate the correlation matrix of asset returns."""
    return returns.corr()

def calculate_historical_var(returns, confidence=0.95):
    """Calculate historical Value at Risk."""
    return returns.quantile(1 - confidence)


def calculate_expected_shortfall(returns, confidence=0.95):
    """Calculate historical Expected Shortfall."""
    var = calculate_historical_var(returns, confidence)
    return returns[returns <= var].mean()


def calculate_historical_risk_metrics(portfolio_returns):
    """Calculate historical VaR and Expected Shortfall."""
    var_95 = calculate_historical_var(portfolio_returns, 0.95)
    var_99 = calculate_historical_var(portfolio_returns, 0.99)

    es_95 = calculate_expected_shortfall(
        portfolio_returns,
        0.95,
    )

    es_99 = calculate_expected_shortfall(
        portfolio_returns,
        0.99,
    )

    return {
        "VaR 95%": var_95,
        "VaR 99%": var_99,
        "ES 95%": es_95,
        "ES 99%": es_99,
    }

def calculate_maximum_drawdown(returns):
    """Calculate maximum drawdown from a return series."""
    cumulative = (1 + returns).cumprod()

    running_max = cumulative.cummax()

    drawdown = cumulative / running_max - 1

    return drawdown.min()

def create_risk_summary(portfolio_returns):
    """Create a compact historical risk summary."""
    volatility = portfolio_returns.std() * (252 ** 0.5)

    risk = calculate_historical_risk_metrics(
        portfolio_returns
    )

    max_drawdown = calculate_maximum_drawdown(
        portfolio_returns
    )

    return pd.Series({
        "Annualised Volatility": volatility,
        "VaR 95%": risk["VaR 95%"],
        "VaR 99%": risk["VaR 99%"],
        "ES 95%": risk["ES 95%"],
        "ES 99%": risk["ES 99%"],
        "Maximum Drawdown": max_drawdown,
    })

def calculate_rolling_volatility(returns, window=63):
    """Calculate annualised rolling volatility."""
    return returns.rolling(window).std() * (252 ** 0.5)

def calculate_rolling_var(returns, window=252, confidence=0.95):
    """Calculate rolling historical VaR."""
    return returns.rolling(window).quantile(
        1 - confidence
    )

def identify_peak_risk_period(rolling_volatility):
    """Identify the date with the highest rolling volatility."""
    rolling_volatility = rolling_volatility.dropna()

    peak_date = rolling_volatility.idxmax()
    peak_value = rolling_volatility.max()

    return peak_date, peak_value

def classify_risk_regime(
    rolling_volatility,
    low_threshold=0.08,
    high_threshold=0.15,
):
    """Classify rolling volatility into risk regimes."""
    regimes = pd.Series(
        index=rolling_volatility.index,
        dtype="object",
    )

    regimes[rolling_volatility < low_threshold] = "Low"

    regimes[
        (rolling_volatility >= low_threshold)
        & (rolling_volatility < high_threshold)
    ] = "Medium"

    regimes[
        rolling_volatility >= high_threshold
    ] = "High"

    return regimes

def save_rolling_risk(
    rolling_volatility,
    rolling_var,
    output_path,
):
    """Save rolling risk measures to CSV."""
    rolling_risk = pd.DataFrame({
        "Rolling_Volatility": rolling_volatility,
        "Rolling_VaR_95": rolling_var,
    })

    rolling_risk.to_csv(output_path)

    return rolling_risk

if __name__ == "__main__":
    filepath = "data/historical/market_data.csv"

    df = load_historical_data(filepath)

    returns = calculate_historical_returns(df)

    summary = calculate_summary_statistics(returns)

    print("\nHistorical Return Statistics")
    print("=" * 40)
    print(summary)

    correlation = calculate_correlation_matrix(returns)

    print("\nReturn Correlation Matrix")
    print("=" * 40)
    print(correlation.round(3))

    extreme_returns = identify_extreme_returns(returns)

    print("\nExtreme Daily Returns (>5%)")
    print("=" * 40)

    if extreme_returns.empty:
        print("No extreme daily returns detected.")
    else:
        print(extreme_returns)
    

    weights = pd.Series({
        "Australian_Equity": 0.30,
        "International_Equity": 0.40,
        "Bonds": 0.20,
        "Cash": 0.10,
    })

    portfolio_returns = calculate_portfolio_returns(
        returns,
        weights,
    )

    rolling_volatility = calculate_rolling_volatility(
        portfolio_returns,
        window=63,
    )

    print("\nRolling Portfolio Volatility")
    print("=" * 40)
    print(rolling_volatility.dropna().tail(10))

    portfolio_volatility = portfolio_returns.std() * (252 ** 0.5)

    print("\nPortfolio Statistics")
    print("=" * 40)
    print(
        f"Annualised Portfolio Volatility: "
        f"{portfolio_volatility:.2%}"
    )

    historical_risk = calculate_historical_risk_metrics(
        portfolio_returns
    )

    print("\nHistorical Portfolio Risk")
    print("=" * 40)

    for metric, value in historical_risk.items():
        print(f"{metric}: {value:.4%}")

    max_drawdown = calculate_maximum_drawdown(
        portfolio_returns
    )

    print(
        f"Maximum Drawdown: "
        f"{max_drawdown:.2%}"
    )

    risk_summary = create_risk_summary(
        portfolio_returns
    )

    print("\nHistorical Portfolio Risk Summary")
    print("=" * 50)

    print(
        risk_summary.to_frame(
            name="Risk Measure"
        ).to_string()
    )

    rolling_var = calculate_rolling_var(
        portfolio_returns,
        window=252,
        confidence=0.95,
    )

    print("\nRolling 95% Historical VaR")
    print("=" * 40)
    print(rolling_var.dropna().tail(10))

    peak_date, peak_volatility = identify_peak_risk_period(
        rolling_volatility
    )

    print("\nPeak Rolling Risk")
    print("=" * 40)
    print(f"Date: {peak_date.date()}")
    print(f"Rolling Volatility: {peak_volatility:.2%}")

    risk_regimes = classify_risk_regime(
        rolling_volatility
    )

    print("\nRisk Regime Distribution")
    print("=" * 40)
    print(risk_regimes.value_counts())

    rolling_risk = save_rolling_risk(
        rolling_volatility,
        rolling_var,
        "data/historical/rolling_risk.csv",
    )

    print(
        "\nSaved rolling risk data to "
        "data/historical/rolling_risk.csv"
    )