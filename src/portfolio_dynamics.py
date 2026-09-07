import pandas as pd


WEIGHTS = pd.Series({
    "Australian_Equity": 0.30,
    "International_Equity": 0.40,
    "Bonds": 0.20,
    "Cash": 0.10,
})


def calculate_rebalanced_returns(asset_returns, weights):
    """
    Calculate portfolio returns assuming the portfolio is
    rebalanced to the target weights every period.
    """
    weights = weights.reindex(asset_returns.columns)

    if weights.isna().any():
        raise ValueError("Weights are missing one or more assets.")

    return asset_returns.mul(weights, axis=1).sum(axis=1)


def calculate_buy_and_hold_values(
    asset_returns,
    weights,
    initial_value=100_000,
):
    """
    Calculate portfolio value assuming no rebalancing.
    """

    weights = weights.reindex(asset_returns.columns)

    if weights.isna().any():
        raise ValueError("Weights are missing one or more assets.")

    if not abs(weights.sum() - 1.0) < 1e-10:
        raise ValueError("Portfolio weights must sum to 1.")

    holdings = weights * initial_value

    values = []

    for date, returns in asset_returns.iterrows():
        holdings = holdings * (1 + returns)

        total_value = holdings.sum()

        values.append(total_value)

    return pd.Series(
        values,
        index=asset_returns.index,
        name="Portfolio Value",
    )


def calculate_buy_and_hold_returns(
    asset_returns,
    weights,
    initial_value=100_000,
):
    """
    Calculate returns of a buy-and-hold portfolio.
    """

    values = calculate_buy_and_hold_values(
        asset_returns,
        weights,
        initial_value,
    )

    previous_value = values.shift(1)

    returns = values / previous_value - 1

    returns.iloc[0] = (
        values.iloc[0] / initial_value - 1
    )

    return returns


def calculate_buy_and_hold_weights(
    asset_returns,
    weights,
    initial_value=100_000,
):
    """
    Calculate how portfolio weights drift over time
    under a buy-and-hold strategy.
    """

    weights = weights.reindex(asset_returns.columns)

    if weights.isna().any():
        raise ValueError("Weights are missing one or more assets.")

    holdings = weights * initial_value

    records = []

    for date, returns in asset_returns.iterrows():

        holdings = holdings * (1 + returns)

        total_value = holdings.sum()

        current_weights = holdings / total_value

        records.append(current_weights)

    return pd.DataFrame(
        records,
        index=asset_returns.index,
    )


def compare_portfolio_strategies(
    asset_returns,
    weights,
    initial_value=100_000,
):
    """
    Compare daily rebalancing with buy-and-hold.
    """

    rebalanced_returns = calculate_rebalanced_returns(
        asset_returns,
        weights,
    )

    buy_and_hold_values = calculate_buy_and_hold_values(
        asset_returns,
        weights,
        initial_value,
    )

    buy_and_hold_returns = (
        buy_and_hold_values
        .pct_change()
        .fillna(
            buy_and_hold_values.iloc[0] / initial_value - 1
        )
    )

    comparison = pd.DataFrame({
        "Rebalanced Return": rebalanced_returns,
        "Buy and Hold Return": buy_and_hold_returns,
    })

    comparison["Return Difference"] = (
        comparison["Rebalanced Return"]
        - comparison["Buy and Hold Return"]
    )

    return comparison


if __name__ == "__main__":

    from src.historical_data import load_historical_data

    historical_data = load_historical_data(
        "data/historical/market_data.csv"
    )

    asset_returns = historical_data.pct_change().dropna()

    print("\nPORTFOLIO WEIGHT DYNAMICS")
    print("=" * 80)

    print("\nInitial Target Weights")
    print("-" * 80)
    print(WEIGHTS)

    # Rebalanced portfolio
    rebalanced_returns = calculate_rebalanced_returns(
        asset_returns,
        WEIGHTS,
    )

    # Buy-and-hold portfolio
    buy_and_hold_values = calculate_buy_and_hold_values(
        asset_returns,
        WEIGHTS,
        initial_value=100_000,
    )

    buy_and_hold_weights = calculate_buy_and_hold_weights(
        asset_returns,
        WEIGHTS,
        initial_value=100_000,
    )

    print("\nFinal Buy-and-Hold Portfolio Value")
    print("-" * 80)
    print(f"${buy_and_hold_values.iloc[-1]:,.2f}")

    print("\nFinal Buy-and-Hold Weights")
    print("-" * 80)
    print(
        buy_and_hold_weights.iloc[-1].to_string(
            float_format=lambda x: f"{x:.2%}"
        )
    )

    print("\nWeight Drift")
    print("-" * 80)

    weight_drift = (
        buy_and_hold_weights.iloc[-1]
        - WEIGHTS
    )

    print(
        weight_drift.to_string(
            float_format=lambda x: f"{x:+.2%}"
        )
    )

    print("\nAnnualised Volatility")
    print("-" * 80)

    rebalanced_volatility = (
        rebalanced_returns.std()
        * (252 ** 0.5)
    )

    buy_and_hold_returns = (
        buy_and_hold_values.pct_change().dropna()
    )

    buy_and_hold_volatility = (
        buy_and_hold_returns.std()
        * (252 ** 0.5)
    )

    print(
        f"Rebalanced:  {rebalanced_volatility:.2%}"
    )

    print(
        f"Buy-and-hold: {buy_and_hold_volatility:.2%}"
    )