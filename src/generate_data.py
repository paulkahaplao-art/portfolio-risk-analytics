import numpy as np
import pandas as pd


# Make the simulation reproducible
np.random.seed(42)


# --------------------------------------------------
# 1. Create trading dates
# --------------------------------------------------

dates = pd.bdate_range(
    start="2023-01-03",
    end="2025-12-31",
)


# --------------------------------------------------
# 2. Define annual expected returns
# --------------------------------------------------

expected_returns = {
    "Australian_Equity": 0.08,
    "International_Equity": 0.09,
    "Bonds": 0.04,
    "Cash": 0.035,
}


# --------------------------------------------------
# 3. Define annual volatility
# --------------------------------------------------

volatility = {
    "Australian_Equity": 0.18,
    "International_Equity": 0.20,
    "Bonds": 0.06,
    "Cash": 0.01,
}


# --------------------------------------------------
# 4. Convert annual figures to daily
# --------------------------------------------------

trading_days = 252

daily_expected_returns = {
    asset: expected_returns[asset] / trading_days
    for asset in expected_returns
}

daily_volatility = {
    asset: volatility[asset] / np.sqrt(trading_days)
    for asset in volatility
}


# --------------------------------------------------
# 5. Generate daily returns
# --------------------------------------------------

data = pd.DataFrame(index=dates)

for asset in expected_returns:

    data[asset] = np.random.normal(
        loc=daily_expected_returns[asset],
        scale=daily_volatility[asset],
        size=len(dates),
    )


# --------------------------------------------------
# 6. Convert returns to price indices
# --------------------------------------------------

for asset in expected_returns:

    data[asset] = (
        100
        * (1 + data[asset]).cumprod()
    )


# --------------------------------------------------
# 7. Restore Date as a column
# --------------------------------------------------

data = data.reset_index()

data = data.rename(
    columns={"index": "Date"}
)


# --------------------------------------------------
# 8. Save dataset
# --------------------------------------------------

data.to_csv(
    "data/market_data.csv",
    index=False,
)


print(
    f"Generated {len(data)} daily observations."
)

print(data.head())

