import pandas as pd
import numpy as np

data = pd.read_csv ("data/portfolio.csv", parse_dates = ["Date"])
print ("First five rows:")
print (data.head())

data["Australian_Return"] = data["Australian_Equity"].pct_change()
print ("Australian equity returns:")
print (data[["Date", "Australian_Return"]])



print("Data types:") 
print(data.dtypes)

print("Summary statistics:")
print(data.describe())  

data["Australian_Return"] = (
    data["Australian_Equity"].pct_change()
)

print(data)

assets = [
    "Australian_Equity",
    "International_Equity",
    "Bonds",
    "Cash"
]

for asset in assets:
    data[f"{asset}_Return"] = data[asset].pct_change()
    

print(data)

weights = {
    "Australian_Equity": 0.4,
    "International_Equity": 0.3,
    "Bonds": 0.2,
    "Cash": 0.1
}


    

data["Portfolio_Return"] = (
    data["Australian_Equity_Return"]
    * weights["Australian_Equity"]
    + data["International_Equity_Return"]
    * weights["International_Equity"]
    + data["Bonds_Return"]
    * weights["Bonds"]
    + data["Cash_Return"]
    * weights["Cash"]
)

data.replace(np.nan, 0, inplace=True)

print(
    data[["Date", "Portfolio_Return"]]
)

data["Portfolio_Growth"] = (
    1 + data["Portfolio_Return"]
).cumprod()

total_return = data["Portfolio_Growth"].iloc[-1] - 1

print(f"Total portfolio return ; {total_return:.2%}")

# Risk metrics

daily_volatility = data["Portfolio_Return"].std()

annualised_volatility = (
    daily_volatility * (252 ** 0.5)
)

print(
    f"\nDaily volatility: {daily_volatility:.2%}"
)


print(
    f"Annualised volatility: {annualised_volatility:.2%}"
)

# Sharpe ratio

annual_risk_free_rate = 0.04

daily_risk_free_rate = (
    (1 + annual_risk_free_rate) ** (1 / 252) - 1
)

excess_returns = (
    data["Portfolio_Return"]
    - daily_risk_free_rate
)

sharpe_ratio = (
    excess_returns.mean()
    / data["Portfolio_Return"].std()
    * (252 ** 0.5)
)

print(f"Sharpe ratio: {sharpe_ratio:.2f}")

# Maximum drawdown

data["Running_Peak"] = data["Portfolio_Growth"].cummax()
data["Drawdown"] = (data["Portfolio_Growth"] / data["Running_Peak"] - 1)
maximum_drawdown = data["Drawdown"].min()

print(f"Maximum drawdown: {maximum_drawdown:.2%}")

#Benchmark

benchmark_weights = {
    "Australian_Equity_Return": 0.35,
    "International_Equity_Return": 0.45,
    "Bonds_Return": 0.15,
    "Cash_Return": 0.05
}

data["Benchmark_Return"] = (
    data["Australian_Equity_Return"]
    * benchmark_weights["Australian_Equity_Return"]
    + data["International_Equity_Return"]
    * benchmark_weights["International_Equity_Return"]
    + data["Bonds_Return"]
    * benchmark_weights["Bonds_Return"]
    + data["Cash_Return"]
    * benchmark_weights["Cash_Return"]
)

data["Excess_Return"] = (
    data["Portfolio_Return"]
    - data["Benchmark_Return"]
)

print(
    data[
        [
            "Date",
            "Portfolio_Return",
            "Benchmark_Return",
            "Excess_Return"
        ]
    ]
)

# Tracking error

tracking_error_daily = (
    data["Excess_Return"].std()
)

tracking_error_annual = tracking_error_daily * (252 ** 0.5)

print(f"Tracking error (daily): {tracking_error_daily:.2%}")
print(f"Tracking error (annualised): {tracking_error_annual:.2%}")      


# Information Ratio

annualised_active_return = (
    data["Excess_Return"].mean() * 252
)

information_ratio = (
    annualised_active_return
    / tracking_error_annual
)

print(
    f"Information ratio: "
    f"{information_ratio:.2f}"
)


data["Benchmark_Growth"] = (
    1 + data["Benchmark_Return"]
).fillna(1).cumprod()

data["Portfolio_Growth"] = (
    1 + data["Portfolio_Return"].fillna(0)
).cumprod()

print(
    data[
        [
            "Date",
            "Portfolio_Growth",
            "Benchmark_Growth"
        ]
    ]
)

print("\n=== Portfolio Risk & Performance ===")

print(
    f"Total portfolio return: "
    f"{total_return:.2%}"
)

print(
    f"Annualised volatility: "
    f"{annualised_volatility:.2%}"
)

print(
    f"Sharpe ratio: "
    f"{sharpe_ratio:.2f}"
)

print(
    f"Maximum drawdown: "
    f"{maximum_drawdown:.2%}"
)

print(
    f"Annualised tracking error: "
    f"{tracking_error_annual:.2%}"
)

print(
    f"Information ratio: "
    f"{information_ratio:.2f}"
)