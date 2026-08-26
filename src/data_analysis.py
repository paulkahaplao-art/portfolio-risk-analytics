import pandas as pd

from risk_metrics import (
    annualised_volatility,
    sharpe_ratio,
    maximum_drawdown,
    tracking_error,
    information_ratio,
)


# --------------------------------------------------
# 1. Load data
# --------------------------------------------------

data = pd.read_csv(
    "data/portfolio.csv",
    parse_dates=["Date"]
)


# --------------------------------------------------
# 2. Calculate asset returns
# --------------------------------------------------

assets = [
    "Australian_Equity",
    "International_Equity",
    "Bonds",
    "Cash",
]

for asset in assets:
    data[f"{asset}_Return"] = data[asset].pct_change()


# --------------------------------------------------
# 3. Define portfolio weights
# --------------------------------------------------

weights = {
    "Australian_Equity_Return": 0.30,
    "International_Equity_Return": 0.40,
    "Bonds_Return": 0.20,
    "Cash_Return": 0.10,
}


# --------------------------------------------------
# 4. Calculate portfolio return
# --------------------------------------------------

data["Portfolio_Return"] = (
    data["Australian_Equity_Return"]
    * weights["Australian_Equity_Return"]
    + data["International_Equity_Return"]
    * weights["International_Equity_Return"]
    + data["Bonds_Return"]
    * weights["Bonds_Return"]
    + data["Cash_Return"]
    * weights["Cash_Return"]
)


# --------------------------------------------------
# 5. Define benchmark weights
# --------------------------------------------------

benchmark_weights = {
    "Australian_Equity_Return": 0.35,
    "International_Equity_Return": 0.45,
    "Bonds_Return": 0.15,
    "Cash_Return": 0.05,
}


# --------------------------------------------------
# 6. Calculate benchmark return
# --------------------------------------------------

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


# --------------------------------------------------
# 7. Calculate portfolio growth
# --------------------------------------------------

data["Portfolio_Growth"] = (
    1 + data["Portfolio_Return"].fillna(0)
).cumprod()

data["Benchmark_Growth"] = (
    1 + data["Benchmark_Return"].fillna(0)
).cumprod()

running_peak = data["Portfolio_Growth"].cummax()

data["Drawdown"] = (
    data["Portfolio_Growth"]
    / running_peak
    - 1
)

# --------------------------------------------------
# 8. Risk metrics
# --------------------------------------------------

volatility = annualised_volatility(
    data["Portfolio_Return"]
)

sharpe = sharpe_ratio(
    data["Portfolio_Return"]
)

mdd = maximum_drawdown(
    data["Portfolio_Growth"]
)

te = tracking_error(
    data["Portfolio_Return"],
    data["Benchmark_Return"]
)

ir = information_ratio(
    data["Portfolio_Return"],
    data["Benchmark_Return"]
)


# --------------------------------------------------
# 9. Report
# --------------------------------------------------

print("\n=== Portfolio Risk Report ===")

print(f"Annualised volatility: {volatility:.2%}")
print(f"Sharpe ratio: {sharpe:.2f}")
print(f"Maximum drawdown: {mdd:.2%}")
print(f"Tracking error: {te:.2%}")
print(f"Information ratio: {ir:.2f}")


# Visualisation

from visualisation import (
    plot_portfolio_vs_benchmark,
    plot_drawdown,  
    plot_rolling_volatility,
    plot_rolling_sharpe
)

plot_portfolio_vs_benchmark(data)
plot_drawdown(data)
plot_rolling_volatility(data)
plot_rolling_sharpe(data)