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
