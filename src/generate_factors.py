import numpy as np
import pandas as pd


def generate_factor_data(
    dates,
    seed=42,
):
    """Generate synthetic factor returns."""

    rng = np.random.default_rng(seed)

    n = len(dates)

    equity = rng.normal(
        0.0003,
        0.0100,
        n,
    )

    rates = rng.normal(
        0.0000,
        0.0040,
        n,
    )

    credit = rng.normal(
        0.0001,
        0.0050,
        n,
    )

    return pd.DataFrame(
        {
            "Date": dates,
            "Equity": equity,
            "Rates": rates,
            "Credit": credit,
        }
    )


if __name__ == "__main__":

    historical_data = pd.read_csv(
        "data/historical/market_data.csv"
    )

    dates = pd.to_datetime(
        historical_data["Date"]
    )

    factor_data = generate_factor_data(
        dates
    )

    factor_data.to_csv(
        "data/historical/factor_data.csv",
        index=False,
    )

    print(
        "Factor data created:"
    )

    print(
        factor_data.head()
    )