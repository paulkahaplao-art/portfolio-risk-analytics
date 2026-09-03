from pathlib import Path

import pandas as pd


REQUIRED_COLUMNS = [
    "Date",
    "Australian_Equity",
    "International_Equity",
    "Bonds",
    "Cash",
]


def load_historical_data(filepath):
    """
    Load historical portfolio market data.

    Parameters
    ----------
    filepath : str or Path
        Location of the historical CSV file.

    Returns
    -------
    pandas.DataFrame
        Historical market data indexed by date.
    """

    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(
            f"Historical data file not found: {filepath}"
        )

    df = pd.read_csv(filepath)

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df["Date"] = pd.to_datetime(df["Date"])

    df = df.sort_values("Date")

    df = df.set_index("Date")

    return df