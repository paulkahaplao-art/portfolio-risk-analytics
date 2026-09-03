import pandas as pd


def validate_historical_data(df):
    """
    Validate historical portfolio price/index data.
    """

    required_columns = [
        "Date",
        "Australian_Equity",
        "International_Equity",
        "Bonds",
        "Cash",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if df["Date"].duplicated().any():
        raise ValueError("Duplicate dates detected.")

    if not df["Date"].is_monotonic_increasing:
        raise ValueError(
            "Dates are not in chronological order."
        )

    price_columns = required_columns[1:]

    if df[price_columns].isna().any().any():
        raise ValueError(
            "Missing values detected in price/index data."
        )

    if (df[price_columns] <= 0).any().any():
        raise ValueError(
            "Non-positive price/index values detected."
        )

    return True