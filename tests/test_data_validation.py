import pandas as pd
import pytest

from src.data_validation import validate_historical_data


def test_valid_data():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=3),
        "Australian_Equity": [100, 101, 102],
        "International_Equity": [100, 100.5, 101],
        "Bonds": [100, 100.1, 100.2],
        "Cash": [100, 100.01, 100.02],
    })

    assert validate_historical_data(df) is True


def test_missing_column():
    df = pd.DataFrame({
        "Date": pd.date_range("2025-01-01", periods=3),
        "Australian_Equity": [100, 101, 102],
        "International_Equity": [100, 100.5, 101],
        "Bonds": [100, 100.1, 100.2],
    })

    with pytest.raises(ValueError):
        validate_historical_data(df)


def test_duplicate_dates():
    df = pd.DataFrame({
        "Date": [
            "2025-01-01",
            "2025-01-01",
            "2025-01-03",
        ],
        "Australian_Equity": [100, 101, 102],
        "International_Equity": [100, 100.5, 101],
        "Bonds": [100, 100.1, 100.2],
        "Cash": [100, 100.01, 100.02],
    })

    df["Date"] = pd.to_datetime(df["Date"])

    with pytest.raises(ValueError):
        validate_historical_data(df)