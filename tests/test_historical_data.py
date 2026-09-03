import pandas as pd
import pytest

from src.historical_data import load_historical_data


def test_load_historical_data(tmp_path):

    filepath = tmp_path / "historical.csv"

    df = pd.DataFrame({
        "Date": [
            "2025-01-02",
            "2025-01-01",
            "2025-01-03",
        ],
        "Australian_Equity": [101, 100, 102],
        "International_Equity": [101, 100, 102],
        "Bonds": [100.1, 100.0, 100.2],
        "Cash": [100.01, 100.00, 100.02],
    })

    df.to_csv(filepath, index=False)

    result = load_historical_data(filepath)

    assert isinstance(result.index, pd.DatetimeIndex)

    assert result.index.is_monotonic_increasing

    assert list(result.columns) == [
        "Australian_Equity",
        "International_Equity",
        "Bonds",
        "Cash",
    ]


def test_missing_file():

    with pytest.raises(FileNotFoundError):

        load_historical_data(
            "data/historical/does_not_exist.csv"
        )