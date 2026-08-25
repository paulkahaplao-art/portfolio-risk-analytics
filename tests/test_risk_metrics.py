import pandas as pd

from src.risk_metrics import (
    maximum_drawdown,
)


def test_maximum_drawdown():
    growth = pd.Series([
        1.00,
        1.10,
        1.05,
        0.90,
        1.00,
    ])

    result = maximum_drawdown(growth)

    expected = 0.90 / 1.10 - 1

    assert abs(result - expected) < 1e-10