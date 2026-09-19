import numpy as np
import pandas as pd

from src.risk.historical import (
    historical_var,
    historical_expected_shortfall,
    historical_var_es,
)


def make_returns():
    return pd.Series(
        [
            0.02,
            0.01,
            0.005,
            0.0,
            -0.005,
            -0.01,
            -0.02,
            -0.03,
            -0.04,
            -0.05,
        ]
    )


def test_historical_var_is_positive_for_tail_loss():
    returns = make_returns()

    var = historical_var(
        returns,
        1_000_000,
        confidence_level=0.90,
    )

    assert var > 0
    assert np.isfinite(var)


def test_historical_es_is_at_least_var():
    returns = make_returns()

    var = historical_var(
        returns,
        1_000_000,
        confidence_level=0.90,
    )

    es = historical_expected_shortfall(
        returns,
        1_000_000,
        confidence_level=0.90,
    )

    assert es >= var


def test_historical_var_es_returns_both():
    returns = make_returns()

    result = historical_var_es(
        returns,
        1_000_000,
        confidence_level=0.95,
    )

    assert "VaR" in result
    assert "Expected_Shortfall" in result

    assert np.isfinite(
        result["VaR"]
    )

    assert np.isfinite(
        result["Expected_Shortfall"]
    )