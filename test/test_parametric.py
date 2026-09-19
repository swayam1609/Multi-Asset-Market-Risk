import numpy as np

from src.risk.parametric import (
    parametric_var,
    parametric_expected_shortfall,
    parametric_var_es,
)


def test_parametric_var_positive():
    var = parametric_var(
        mean_return=0.0,
        volatility=0.01,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert var > 0
    assert np.isfinite(var)


def test_parametric_es_positive():
    es = parametric_expected_shortfall(
        mean_return=0.0,
        volatility=0.01,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert es > 0
    assert np.isfinite(es)


def test_parametric_es_exceeds_var():
    var = parametric_var(
        mean_return=0.0,
        volatility=0.01,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    es = parametric_expected_shortfall(
        mean_return=0.0,
        volatility=0.01,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert es > var


def test_parametric_bundle():
    result = parametric_var_es(
        mean_return=0.0,
        volatility=0.01,
        portfolio_value=1_000_000,
        confidence_level=0.99,
    )

    assert set(result.keys()) == {
        "VaR",
        "Expected_Shortfall",
    }

    assert result["VaR"] > 0
    assert result["Expected_Shortfall"] > result["VaR"]