import numpy as np
import pandas as pd
import pytest

from src.risk.conditional_var import (
    validate_conditional_volatility,
    conditional_var_normal,
    conditional_var_report,
)


def test_validate_conditional_volatility_accepts_valid_series():
    volatility = pd.Series(
        [0.01, 0.02, 0.015]
    )

    result = validate_conditional_volatility(
        volatility
    )

    assert result.dtype == float
    assert result.equals(
        volatility.astype(float)
    )


def test_validate_conditional_volatility_rejects_non_series():
    with pytest.raises(TypeError):
        validate_conditional_volatility(
            [0.01, 0.02]
        )


def test_validate_conditional_volatility_rejects_nan():
    volatility = pd.Series(
        [0.01, np.nan, 0.02]
    )

    with pytest.raises(ValueError):
        validate_conditional_volatility(
            volatility
        )


def test_validate_conditional_volatility_rejects_non_positive_values():
    volatility = pd.Series(
        [0.01, 0.0, 0.02]
    )

    with pytest.raises(ValueError):
        validate_conditional_volatility(
            volatility
        )


def test_conditional_var_is_positive_and_finite():
    volatility = pd.Series(
        [0.01, 0.02, 0.015]
    )

    result = conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert (result > 0).all()
    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_conditional_var_increases_with_volatility():
    volatility = pd.Series(
        [0.01, 0.02]
    )

    result = conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert result.iloc[1] > result.iloc[0]


def test_conditional_var_increases_with_confidence_level():
    volatility = pd.Series(
        [0.02]
    )

    var_95 = conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    var_99 = conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_level=0.99,
    )

    assert var_99.iloc[0] > var_95.iloc[0]


def test_conditional_var_preserves_index():
    index = pd.date_range(
        "2024-01-01",
        periods=3,
        freq="D",
    )

    volatility = pd.Series(
        [0.01, 0.02, 0.015],
        index=index,
    )

    result = conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    assert result.index.equals(index)


def test_conditional_var_report_contains_both_levels():
    volatility = pd.Series(
        [0.01, 0.02, 0.015]
    )

    report = conditional_var_report(
        conditional_volatility=volatility,
        portfolio_value=1_000_000,
        confidence_levels=(0.95, 0.99),
    )

    assert list(report.columns) == [
        "Conditional_VaR_95",
        "Conditional_VaR_99",
    ]


def test_conditional_var_rejects_invalid_portfolio_value():
    volatility = pd.Series(
        [0.01, 0.02]
    )

    with pytest.raises(ValueError):
        conditional_var_normal(
            conditional_volatility=volatility,
            portfolio_value=0,
            confidence_level=0.95,
        )


def test_conditional_var_rejects_invalid_confidence_level():
    volatility = pd.Series(
        [0.01, 0.02]
    )

    with pytest.raises(ValueError):
        conditional_var_normal(
            conditional_volatility=volatility,
            portfolio_value=1_000_000,
            confidence_level=1.0,
        )