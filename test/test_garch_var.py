import numpy as np
import pandas as pd
import pytest

from src.risk.garch_var import (
    validate_returns,
    fit_garch11,
    garch_conditional_volatility,
    garch_conditional_var,
    garch_parameters,
    garch_persistence,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=500,
    )

    return pd.Series(
        values,
        index=pd.date_range(
            "2020-01-01",
            periods=500,
            freq="D",
        ),
        name="Returns",
    )


def test_validate_returns_accepts_valid_series(
    returns,
):
    result = validate_returns(returns)

    assert result.dtype == float
    assert result.equals(
        returns.astype(float)
    )


def test_validate_returns_rejects_non_series():
    with pytest.raises(TypeError):
        validate_returns(
            [0.01, 0.02, 0.03]
        )


def test_validate_returns_rejects_nan():
    returns = pd.Series(
        [0.01, np.nan, 0.02]
    )

    with pytest.raises(ValueError):
        validate_returns(returns)


def test_fit_garch11_returns_fitted_model(
    returns,
):
    result = fit_garch11(returns)

    assert hasattr(
        result,
        "params",
    )

    assert "omega" in result.params.index
    assert "alpha[1]" in result.params.index
    assert "beta[1]" in result.params.index


def test_garch_conditional_volatility_is_positive(
    returns,
):
    volatility = garch_conditional_volatility(
        returns
    )

    valid = volatility.dropna()

    assert len(valid) > 0
    assert (valid > 0).all()


def test_garch_conditional_volatility_is_finite(
    returns,
):
    volatility = garch_conditional_volatility(
        returns
    )

    valid = volatility.dropna()

    assert np.isfinite(
        valid.to_numpy()
    ).all()


def test_garch_conditional_volatility_preserves_index(
    returns,
):
    volatility = garch_conditional_volatility(
        returns
    )

    assert volatility.index.equals(
        returns.index
    )


def test_garch_conditional_var_is_positive(
    returns,
):
    var = garch_conditional_var(
        returns=returns,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    valid = var.dropna()

    assert len(valid) > 0
    assert (valid > 0).all()


def test_garch_conditional_var_is_finite(
    returns,
):
    var = garch_conditional_var(
        returns=returns,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    valid = var.dropna()

    assert np.isfinite(
        valid.to_numpy()
    ).all()


def test_garch_parameters_contain_core_parameters(
    returns,
):
    parameters = garch_parameters(
        returns
    )

    assert "omega" in parameters.index
    assert "alpha[1]" in parameters.index
    assert "beta[1]" in parameters.index


def test_garch_persistence_is_finite(
    returns,
):
    persistence = garch_persistence(
        returns
    )

    assert np.isfinite(
        persistence
    )


def test_garch_persistence_is_non_negative(
    returns,
):
    persistence = garch_persistence(
        returns
    )

    assert persistence >= 0.0