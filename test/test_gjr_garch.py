import numpy as np
import pandas as pd
import pytest

from src.risk.gjr_garch import (
    fit_gjr_garch,
    gjr_garch_conditional_volatility,
    gjr_garch_conditional_var,
    gjr_garch_parameters,
    gjr_garch_asymmetry,
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


def test_fit_gjr_garch_contains_core_parameters(
    returns,
):
    result = fit_gjr_garch(returns)

    assert "omega" in result.params.index
    assert "alpha[1]" in result.params.index
    assert "gamma[1]" in result.params.index
    assert "beta[1]" in result.params.index


def test_gjr_conditional_volatility_is_positive(
    returns,
):
    volatility = (
        gjr_garch_conditional_volatility(
            returns
        )
    )

    valid = volatility.dropna()

    assert len(valid) > 0
    assert (valid > 0).all()


def test_gjr_conditional_volatility_is_finite(
    returns,
):
    volatility = (
        gjr_garch_conditional_volatility(
            returns
        )
    )

    valid = volatility.dropna()

    assert np.isfinite(
        valid.to_numpy()
    ).all()


def test_gjr_conditional_volatility_preserves_index(
    returns,
):
    volatility = (
        gjr_garch_conditional_volatility(
            returns
        )
    )

    assert volatility.index.equals(
        returns.index
    )


def test_gjr_conditional_var_is_positive(
    returns,
):
    var = gjr_garch_conditional_var(
        returns=returns,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    valid = var.dropna()

    assert len(valid) > 0
    assert (valid > 0).all()


def test_gjr_conditional_var_is_finite(
    returns,
):
    var = gjr_garch_conditional_var(
        returns=returns,
        portfolio_value=1_000_000,
        confidence_level=0.95,
    )

    valid = var.dropna()

    assert np.isfinite(
        valid.to_numpy()
    ).all()


def test_gjr_parameters_return_series(
    returns,
):
    parameters = gjr_garch_parameters(
        returns
    )

    assert isinstance(
        parameters,
        pd.Series,
    )


def test_gjr_parameters_contain_asymmetry(
    returns,
):
    parameters = gjr_garch_parameters(
        returns
    )

    assert "gamma[1]" in parameters.index


def test_gjr_asymmetry_is_finite(
    returns,
):
    gamma = gjr_garch_asymmetry(
        returns
    )

    assert np.isfinite(gamma)


def test_gjr_asymmetry_is_numeric(
    returns,
):
    gamma = gjr_garch_asymmetry(
        returns
    )

    assert isinstance(
        gamma,
        float,
    )


def test_gjr_model_uses_asymmetric_specification(
    returns,
):
    result = fit_gjr_garch(returns)

    assert "gamma[1]" in result.params.index
    assert result.model.volatility.__class__.__name__ == "GARCH"
    assert result.model.volatility.o == 1


def test_gjr_conditional_var_preserves_index(
    returns,
):
    var = gjr_garch_conditional_var(
        returns=returns,
        portfolio_value=1_000_000,
        confidence_level=0.99,
    )

    assert var.index.equals(
        returns.index
    )