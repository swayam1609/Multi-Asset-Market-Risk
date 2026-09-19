import numpy as np
import pandas as pd
import pytest

from src.risk.conditional_portfolio import (
    validate_portfolio_returns,
    conditional_portfolio_volatility,
    conditional_portfolio_mean,
    conditional_portfolio_var,
    conditional_portfolio_var_report,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(400, 3),
    )

    return pd.DataFrame(
        values,
        index=pd.date_range(
            "2020-01-01",
            periods=400,
            freq="D",
        ),
        columns=[
            "SPY",
            "EURUSD=X",
            "TLT",
        ],
    )


@pytest.fixture
def weights():
    return pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )


@pytest.fixture
def covariance_matrices(
    returns,
):
    matrices = {}

    covariance = pd.DataFrame(
        [
            [0.0001, 0.00001, 0.000005],
            [0.00001, 0.00009, 0.000004],
            [0.000005, 0.000004, 0.00008],
        ],
        index=returns.columns,
        columns=returns.columns,
    )

    for timestamp in returns.index[252:]:
        matrices[timestamp] = covariance.copy()

    return matrices


def test_validate_portfolio_returns_accepts_valid_data(
    returns,
    weights,
):
    result_returns, result_weights = (
        validate_portfolio_returns(
            returns,
            weights,
        )
    )

    assert result_returns.equals(
        returns.astype(float)
    )

    assert result_weights.equals(
        weights.astype(float)
    )


def test_validate_portfolio_returns_rejects_invalid_returns(
    weights,
):
    invalid_returns = np.ones(
        (100, 3)
    )

    with pytest.raises(TypeError):
        validate_portfolio_returns(
            invalid_returns,
            weights,
        )


def test_validate_portfolio_returns_rejects_nan(
    returns,
    weights,
):
    modified = returns.copy()

    modified.iloc[10, 0] = np.nan

    with pytest.raises(ValueError):
        validate_portfolio_returns(
            modified,
            weights,
        )


def test_conditional_portfolio_volatility(
    weights,
    covariance_matrices,
):
    result = conditional_portfolio_volatility(
        covariance_matrices,
        weights,
    )

    assert len(result) == len(
        covariance_matrices
    )

    assert (
        result > 0
    ).all()

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_conditional_portfolio_volatility_formula(
    weights,
):
    covariance = pd.DataFrame(
        [
            [0.0001, 0.00001, 0.000005],
            [0.00001, 0.00009, 0.000004],
            [0.000005, 0.000004, 0.00008],
        ],
        index=weights.index,
        columns=weights.index,
    )

    timestamp = pd.Timestamp(
        "2025-01-01"
    )

    result = conditional_portfolio_volatility(
        {
            timestamp: covariance
        },
        weights,
    )

    expected_variance = float(
        weights.to_numpy().T
        @ covariance.to_numpy()
        @ weights.to_numpy()
    )

    expected = np.sqrt(
        expected_variance
    )

    assert np.isclose(
        result.loc[timestamp],
        expected,
    )


def test_conditional_portfolio_mean(
    returns,
    weights,
):
    result = conditional_portfolio_mean(
        returns,
        weights,
        window=100,
    )

    assert len(result) == 300

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_conditional_portfolio_mean_no_look_ahead(
    returns,
    weights,
):
    original = conditional_portfolio_mean(
        returns,
        weights,
        window=100,
    )

    modified = returns.copy()

    modified.iloc[250, 0] = 10.0

    changed = conditional_portfolio_mean(
        modified,
        weights,
        window=100,
    )

    timestamp = original.index[50]

    assert np.isclose(
        original.loc[timestamp],
        changed.loc[timestamp],
    )


def test_conditional_portfolio_var_is_positive(
):
    index = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    mean = pd.Series(
        0.0001,
        index=index,
    )

    volatility = pd.Series(
        0.01,
        index=index,
    )

    result = conditional_portfolio_var(
        mean,
        volatility,
        portfolio_value=1_000_000.0,
        confidence_level=0.95,
    )

    assert (
        result > 0
    ).all()

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_conditional_portfolio_var_99_greater_than_95(
):
    index = pd.date_range(
        "2025-01-01",
        periods=10,
        freq="D",
    )

    mean = pd.Series(
        0.0001,
        index=index,
    )

    volatility = pd.Series(
        0.01,
        index=index,
    )

    var_95 = conditional_portfolio_var(
        mean,
        volatility,
        1_000_000.0,
        0.95,
    )

    var_99 = conditional_portfolio_var(
        mean,
        volatility,
        1_000_000.0,
        0.99,
    )

    assert (
        var_99 > var_95
    ).all()


def test_conditional_portfolio_var_rejects_negative_volatility(
):
    index = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    mean = pd.Series(
        0.0,
        index=index,
    )

    volatility = pd.Series(
        0.01,
        index=index,
    )

    volatility.iloc[0] = -0.01

    with pytest.raises(ValueError):
        conditional_portfolio_var(
            mean,
            volatility,
            1_000_000.0,
            0.95,
        )


def test_conditional_portfolio_var_rejects_mismatched_indices(
):
    index_1 = pd.date_range(
        "2025-01-01",
        periods=5,
        freq="D",
    )

    index_2 = pd.date_range(
        "2025-01-02",
        periods=5,
        freq="D",
    )

    mean = pd.Series(
        0.0,
        index=index_1,
    )

    volatility = pd.Series(
        0.01,
        index=index_2,
    )

    with pytest.raises(ValueError):
        conditional_portfolio_var(
            mean,
            volatility,
            1_000_000.0,
            0.95,
        )


def test_conditional_portfolio_var_report(
    returns,
    weights,
    covariance_matrices,
):
    result = conditional_portfolio_var_report(
        returns=returns,
        weights=weights,
        covariance_matrices=covariance_matrices,
        portfolio_value=1_000_000.0,
        confidence_levels=(0.95, 0.99),
        mean_window=252,
    )

    assert (
        "conditional_portfolio_mean"
        in result.columns
    )

    assert (
        "conditional_portfolio_volatility"
        in result.columns
    )

    assert (
        "Conditional_VaR_95"
        in result.columns
    )

    assert (
        "Conditional_VaR_99"
        in result.columns
    )

    assert len(result) > 0

    assert np.isfinite(
        result.to_numpy()
    ).all()


def test_conditional_portfolio_var_report_no_look_ahead(
    returns,
    weights,
    covariance_matrices,
):
    original = conditional_portfolio_var_report(
        returns=returns,
        weights=weights,
        covariance_matrices=covariance_matrices,
        portfolio_value=1_000_000.0,
        confidence_levels=(0.95,),
        mean_window=252,
    )

    modified_returns = returns.copy()

    # Change an observation after the first
    # portfolio VaR estimate.
    modified_returns.iloc[350, 0] = 100.0

    changed = conditional_portfolio_var_report(
        returns=modified_returns,
        weights=weights,
        covariance_matrices=covariance_matrices,
        portfolio_value=1_000_000.0,
        confidence_levels=(0.95,),
        mean_window=252,
    )

    timestamp = original.index[20]

    assert np.isclose(
        original.loc[
            timestamp,
            "Conditional_VaR_95",
        ],
        changed.loc[
            timestamp,
            "Conditional_VaR_95",
        ],
    )