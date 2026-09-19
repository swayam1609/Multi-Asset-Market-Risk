import numpy as np
import pandas as pd

from src.analytics.dependence import (
    log_to_simple_returns,
    covariance_matrix,
    rolling_covariance_matrix,
    portfolio_variance,
    portfolio_volatility,
    rolling_portfolio_volatility,
)


def make_test_returns(
    n=500,
    seed=42,
):
    rng = np.random.default_rng(seed)

    values = rng.normal(
        loc=0.0,
        scale=0.01,
        size=(n, 3),
    )

    index = pd.date_range(
        "2010-01-01",
        periods=n,
        freq="B",
    )

    return pd.DataFrame(
        values,
        index=index,
        columns=[
            "SPY",
            "EURUSD",
            "TLT",
        ],
    )


def test_log_to_simple_returns():
    returns = make_test_returns()

    simple = log_to_simple_returns(
        returns
    )

    assert simple.shape == returns.shape

    assert np.isfinite(
        simple.to_numpy()
    ).all()


def test_covariance_matrix():
    returns = make_test_returns()

    covariance = covariance_matrix(
        returns
    )

    assert covariance.shape == (3, 3)

    assert list(
        covariance.index
    ) == list(
        returns.columns
    )

    assert list(
        covariance.columns
    ) == list(
        returns.columns
    )

    assert np.isfinite(
        covariance.to_numpy()
    ).all()

    # Covariance matrix must be symmetric.
    assert np.allclose(
        covariance.to_numpy(),
        covariance.to_numpy().T,
    )


def test_annualized_covariance():
    returns = make_test_returns()

    daily = covariance_matrix(
        returns,
        annualize=False,
    )

    annualized = covariance_matrix(
        returns,
        annualize=True,
    )

    assert np.allclose(
        annualized.to_numpy(),
        daily.to_numpy() * 252,
    )


def test_rolling_covariance_matrix():
    returns = make_test_returns()

    result = rolling_covariance_matrix(
        returns,
        window=63,
    )

    assert len(result) > 0

    first_matrix = next(
        iter(result.values())
    )

    assert first_matrix.shape == (3, 3)

    assert np.allclose(
        first_matrix.to_numpy(),
        first_matrix.to_numpy().T,
    )


def test_portfolio_variance():
    returns = make_test_returns()

    weights = pd.Series(
        {
            "SPY": 1 / 3,
            "EURUSD": 1 / 3,
            "TLT": 1 / 3,
        }
    )

    variance = portfolio_variance(
        returns,
        weights,
    )

    assert np.isfinite(variance)
    assert variance >= 0


def test_portfolio_volatility():
    returns = make_test_returns()

    weights = pd.Series(
        {
            "SPY": 1 / 3,
            "EURUSD": 1 / 3,
            "TLT": 1 / 3,
        }
    )

    variance = portfolio_variance(
        returns,
        weights,
    )

    volatility = portfolio_volatility(
        returns,
        weights,
    )

    assert np.isclose(
        volatility,
        np.sqrt(variance),
    )


def test_rolling_portfolio_volatility():
    returns = make_test_returns()

    weights = pd.Series(
        {
            "SPY": 1 / 3,
            "EURUSD": 1 / 3,
            "TLT": 1 / 3,
        }
    )

    result = rolling_portfolio_volatility(
        returns,
        weights,
        window=63,
    )

    assert len(result) == len(returns)

    assert result.iloc[:62].isna().all()

    assert result.iloc[62:].notna().all()

    assert np.isfinite(
        result.iloc[62:]
    ).all()

    assert (
        result.iloc[62:] >= 0
    ).all()


def test_weights_must_sum_to_one():
    returns = make_test_returns()

    weights = pd.Series(
        {
            "SPY": 0.5,
            "EURUSD": 0.3,
            "TLT": 0.3,
        }
    )

    try:
        portfolio_variance(
            returns,
            weights,
        )

        assert False, (
            "Expected ValueError for "
            "weights not summing to 1."
        )

    except ValueError:
        pass