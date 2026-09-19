import numpy as np
import pandas as pd
import pytest

from src.risk.portfolio import (
    portfolio_returns,
    portfolio_losses,
    portfolio_summary,
)


def make_log_returns():
    return pd.DataFrame(
        {
            "SPY": [
                0.01,
                -0.02,
                0.005,
                0.00,
            ],
            "EURUSD=X": [
                0.005,
                -0.01,
                0.002,
                -0.003,
            ],
            "TLT": [
                -0.004,
                0.008,
                0.001,
                -0.006,
            ],
        }
    )


def make_weights():
    return pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )


def test_portfolio_weights_sum_to_one():

    weights = make_weights()

    assert np.isclose(
        weights.sum(),
        1.0,
    )


def test_portfolio_returns_are_finite():

    returns = portfolio_returns(
        make_log_returns(),
        make_weights(),
    )

    assert isinstance(
        returns,
        pd.Series,
    )

    assert len(returns) == 4

    assert np.isfinite(
        returns.to_numpy()
    ).all()


def test_portfolio_return_uses_simple_returns():

    log_returns = make_log_returns()
    weights = make_weights()

    result = portfolio_returns(
        log_returns,
        weights,
    )

    expected_simple_returns = (
        np.exp(log_returns) - 1.0
    )

    expected = (
        expected_simple_returns
        .dot(weights)
    )

    pd.testing.assert_series_equal(
        result,
        expected.rename("Portfolio"),
    )


def test_portfolio_losses_follow_loss_convention():

    returns = portfolio_returns(
        make_log_returns(),
        make_weights(),
    )

    losses = portfolio_losses(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    expected = (
        -1_000_000 * returns
    ).astype(int)

    pd.testing.assert_series_equal(
        losses,
        expected.rename("Loss"),
    )


def test_positive_return_creates_negative_loss():

    log_returns = pd.DataFrame(
        {
            "SPY": [0.01],
            "EURUSD=X": [0.01],
            "TLT": [0.01],
        }
    )

    weights = pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )

    losses = portfolio_losses(
        log_returns,
        weights,
        1_000_000,
    )

    assert losses.iloc[0] < 0


def test_negative_return_creates_positive_loss():

    log_returns = pd.DataFrame(
        {
            "SPY": [-0.01],
            "EURUSD=X": [-0.01],
            "TLT": [-0.01],
        }
    )

    weights = pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )

    losses = portfolio_losses(
        log_returns,
        weights,
        1_000_000,
    )

    assert losses.iloc[0] > 0


def test_portfolio_summary_contains_required_fields():

    summary = portfolio_summary(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    required_fields = {
        "Portfolio_Value",
        "Weights",
        "Number_of_Observations",
        "Mean_Return",
        "Volatility",
        "Minimum_Return",
        "Maximum_Return",
        "Mean_Loss",
        "Maximum_Loss",
    }

    assert required_fields.issubset(
        summary.keys()
    )


def test_portfolio_summary_has_correct_value():

    summary = portfolio_summary(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert (
        summary["Portfolio_Value"]
        == 1_000_000
    )

    assert (
        summary["Number_of_Observations"]
        == 4
    )


def test_missing_weight_is_rejected():

    weights = pd.Series(
        {
            "SPY": 0.50,
            "TLT": 0.50,
        }
    )

    with pytest.raises(ValueError):
        portfolio_returns(
            make_log_returns(),
            weights,
        )


def test_weights_not_equal_to_one_are_rejected():

    weights = pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.20,
        }
    )

    with pytest.raises(ValueError):
        portfolio_returns(
            make_log_returns(),
            weights,
        )


def test_invalid_portfolio_value_is_rejected():

    with pytest.raises(ValueError):
        portfolio_losses(
            make_log_returns(),
            make_weights(),
            0,
        )