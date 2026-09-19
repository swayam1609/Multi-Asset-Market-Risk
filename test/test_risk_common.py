import numpy as np
import pandas as pd
import pytest

from src.risk.common import (
    returns_to_losses,
    validate_confidence_level,
    validate_portfolio_value,
)


def test_returns_to_losses():
    returns = pd.Series(
        [0.01, -0.02, 0.03]
    )

    losses = returns_to_losses(
        returns,
        1_000_000,
    )

    expected = pd.Series(
        [
            -10_000,
            20_000,
            -30_000,
        ],
        name="Loss",
    )

    pd.testing.assert_series_equal(
        losses,
        expected,
    )


def test_loss_convention():
    returns = pd.Series(
        [-0.05]
    )

    losses = returns_to_losses(
        returns,
        1_000_000,
    )

    assert losses.iloc[0] == 50_000


def test_confidence_level_validation():
    validate_confidence_level(0.95)

    with pytest.raises(ValueError):
        validate_confidence_level(0)

    with pytest.raises(ValueError):
        validate_confidence_level(1)

    with pytest.raises(ValueError):
        validate_confidence_level(1.5)


def test_portfolio_value_validation():
    validate_portfolio_value(
        1_000_000
    )

    with pytest.raises(ValueError):
        validate_portfolio_value(0)

    with pytest.raises(ValueError):
        validate_portfolio_value(-100)


def test_nonfinite_returns_rejected():
    returns = pd.Series(
        [0.01, np.nan, 0.02]
    )

    with pytest.raises(ValueError):
        returns_to_losses(
            returns,
            1_000_000,
        )