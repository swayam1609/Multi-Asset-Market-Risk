"""
common.py

Common utilities used by the Step 10 market-risk engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_returns(
    returns: pd.Series,
) -> None:
    """Validate a one-dimensional return series."""

    if not isinstance(returns, pd.Series):
        raise TypeError(
            "returns must be a pandas Series."
        )

    if returns.empty:
        raise ValueError(
            "returns is empty."
        )

    if returns.isna().any():
        raise ValueError(
            "returns contains missing values."
        )

    if not np.isfinite(
        returns.to_numpy()
    ).all():
        raise ValueError(
            "returns contains non-finite values."
        )

    if (returns <= -1).any():
        raise ValueError(
            "Simple returns must be greater than -1."
        )


def validate_confidence_level(
    confidence_level: float,
) -> None:
    """Validate a VaR confidence level."""

    if not 0 < confidence_level < 1:
        raise ValueError(
            "confidence_level must be between 0 and 1."
        )


def validate_portfolio_value(
    portfolio_value: float,
) -> None:
    """Validate portfolio market value."""

    if not np.isfinite(portfolio_value):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )


def returns_to_losses(
    returns: pd.Series,
    portfolio_value: float,
) -> pd.Series:
    """
    Convert portfolio returns into monetary losses.

    Loss convention:

        L_t = -V_0 * r_t

    Positive values represent losses.
    Negative values represent gains.
    """

    validate_returns(returns)
    validate_portfolio_value(
        portfolio_value
    )

    losses = -portfolio_value * returns
    losses = losses.astype(int)
    losses.name = "Loss"

    return losses