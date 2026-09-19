"""
historical.py

Historical simulation Value-at-Risk and
Expected Shortfall.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.common import (
    returns_to_losses,
    validate_confidence_level,
)


def historical_var(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Historical VaR.

    VaR is the empirical loss quantile at
    the selected confidence level.
    """

    validate_confidence_level(
        confidence_level
    )

    losses = returns_to_losses(
        returns,
        portfolio_value,
    )

    return float(
        np.quantile(
            losses,
            confidence_level,
        )
    )


def historical_expected_shortfall(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Historical Expected Shortfall.

    ES is the average loss in the tail
    beyond the VaR threshold.
    """

    validate_confidence_level(
        confidence_level
    )

    losses = returns_to_losses(
        returns,
        portfolio_value,
    )

    var = historical_var(
        returns,
        portfolio_value,
        confidence_level,
    )

    tail_losses = losses[
        losses >= var
    ]

    if tail_losses.empty:
        return var

    return float(
        tail_losses.mean()
    )


def historical_var_es(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> dict:
    """Return Historical VaR and ES together."""

    var = historical_var(
        returns,
        portfolio_value,
        confidence_level,
    )

    es = historical_expected_shortfall(
        returns,
        portfolio_value,
        confidence_level,
    )

    return {
        "VaR": var,
        "Expected_Shortfall": es,
    }