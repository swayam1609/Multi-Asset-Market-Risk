"""
parametric.py

Parametric Value-at-Risk and Expected Shortfall
under a normal-return assumption.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.common import (
    validate_confidence_level,
    validate_portfolio_value,
)


def parametric_var(
    mean_return: float,
    volatility: float,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate one-period Parametric VaR.

    Return quantile:

        q_alpha = mu + sigma * z_(1-alpha)

    Loss VaR:

        VaR = -V0 * q_alpha
    """

    validate_confidence_level(
        confidence_level
    )

    validate_portfolio_value(
        portfolio_value
    )

    if not np.isfinite(mean_return):
        raise ValueError(
            "mean_return must be finite."
        )

    if not np.isfinite(volatility):
        raise ValueError(
            "volatility must be finite."
        )

    if volatility < 0:
        raise ValueError(
            "volatility cannot be negative."
        )

    z = norm.ppf(
        1 - confidence_level
    )

    return float(
        -portfolio_value
        * (
            mean_return
            + volatility * z
        )
    )


def parametric_expected_shortfall(
    mean_return: float,
    volatility: float,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate one-period Parametric Expected Shortfall
    under the normal-return assumption.

    ES return threshold:

        E[R | R <= q_alpha]

    For a normal distribution:

        ES_return =
        mu - sigma * phi(z_alpha)/(1-alpha)

    Loss ES is the negative of the
    corresponding return multiplied by portfolio value.
    """

    validate_confidence_level(
        confidence_level
    )

    validate_portfolio_value(
        portfolio_value
    )

    if not np.isfinite(mean_return):
        raise ValueError(
            "mean_return must be finite."
        )

    if not np.isfinite(volatility):
        raise ValueError(
            "volatility must be finite."
        )

    if volatility < 0:
        raise ValueError(
            "volatility cannot be negative."
        )

    z = norm.ppf(
        1 - confidence_level
    )

    tail_mean_return = (
        mean_return
        - volatility
        * norm.pdf(z)
        / (1 - confidence_level)
    )

    return float(
        -portfolio_value
        * tail_mean_return
    )


def parametric_var_es(
    mean_return: float,
    volatility: float,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> dict:
    """Return Parametric VaR and ES together."""

    var = parametric_var(
        mean_return,
        volatility,
        portfolio_value,
        confidence_level,
    )

    es = parametric_expected_shortfall(
        mean_return,
        volatility,
        portfolio_value,
        confidence_level,
    )

    return {
        "VaR": var,
        "Expected_Shortfall": es,
    }