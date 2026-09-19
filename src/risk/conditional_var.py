from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.risk.common import validate_confidence_level


def validate_conditional_volatility(
    conditional_volatility: pd.Series,
) -> pd.Series:
    """
    Validate a conditional volatility series.
    """
    if not isinstance(
        conditional_volatility,
        pd.Series,
    ):
        raise TypeError(
            "conditional_volatility must be a pandas Series."
        )

    if conditional_volatility.empty:
        raise ValueError(
            "conditional_volatility must not be empty."
        )

    if conditional_volatility.isna().any():
        raise ValueError(
            "conditional_volatility must not contain NaN values."
        )

    values = conditional_volatility.to_numpy(
        dtype=float
    )

    if not np.isfinite(values).all():
        raise ValueError(
            "conditional_volatility must contain only finite values."
        )

    if (values <= 0).any():
        raise ValueError(
            "conditional_volatility must be strictly positive."
        )

    return conditional_volatility.astype(float)


def conditional_var_normal(
    conditional_volatility: pd.Series,
    portfolio_value: float,
    confidence_level: float,
    conditional_mean: float = 0.0,
) -> pd.Series:
    """
    Calculate one-step-ahead conditional VaR under
    a conditional normal-return assumption.

    Loss convention:

        Loss = -V0 * Return

    Positive VaR represents a positive dollar loss threshold.
    """
    conditional_volatility = (
        validate_conditional_volatility(
            conditional_volatility
        )
    )

    validate_confidence_level(
        confidence_level
    )

    if not isinstance(
        portfolio_value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "portfolio_value must be numeric."
        )

    if not np.isfinite(portfolio_value):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )

    if not np.isfinite(conditional_mean):
        raise ValueError(
            "conditional_mean must be finite."
        )

    z_score = norm.ppf(
        confidence_level
    )

    var_values = (
        portfolio_value
        * (
            z_score * conditional_volatility
            - conditional_mean
        )
    )

    return pd.Series(
        var_values,
        index=conditional_volatility.index,
        name=f"Conditional_VaR_{int(confidence_level * 100)}",
    )


def conditional_var_report(
    conditional_volatility: pd.Series,
    portfolio_value: float,
    confidence_levels: tuple[float, ...] = (
        0.95,
        0.99,
    ),
    conditional_mean: float = 0.0,
) -> pd.DataFrame:
    """
    Generate conditional normal VaR estimates
    for multiple confidence levels.
    """
    report = pd.DataFrame(
        index=conditional_volatility.index
    )

    for confidence_level in confidence_levels:
        var_series = conditional_var_normal(
            conditional_volatility=conditional_volatility,
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
            conditional_mean=conditional_mean,
        )

        report[var_series.name] = var_series

    return report