from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.common import validate_confidence_level


def validate_rolling_window(
    window: int,
    minimum_window: int = 30,
) -> int:
    """
    Validate the rolling estimation window.
    """
    if not isinstance(window, (int, np.integer)):
        raise TypeError("window must be an integer.")

    if window < minimum_window:
        raise ValueError(
            f"window must be at least {minimum_window} observations."
        )

    return int(window)


def rolling_historical_var(
    losses: pd.Series,
    confidence_level: float,
    window: int = 756,
) -> pd.Series:
    """
    Calculate rolling historical VaR using only past observations.

    The VaR for observation t is estimated from observations
    t-window through t-1. The current observation is never included
    in its own VaR estimate.
    """
    if not isinstance(losses, pd.Series):
        raise TypeError("losses must be a pandas Series.")

    if losses.empty:
        raise ValueError("losses must not be empty.")

    if losses.isna().any():
        raise ValueError("losses must not contain NaN values.")

    if not np.isfinite(losses.to_numpy(dtype=float)).all():
        raise ValueError("losses must contain only finite values.")

    validate_confidence_level(confidence_level)
    window = validate_rolling_window(window)

    if len(losses) <= window:
        raise ValueError(
            "losses must contain more observations than the rolling window."
        )

    var_values = np.full(len(losses), np.nan, dtype=float)

    loss_values = losses.to_numpy(dtype=float)

    for t in range(window, len(loss_values)):
        estimation_losses = pd.Series(
            loss_values[t - window:t],
            index=losses.index[t - window:t],
            name=losses.name,
        )

        var_values[t] = estimation_losses.quantile(
            confidence_level,
        )

    return pd.Series(
        var_values,
        index=losses.index,
        name=f"VaR_{int(confidence_level * 100)}",
    )


def rolling_historical_var_report(
    losses: pd.Series,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
    window: int = 756,
) -> pd.DataFrame:
    """
    Generate rolling historical VaR estimates for multiple confidence levels.
    """
    window = validate_rolling_window(window)

    report = pd.DataFrame(index=losses.index)

    for confidence_level in confidence_levels:
        validate_confidence_level(confidence_level)

        var_series = rolling_historical_var(
            losses=losses,
            confidence_level=confidence_level,
            window=window,
        )

        report[var_series.name] = var_series

    return report