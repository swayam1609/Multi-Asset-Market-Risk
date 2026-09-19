"""
returns.py

Return analytics for daily logarithmic returns.

Functions:
    cumulative_growth
    total_return
    annualized_return
    rolling_return
    drawdown
    max_drawdown
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_log_returns(
    log_returns: pd.DataFrame,
) -> None:
    """
    Validate a DataFrame of logarithmic returns.
    """

    if not isinstance(log_returns, pd.DataFrame):
        raise TypeError(
            "log_returns must be a pandas DataFrame."
        )

    if log_returns.empty:
        raise ValueError(
            "log_returns is empty."
        )

    if log_returns.isna().any().any():
        raise ValueError(
            "log_returns contains missing values."
        )

    if not np.isfinite(
        log_returns.to_numpy()
    ).all():
        raise ValueError(
            "log_returns contains non-finite values."
        )


def cumulative_growth(
    log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate growth of 1 unit invested at the
    beginning of the sample.

    For logarithmic returns:

        Growth_t = exp(sum(r_1, ..., r_t))
    """

    _validate_log_returns(log_returns)

    return np.exp(
        log_returns.cumsum()
    )


def total_return(
    log_returns: pd.DataFrame,
) -> pd.Series:
    """
    Calculate total cumulative simple return.

        Total Return = Growth_T - 1
    """

    growth = cumulative_growth(
        log_returns
    )

    return growth.iloc[-1] - 1.0


def annualized_return(
    log_returns: pd.DataFrame,
    periods: int = 252,
) -> pd.Series:
    """
    Calculate annualized geometric return.

    The mean daily logarithmic return is
    annualized and converted back to a simple return:

        Annualized Return =
            exp(mean(log_return) * periods) - 1
    """

    _validate_log_returns(log_returns)

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    mean_daily_log = (
        log_returns.mean()
    )

    annualized_log = (
        mean_daily_log * periods
    )

    return np.exp(
        annualized_log
    ) - 1.0


def rolling_return(
    log_returns: pd.DataFrame,
    window: int,
) -> pd.DataFrame:
    """
    Calculate trailing cumulative simple return
    over a rolling window.

        Rolling Return =
            exp(sum(window log returns)) - 1
    """

    _validate_log_returns(log_returns)

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    return (
        np.exp(
            log_returns
            .rolling(window=window)
            .sum()
        )
        - 1.0
    )


def drawdown(
    growth_index: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate drawdown from a growth index.

        Drawdown_t =
            Growth_t / RunningMax_t - 1

    Drawdown values are zero at historical peaks
    and negative below previous peaks.
    """

    if not isinstance(
        growth_index,
        pd.DataFrame,
    ):
        raise TypeError(
            "growth_index must be a pandas DataFrame."
        )

    if growth_index.empty:
        raise ValueError(
            "growth_index is empty."
        )

    if growth_index.isna().any().any():
        raise ValueError(
            "growth_index contains missing values."
        )

    running_max = (
        growth_index.cummax()
    )

    return (
        growth_index
        / running_max
        - 1.0
    )


def max_drawdown(
    growth_index: pd.DataFrame,
) -> pd.Series:
    """
    Calculate maximum drawdown for each asset.
    """

    dd = drawdown(
        growth_index
    )

    return dd.min()