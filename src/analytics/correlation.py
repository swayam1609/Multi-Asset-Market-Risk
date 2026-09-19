"""
correlation.py

Correlation analytics for multi-asset returns.

Functions:
    correlation_matrix
    rolling_correlation
"""

from __future__ import annotations

import pandas as pd


def correlation_matrix(
    log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate the full-sample Pearson correlation
    matrix of daily logarithmic returns.
    """

    if log_returns.empty:
        raise ValueError(
            "log_returns is empty."
        )

    if log_returns.isna().any().any():
        raise ValueError(
            "log_returns contains missing values."
        )

    return log_returns.corr(
        method="pearson"
    )


def rolling_correlation(
    log_returns: pd.DataFrame,
    asset_x: str,
    asset_y: str,
    window: int = 126,
) -> pd.Series:
    """
    Calculate trailing rolling Pearson correlation
    between two assets.

    Default window:
        126 observations ≈ six months.
    """

    if log_returns.empty:
        raise ValueError(
            "log_returns is empty."
        )

    if asset_x not in log_returns.columns:
        raise KeyError(
            f"'{asset_x}' not found in returns."
        )

    if asset_y not in log_returns.columns:
        raise KeyError(
            f"'{asset_y}' not found in returns."
        )

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    return (
        log_returns[asset_x]
        .rolling(window=window)
        .corr(
            log_returns[asset_y]
        )
    )