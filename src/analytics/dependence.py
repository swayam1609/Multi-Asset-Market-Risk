"""
dependence.py

Dependence and portfolio-risk analytics for the
multi-asset quantitative market risk engine.

Includes:
    - Static covariance matrix
    - Rolling covariance matrices
    - Simple-return conversion
    - Portfolio variance
    - Portfolio volatility
    - Rolling portfolio volatility

Important methodological principle:

Log returns are useful for time-series calculations,
but portfolio returns must be aggregated using simple returns.

Therefore portfolio covariance and portfolio volatility
are calculated using simple returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Validation
# ============================================================

def _validate_log_returns(
    log_returns: pd.DataFrame,
) -> None:
    """Validate a DataFrame of logarithmic returns."""

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


def _validate_weights(
    weights: pd.Series,
    columns: pd.Index,
) -> pd.Series:
    """Validate and align portfolio weights."""

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "weights must be a pandas Series."
        )

    weights = weights.reindex(columns)

    if weights.isna().any():
        raise ValueError(
            "Weights must be provided for every asset."
        )

    if not np.isfinite(
        weights.to_numpy()
    ).all():
        raise ValueError(
            "Weights contain non-finite values."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):
        raise ValueError(
            "Portfolio weights must sum to 1.0."
        )

    return weights


# ============================================================
# Return conversion
# ============================================================

def log_to_simple_returns(
    log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert logarithmic returns to simple returns.

    simple return = exp(log return) - 1
    """

    _validate_log_returns(log_returns)

    return np.expm1(log_returns)


# ============================================================
# Static covariance
# ============================================================

def covariance_matrix(
    log_returns: pd.DataFrame,
    annualize: bool = False,
    periods: int = 252,
) -> pd.DataFrame:
    """
    Calculate the covariance matrix of simple returns.

    Parameters
    ----------
    log_returns:
        DataFrame containing daily log returns.

    annualize:
        If True, multiply covariance by periods.

    periods:
        Number of periods per year, normally 252.
    """

    _validate_log_returns(log_returns)

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    simple_returns = log_to_simple_returns(
        log_returns
    )

    covariance = simple_returns.cov()

    if annualize:
        covariance = covariance * periods

    return covariance


# ============================================================
# Rolling covariance
# ============================================================

def rolling_covariance_matrix(
    log_returns: pd.DataFrame,
    window: int = 126,
    annualize: bool = False,
    periods: int = 252,
) -> dict:
    """
    Calculate rolling covariance matrices.

    Returns a dictionary where each date maps to
    its covariance matrix.

    Only information inside the trailing window
    is used for each covariance estimate.
    """

    _validate_log_returns(log_returns)

    if window <= 1:
        raise ValueError(
            "window must be greater than 1."
        )

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    simple_returns = log_to_simple_returns(
        log_returns
    )

    rolling_covariance = (
        simple_returns
        .rolling(window=window)
        .cov()
    )

    if annualize:
        rolling_covariance = (
            rolling_covariance * periods
        )

    covariance_matrices = {}

    for date in log_returns.index:
        try:
            matrix = rolling_covariance.loc[date]
        except KeyError:
            continue

        if matrix.isna().all().all():
            continue

        covariance_matrices[date] = matrix

    return covariance_matrices


# ============================================================
# Portfolio variance
# ============================================================

def portfolio_variance(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    annualize: bool = False,
    periods: int = 252,
) -> float:
    """
    Calculate portfolio variance:

        w' Sigma w

    The covariance matrix is calculated from
    simple asset returns.
    """

    _validate_log_returns(log_returns)

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    weights = _validate_weights(
        weights,
        log_returns.columns,
    )

    covariance = covariance_matrix(
        log_returns,
        annualize=annualize,
        periods=periods,
    )

    variance = float(
        weights.T
        @ covariance
        @ weights
    )

    return variance


# ============================================================
# Portfolio volatility
# ============================================================

def portfolio_volatility(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    annualize: bool = False,
    periods: int = 252,
) -> float:
    """
    Calculate portfolio volatility from:

        sqrt(w' Sigma w)
    """

    variance = portfolio_variance(
        log_returns=log_returns,
        weights=weights,
        annualize=annualize,
        periods=periods,
    )

    return float(
        np.sqrt(variance)
    )


# ============================================================
# Rolling portfolio volatility
# ============================================================

def rolling_portfolio_volatility(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    window: int = 126,
    annualize: bool = True,
    periods: int = 252,
) -> pd.Series:
    """
    Calculate trailing rolling portfolio volatility.

    For each date t, only the previous `window`
    observations available at that point are used.
    """

    _validate_log_returns(log_returns)

    if window <= 1:
        raise ValueError(
            "window must be greater than 1."
        )

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    weights = _validate_weights(
        weights,
        log_returns.columns,
    )

    simple_returns = log_to_simple_returns(
        log_returns
    )

    portfolio_returns = (
        simple_returns
        .dot(weights)
    )

    rolling_std = (
        portfolio_returns
        .rolling(window=window)
        .std(ddof=1)
    )

    if annualize:
        rolling_std = (
            rolling_std
            * np.sqrt(periods)
        )

    rolling_std.name = (
        "Rolling_Portfolio_Volatility"
    )

    return rolling_std