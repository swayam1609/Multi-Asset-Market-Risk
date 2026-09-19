"""
portfolio.py

Portfolio construction and portfolio-level analytics.

Important methodological principle:

Log returns are additive through TIME,
but they are NOT additive across ASSETS.

Therefore:

1. Convert individual asset log returns
   into simple returns.

2. Combine simple returns using portfolio weights.

3. Optionally convert the resulting portfolio
   simple return back into a logarithmic return.

The portfolio constructed here assumes
daily rebalancing to fixed target weights
and ignores transaction costs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def equal_weights(
    log_returns: pd.DataFrame,
) -> pd.Series:
    """
    Create equal portfolio weights.

    For N assets:

        w_i = 1 / N
    """

    if log_returns.empty:
        raise ValueError(
            "log_returns is empty."
        )

    n_assets = log_returns.shape[1]

    if n_assets == 0:
        raise ValueError(
            "No assets found."
        )

    return pd.Series(
        1.0 / n_assets,
        index=log_returns.columns,
        name="Weight",
    )


def validate_weights(
    weights: pd.Series,
    columns: pd.Index,
) -> pd.Series:
    """
    Validate and align portfolio weights.
    """

    if not isinstance(
        weights,
        pd.Series,
    ):
        raise TypeError(
            "weights must be a pandas Series."
        )

    weights = weights.reindex(
        columns
    )

    if weights.isna().any():
        raise ValueError(
            "Weights must be provided "
            "for every asset."
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
            f"Weights must sum to 1.0. "
            f"Current sum = {weights.sum():.6f}"
        )

    return weights


def log_to_simple(
    log_returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert logarithmic returns into simple returns.

        R_t = exp(r_t) - 1
    """

    return np.exp(
        log_returns
    ) - 1.0


def simple_to_log(
    simple_returns: pd.Series,
) -> pd.Series:
    """
    Convert simple returns into logarithmic returns.

        r_t = ln(1 + R_t)
    """

    if (
        simple_returns <= -1
    ).any():
        raise ValueError(
            "Simple returns must be greater than -1."
        )

    return np.log1p(
        simple_returns
    )


def portfolio_simple_returns(
    log_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Calculate daily portfolio simple returns.

        R_p,t = sum(w_i R_i,t)

    where R_i,t are individual asset simple returns.

    Fixed target weights imply daily rebalancing.
    """

    if log_returns.empty:
        raise ValueError(
            "log_returns is empty."
        )

    weights = validate_weights(
        weights,
        log_returns.columns,
    )

    simple_returns = (
        log_to_simple(
            log_returns
        )
    )

    portfolio_returns = (
        simple_returns.dot(
            weights
        )
    )

    portfolio_returns.name = (
        "Portfolio"
    )

    return portfolio_returns


def portfolio_log_returns(
    log_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Calculate portfolio logarithmic returns
    from correctly aggregated simple returns.
    """

    portfolio_simple = (
        portfolio_simple_returns(
            log_returns,
            weights,
        )
    )

    portfolio_log = (
        simple_to_log(
            portfolio_simple
        )
    )

    portfolio_log.name = (
        "Portfolio"
    )

    return portfolio_log


def portfolio_covariance(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    periods: int = 252,
) -> float:
    """
    Calculate annualized portfolio variance
    using the covariance matrix.

        sigma_p^2 = w' Sigma w

    Annualized covariance is obtained by
    multiplying the daily covariance matrix
    by the number of periods per year.
    """

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    weights = validate_weights(
        weights,
        log_returns.columns,
    )

    covariance = (
        log_returns
        .cov()
        * periods
    )

    portfolio_variance = float(
        weights.T
        @ covariance
        @ weights
    )

    return portfolio_variance


def portfolio_volatility(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    periods: int = 252,
) -> float:
    """
    Calculate annualized portfolio volatility.

        sigma_p = sqrt(w' Sigma w)
    """

    variance = (
        portfolio_covariance(
            log_returns,
            weights,
            periods,
        )
    )

    return float(
        np.sqrt(
            variance
        )
    )


def sharpe_ratio(
    portfolio_simple_returns: pd.Series,
    risk_free_rate_annual: float = 0.0,
    periods: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio
    using portfolio simple returns.

    Annualized return is estimated from the
    arithmetic mean of periodic simple returns.

        Sharpe =
            (annualized_return - rf)
            / annualized_volatility

    The current exploratory stage assumes
    a constant annual risk-free rate.
    """

    if portfolio_simple_returns.empty:
        raise ValueError(
            "portfolio_simple_returns is empty."
        )

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    if (
        portfolio_simple_returns <= -1
    ).any():
        raise ValueError(
            "Simple returns must be greater than -1."
        )

    mean_return = (
        portfolio_simple_returns.mean()
    )

    annualized_return = (
        mean_return * periods
    )

    annualized_volatility = (
        portfolio_simple_returns.std(
            ddof=1
        )
        * np.sqrt(periods)
    )

    if np.isclose(
        annualized_volatility,
        0.0,
    ):
        raise ValueError(
            "Portfolio volatility is zero."
        )

    return float(
        (
            annualized_return
            - risk_free_rate_annual
        )
        / annualized_volatility
    )