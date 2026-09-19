"""
portfolio.py

Portfolio-level market-risk integration.

This module connects the portfolio construction logic from
src.analytics.portfolio with the Step 10 risk engine.

Methodological conventions
---------------------------
1. The processed asset returns are logarithmic returns.
2. Individual asset log returns are converted to simple returns.
3. Portfolio returns are aggregated using fixed target weights.
4. Portfolio loss is:

       L_t = -V_0 * R_p,t

   Positive values represent losses.
5. The portfolio assumes daily rebalancing to fixed target weights.
6. Transaction costs are ignored.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.portfolio import (
    portfolio_simple_returns,
    validate_weights,
)
from src.risk.common import (
    returns_to_losses,
    validate_portfolio_value,
)


def validate_portfolio_returns(
    log_returns: pd.DataFrame,
    weights: pd.Series,
) -> None:
    """
    Validate the inputs required for portfolio risk calculations.
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

    values = log_returns.to_numpy()

    if not np.isfinite(values).all():
        raise ValueError(
            "log_returns contains non-finite values."
        )

    validate_weights(
        weights,
        log_returns.columns,
    )


def portfolio_returns(
    log_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Calculate daily portfolio simple returns.

    Individual asset log returns are first converted
    to simple returns inside portfolio_simple_returns().

        R_p,t = sum_i w_i R_i,t
    """

    validate_portfolio_returns(
        log_returns,
        weights,
    )

    return portfolio_simple_returns(
        log_returns,
        weights,
    )


def portfolio_losses(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
) -> pd.Series:
    """
    Convert daily portfolio returns into dollar losses.

        L_t = -V_0 * R_p,t

    Positive values represent losses.
    Negative values represent gains.
    """

    validate_portfolio_value(
        portfolio_value
    )

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    return returns_to_losses(
        returns,
        portfolio_value,
    )


def portfolio_summary(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
) -> dict:
    """
    Return the main portfolio-level inputs and
    summary statistics used by the Step 10 risk engine.
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    losses = portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
    )

    return {
        "Portfolio_Value": float(portfolio_value),
        "Weights": weights.reindex(
            log_returns.columns
        ).copy(),
        "Number_of_Observations": int(
            len(returns)
        ),
        "Mean_Return": float(
            returns.mean()
        ),
        "Volatility": float(
            returns.std(ddof=1)
        ),
        "Minimum_Return": float(
            returns.min()
        ),
        "Maximum_Return": float(
            returns.max()
        ),
        "Mean_Loss": float(
            losses.mean()
        ),
        "Maximum_Loss": float(
            losses.max()
        ),
    }