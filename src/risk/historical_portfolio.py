"""
historical_portfolio.py

Historical portfolio Value-at-Risk (VaR) and
Expected Shortfall (ES).

Step 10C
--------

The portfolio is constructed from individual asset
log returns using the following process:

1. Convert asset log returns to simple returns.
2. Combine simple returns using fixed portfolio weights.
3. Convert portfolio returns into dollar losses.
4. Estimate Historical VaR from the empirical loss distribution.
5. Estimate Historical Expected Shortfall from losses
   beyond the VaR threshold.

Loss convention:

    L_t = -V_0 * R_p,t

Positive values represent losses.
"""

from __future__ import annotations

import pandas as pd

from src.risk.historical import (
    historical_var,
    historical_expected_shortfall,
    historical_var_es,
)
from src.risk.portfolio import (
    portfolio_returns,
    portfolio_losses,
)


def historical_portfolio_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Historical VaR for the portfolio.

    Parameters
    ----------
    log_returns:
        Asset-level logarithmic returns.

    weights:
        Portfolio weights.

    portfolio_value:
        Current portfolio value in dollars.

    confidence_level:
        VaR confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    float
        Historical VaR as a positive dollar loss.
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    return historical_var(
        returns,
        portfolio_value,
        confidence_level=confidence_level,
    )


def historical_portfolio_expected_shortfall(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Historical Expected Shortfall for the portfolio.

    Returns the average loss in the tail beyond the
    Historical VaR threshold.
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    return historical_expected_shortfall(
        returns,
        portfolio_value,
        confidence_level=confidence_level,
    )


def historical_portfolio_var_es(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> dict:
    """
    Calculate Historical VaR and Expected Shortfall together.

    Returns
    -------
    dict
        {
            "VaR": ...,
            "Expected_Shortfall": ...
        }
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    return historical_var_es(
        returns,
        portfolio_value,
        confidence_level=confidence_level,
    )


def historical_portfolio_report(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Calculate Historical VaR and ES at both 95% and 99%.

    Returns a DataFrame suitable for saving as a project
    result table.
    """

    portfolio_loss_series = portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
    )

    results = []

    for confidence_level in (0.95, 0.99):

        var = historical_portfolio_var(
            log_returns,
            weights,
            portfolio_value,
            confidence_level,
        )

        es = historical_portfolio_expected_shortfall(
            log_returns,
            weights,
            portfolio_value,
            confidence_level,
        )

        results.append(
            {
                "Confidence_Level": confidence_level,
                "VaR": float(var),
                "Expected_Shortfall": float(es),
                "Observations": int(
                    len(portfolio_loss_series)
                ),
            }
        )

    return pd.DataFrame(results)