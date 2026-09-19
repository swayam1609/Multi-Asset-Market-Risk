"""
parametric_portfolio.py

Portfolio-level Parametric VaR and Expected Shortfall.

Step 10D
--------

The portfolio is constructed from asset log returns by:

1. Converting log returns to simple returns.
2. Applying fixed portfolio weights.
3. Estimating the portfolio mean and volatility.
4. Applying the parametric normal-loss model.

Loss convention:

    L_t = -V_0 * R_p,t

Positive values represent losses.

This implementation uses the existing parametric risk
functions from src.risk.parametric.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.parametric import (
    parametric_var,
    parametric_expected_shortfall,
    parametric_var_es,
)

from src.risk.portfolio import (
    portfolio_returns,
)


def _portfolio_statistics(
    log_returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[float, float]:
    """
    Calculate portfolio mean and sample volatility
    from daily portfolio simple returns.
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    mean_return = float(
        returns.mean()
    )

    volatility = float(
        returns.std(ddof=1)
    )

    if not np.isfinite(mean_return):
        raise ValueError(
            "Portfolio mean return is non-finite."
        )

    if not np.isfinite(volatility):
        raise ValueError(
            "Portfolio volatility is non-finite."
        )

    if volatility <= 0:
        raise ValueError(
            "Portfolio volatility must be positive."
        )

    return mean_return, volatility


def parametric_portfolio_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Parametric VaR for the portfolio.
    """

    mean_return, volatility = _portfolio_statistics(
        log_returns,
        weights,
    )

    return parametric_var(
        mean_return,
        volatility,
        portfolio_value,
        confidence_level=confidence_level,
    )


def parametric_portfolio_expected_shortfall(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> float:
    """
    Calculate Parametric Expected Shortfall for the portfolio.
    """

    mean_return, volatility = _portfolio_statistics(
        log_returns,
        weights,
    )

    return parametric_expected_shortfall(
        mean_return,
        volatility,
        portfolio_value,
        confidence_level=confidence_level,
    )


def parametric_portfolio_var_es(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
) -> dict:
    """
    Calculate Parametric VaR and Expected Shortfall together.
    """

    mean_return, volatility = _portfolio_statistics(
        log_returns,
        weights,
    )

    return parametric_var_es(
        mean_return,
        volatility,
        portfolio_value,
        confidence_level=confidence_level,
    )


def parametric_portfolio_report(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
) -> pd.DataFrame:
    """
    Calculate Parametric VaR and Expected Shortfall
    at 95% and 99% confidence levels.

    Returns
    -------
    pandas.DataFrame
        One row for each confidence level.
    """

    mean_return, volatility = _portfolio_statistics(
        log_returns,
        weights,
    )

    results = []

    for confidence_level in (0.95, 0.99):

        result = parametric_var_es(
            mean_return,
            volatility,
            portfolio_value,
            confidence_level=confidence_level,
        )

        results.append(
            {
                "Confidence_Level": confidence_level,
                "VaR": float(
                    result["VaR"]
                ),
                "Expected_Shortfall": float(
                    result["Expected_Shortfall"]
                ),
                "Mean_Return": mean_return,
                "Volatility": volatility,
                "Observations": int(
                    len(log_returns)
                ),
            }
        )

    return pd.DataFrame(results)