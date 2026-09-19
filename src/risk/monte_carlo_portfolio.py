"""
monte_carlo_portfolio.py

Portfolio-level Monte Carlo VaR and Expected Shortfall.

Step 10E
--------

Historical asset log returns are converted to simple returns.
The mean vector and covariance matrix are estimated from those
simple returns.

A multivariate normal model is then used to simulate 100,000
one-day portfolio outcomes.

Default random seed:

    42

Loss convention:

    L = -V0 * Rp
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.portfolio import (
    portfolio_returns,
)

from src.risk.monte_carlo import (
    monte_carlo_losses,
)

from src.risk.historical import (
    historical_var,
    historical_expected_shortfall,
)


def monte_carlo_portfolio_losses(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> pd.Series:
    """
    Generate Monte Carlo portfolio losses.
    """

    returns = portfolio_returns(
        log_returns,
        weights,
    )

    # Convert asset-level log returns to simple returns
    # for estimation of the multivariate distribution.
    simple_returns = (
        np.exp(log_returns) - 1.0
    )

    mean_returns = (
        simple_returns.mean()
    )

    covariance = (
        simple_returns.cov()
    )

    return monte_carlo_losses(
        mean_returns=mean_returns,
        covariance=covariance,
        weights=weights,
        portfolio_value=portfolio_value,
        n_simulations=n_simulations,
        random_seed=random_seed,
    )


def monte_carlo_portfolio_var(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> float:
    """
    Calculate Monte Carlo VaR.
    """

    losses = monte_carlo_portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
        n_simulations,
        random_seed,
    )

    return historical_var(
        # Convert losses back to returns only for
        # compatibility with the existing risk API.
        -losses / portfolio_value,
        portfolio_value,
        confidence_level=confidence_level,
    )


def monte_carlo_portfolio_expected_shortfall(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> float:
    """
    Calculate Monte Carlo Expected Shortfall.
    """

    losses = monte_carlo_portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
        n_simulations,
        random_seed,
    )

    return historical_expected_shortfall(
        -losses / portfolio_value,
        portfolio_value,
        confidence_level=confidence_level,
    )


def monte_carlo_portfolio_var_es(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    confidence_level: float = 0.95,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> dict:
    """
    Calculate Monte Carlo VaR and Expected Shortfall together.
    """

    losses = monte_carlo_portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
        n_simulations,
        random_seed,
    )

    returns = (
        -losses / portfolio_value
    )

    return {
        "VaR": float(
            historical_var(
                returns,
                portfolio_value,
                confidence_level=confidence_level,
            )
        ),
        "Expected_Shortfall": float(
            historical_expected_shortfall(
                returns,
                portfolio_value,
                confidence_level=confidence_level,
            )
        ),
    }


def monte_carlo_portfolio_report(
    log_returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Calculate Monte Carlo VaR and ES at 95% and 99%.
    """

    losses = monte_carlo_portfolio_losses(
        log_returns,
        weights,
        portfolio_value,
        n_simulations,
        random_seed,
    )

    returns = (
        -losses / portfolio_value
    )

    results = []

    for confidence_level in (
        0.95,
        0.99,
    ):

        var = historical_var(
            returns,
            portfolio_value,
            confidence_level=confidence_level,
        )

        es = historical_expected_shortfall(
            returns,
            portfolio_value,
            confidence_level=confidence_level,
        )

        results.append(
            {
                "Confidence_Level": confidence_level,
                "VaR": float(var),
                "Expected_Shortfall": float(es),
                "Simulations": int(
                    n_simulations
                ),
                "Random_Seed": int(
                    random_seed
                ),
            }
        )

    return pd.DataFrame(results)