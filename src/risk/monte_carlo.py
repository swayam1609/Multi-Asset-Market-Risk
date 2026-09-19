"""
monte_carlo.py

Monte Carlo market-risk estimation.

The simulation uses a multivariate normal model for
one-day simple asset returns.

Methodology
-----------

1. Estimate the mean vector from historical simple returns.
2. Estimate the covariance matrix from historical simple returns.
3. Simulate correlated one-day asset returns.
4. Apply portfolio weights.
5. Convert simulated portfolio returns to dollar losses.
6. Estimate VaR and Expected Shortfall from the simulated
   loss distribution.

The random seed is explicitly controlled for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_simulation_inputs(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    n_simulations: int,
) -> None:
    """
    Validate Monte Carlo simulation inputs.
    """

    if not isinstance(
        mean_returns,
        pd.Series,
    ):
        raise TypeError(
            "mean_returns must be a pandas Series."
        )

    if not isinstance(
        covariance,
        pd.DataFrame,
    ):
        raise TypeError(
            "covariance must be a pandas DataFrame."
        )

    if not isinstance(
        weights,
        pd.Series,
    ):
        raise TypeError(
            "weights must be a pandas Series."
        )

    if mean_returns.empty:
        raise ValueError(
            "mean_returns is empty."
        )

    if covariance.empty:
        raise ValueError(
            "covariance is empty."
        )

    if not np.isfinite(
        mean_returns.to_numpy()
    ).all():
        raise ValueError(
            "mean_returns contains non-finite values."
        )

    if not np.isfinite(
        covariance.to_numpy()
    ).all():
        raise ValueError(
            "covariance contains non-finite values."
        )

    if not np.isfinite(
        weights.to_numpy()
    ).all():
        raise ValueError(
            "weights contain non-finite values."
        )

    if not np.isfinite(
        portfolio_value
    ):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )

    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be positive."
        )

    if not mean_returns.index.equals(
        covariance.index
    ):
        raise ValueError(
            "Mean-return and covariance indices must match."
        )

    if not covariance.index.equals(
        covariance.columns
    ):
        raise ValueError(
            "Covariance matrix must be square."
        )

    if not weights.index.equals(
        mean_returns.index
    ):
        raise ValueError(
            "Weights must match the asset columns."
        )

    if not np.isclose(
        weights.sum(),
        1.0,
    ):
        raise ValueError(
            "Weights must sum to 1.0."
        )


def simulate_multivariate_normal_returns(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Simulate correlated one-day simple returns
    from a multivariate normal distribution.

    Returns
    -------
    pandas.DataFrame
        Simulated asset returns.
    """

    if n_simulations <= 0:
        raise ValueError(
            "n_simulations must be positive."
        )

    mean = mean_returns.to_numpy(
        dtype=float
    )

    cov = covariance.to_numpy(
        dtype=float
    )

    if not np.isfinite(mean).all():
        raise ValueError(
            "Mean vector contains non-finite values."
        )

    if not np.isfinite(cov).all():
        raise ValueError(
            "Covariance matrix contains non-finite values."
        )

    # Numerical symmetry protection.
    covariance_matrix = (
        cov + cov.T
    ) / 2.0

    rng = np.random.default_rng(
        random_seed
    )

    simulated = rng.multivariate_normal(
        mean=mean,
        cov=covariance_matrix,
        size=n_simulations,
    )

    return pd.DataFrame(
        simulated,
        columns=mean_returns.index,
    )


def simulated_portfolio_returns(
    simulated_asset_returns: pd.DataFrame,
    weights: pd.Series,
) -> pd.Series:
    """
    Aggregate simulated asset simple returns
    using portfolio weights.
    """

    if simulated_asset_returns.empty:
        raise ValueError(
            "simulated_asset_returns is empty."
        )

    if not weights.index.equals(
        simulated_asset_returns.columns
    ):
        raise ValueError(
            "Weights must match simulated asset returns."
        )

    portfolio_returns = (
        simulated_asset_returns.dot(
            weights
        )
    )

    portfolio_returns.name = (
        "Portfolio"
    )

    return portfolio_returns


def simulated_losses(
    simulated_portfolio_returns_series: pd.Series,
    portfolio_value: float,
) -> pd.Series:
    """
    Convert simulated portfolio returns into
    dollar losses.

        L = -V0 * Rp
    """

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )

    losses = (
        -portfolio_value
        * simulated_portfolio_returns_series
    )

    losses.name = "Loss"

    return losses


def monte_carlo_losses(
    mean_returns: pd.Series,
    covariance: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float,
    n_simulations: int = 100_000,
    random_seed: int = 42,
) -> pd.Series:
    """
    Run the complete Monte Carlo simulation
    and return simulated portfolio losses.
    """

    validate_simulation_inputs(
        mean_returns,
        covariance,
        weights,
        portfolio_value,
        n_simulations,
    )

    simulated_asset_returns = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations,
            random_seed,
        )
    )

    portfolio_returns = (
        simulated_portfolio_returns(
            simulated_asset_returns,
            weights,
        )
    )

    return simulated_losses(
        portfolio_returns,
        portfolio_value,
    )