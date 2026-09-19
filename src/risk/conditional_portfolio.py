from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm

from src.analytics.portfolio import validate_weights
from src.risk.common import validate_confidence_level


def validate_portfolio_returns(
    returns: pd.DataFrame,
    weights: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Validate portfolio returns and weights.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns must not be empty.")

    if returns.isna().any().any():
        raise ValueError(
            "returns must not contain NaN values."
        )

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "returns must contain only finite values."
        )

    validated_weights = validate_weights(
        weights,
        returns.columns,
    )

    return (
        returns.astype(float),
        validated_weights.astype(float),
    )


def conditional_portfolio_volatility(
    covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
    weights: pd.Series,
) -> pd.Series:
    """
    Calculate conditional portfolio volatility:

        sigma_p,t = sqrt(w' Sigma_t w)

    where Sigma_t is the conditional covariance matrix.
    """
    if not covariance_matrices:
        raise ValueError(
            "covariance_matrices must not be empty."
        )

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "weights must be a pandas Series."
        )

    portfolio_volatility = {}

    for timestamp, covariance in covariance_matrices.items():

        if not isinstance(covariance, pd.DataFrame):
            raise TypeError(
                "Each covariance matrix must be a pandas DataFrame."
            )

        if list(covariance.index) != list(weights.index):
            raise ValueError(
                "Covariance matrix assets must match weight assets."
            )

        matrix = covariance.to_numpy(
            dtype=float
        )

        weight_vector = weights.to_numpy(
            dtype=float
        )

        variance = float(
            weight_vector.T
            @ matrix
            @ weight_vector
        )

        if variance < 0 and not np.isclose(
            variance,
            0.0,
            atol=1e-15,
        ):
            raise ValueError(
                "Conditional portfolio variance cannot be negative."
            )

        variance = max(
            variance,
            0.0,
        )

        portfolio_volatility[timestamp] = np.sqrt(
            variance
        )

    return pd.Series(
        portfolio_volatility,
        name="conditional_portfolio_volatility",
    )


def conditional_portfolio_mean(
    returns: pd.DataFrame,
    weights: pd.Series,
    window: int = 252,
) -> pd.Series:
    """
    Calculate a rolling conditional portfolio mean.

    For time t, only observations before t are used:

        mu_p,t = w' mu_t

    where mu_t is estimated from the preceding window.
    """
    returns, weights = validate_portfolio_returns(
        returns,
        weights,
    )

    if not isinstance(
        window,
        (int, np.integer),
    ):
        raise TypeError(
            "window must be an integer."
        )

    if window < 30:
        raise ValueError(
            "window must be at least 30 observations."
        )

    if len(returns) <= window:
        raise ValueError(
            "returns must contain more observations than the window."
        )

    portfolio_mean = {}

    for t in range(
        window,
        len(returns),
    ):
        historical = returns.iloc[
            t - window:t
        ]

        asset_mean = historical.mean()

        mean = float(
            weights.to_numpy(dtype=float)
            @ asset_mean.to_numpy(dtype=float)
        )

        portfolio_mean[
            returns.index[t]
        ] = mean

    return pd.Series(
        portfolio_mean,
        name="conditional_portfolio_mean",
    )


def conditional_portfolio_var(
    conditional_mean: pd.Series,
    conditional_volatility: pd.Series,
    portfolio_value: float,
    confidence_level: float,
) -> pd.Series:
    """
    Calculate conditional parametric portfolio VaR:

        VaR_t = V0 * (z_alpha * sigma_p,t - mu_p,t)

    Positive values represent dollar losses.
    """
    validate_confidence_level(
        confidence_level
    )

    if not isinstance(
        conditional_mean,
        pd.Series,
    ):
        raise TypeError(
            "conditional_mean must be a pandas Series."
        )

    if not isinstance(
        conditional_volatility,
        pd.Series,
    ):
        raise TypeError(
            "conditional_volatility must be a pandas Series."
        )

    if not np.isfinite(portfolio_value):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )

    if not conditional_mean.index.equals(
        conditional_volatility.index
    ):
        raise ValueError(
            "conditional_mean and conditional_volatility "
            "must have identical indices."
        )

    if conditional_mean.isna().any():
        raise ValueError(
            "conditional_mean must not contain NaN values."
        )

    if conditional_volatility.isna().any():
        raise ValueError(
            "conditional_volatility must not contain NaN values."
        )

    if not np.isfinite(
        conditional_mean.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "conditional_mean must contain only finite values."
        )

    if not np.isfinite(
        conditional_volatility.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "conditional_volatility must contain only finite values."
        )

    if (
        conditional_volatility < 0
    ).any():
        raise ValueError(
            "conditional_volatility must be non-negative."
        )

    z_score = norm.ppf(
        confidence_level
    )

    var = portfolio_value * (
        z_score
        * conditional_volatility
        - conditional_mean
    )

    return pd.Series(
        var,
        index=conditional_mean.index,
        name=f"Conditional_VaR_{int(confidence_level * 100)}",
    )


def conditional_portfolio_var_report(
    returns: pd.DataFrame,
    weights: pd.Series,
    covariance_matrices: dict[pd.Timestamp, pd.DataFrame],
    portfolio_value: float = 1_000_000.0,
    confidence_levels: tuple[float, ...] = (
        0.95,
        0.99,
    ),
    mean_window: int = 252,
) -> pd.DataFrame:
    """
    Generate a conditional portfolio VaR report.

    The report combines:

    - rolling portfolio conditional mean
    - conditional portfolio volatility
    - conditional VaR at requested confidence levels

    Covariance matrices must already be generated using
    the no-look-ahead conditional covariance framework.
    """
    returns, weights = validate_portfolio_returns(
        returns,
        weights,
    )

    conditional_volatility = (
        conditional_portfolio_volatility(
            covariance_matrices,
            weights,
        )
    )

    conditional_mean = (
        conditional_portfolio_mean(
            returns,
            weights,
            window=mean_window,
        )
    )

    common_index = (
        conditional_volatility.index
        .intersection(
            conditional_mean.index
        )
    )

    if len(common_index) == 0:
        raise ValueError(
            "No overlapping timestamps between "
            "conditional covariance and portfolio mean."
        )

    report = pd.DataFrame(
        index=common_index
    )

    report[
        "conditional_portfolio_mean"
    ] = conditional_mean.loc[
        common_index
    ]

    report[
        "conditional_portfolio_volatility"
    ] = conditional_volatility.loc[
        common_index
    ]

    for confidence_level in confidence_levels:

        var = conditional_portfolio_var(
            conditional_mean=report[
                "conditional_portfolio_mean"
            ],
            conditional_volatility=report[
                "conditional_portfolio_volatility"
            ],
            portfolio_value=portfolio_value,
            confidence_level=confidence_level,
        )

        report[var.name] = var

    return report