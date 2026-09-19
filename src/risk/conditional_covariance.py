from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.garch_var import fit_garch11


def validate_returns_dataframe(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Validate a multi-asset return DataFrame.
    """
    if not isinstance(returns, pd.DataFrame):
        raise TypeError("returns must be a pandas DataFrame.")

    if returns.empty:
        raise ValueError("returns must not be empty.")

    if returns.shape[1] < 2:
        raise ValueError(
            "returns must contain at least two assets."
        )

    if returns.isna().any().any():
        raise ValueError(
            "returns must not contain NaN values."
        )

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "returns must contain only finite values."
        )

    return returns.astype(float)


def _fit_garch_until(
    returns: pd.Series,
    end: int,
):
    """
    Fit GARCH(1,1) using observations strictly before `end`.

    This prevents future observations from influencing the
    model used for the covariance forecast at time `end`.
    """
    if end < 30:
        raise ValueError(
            "At least 30 observations are required to fit GARCH."
        )

    training_returns = returns.iloc[:end]

    return fit_garch11(
        training_returns
    )


def _one_step_ahead_garch_volatility(
    returns: pd.Series,
    end: int,
) -> float:
    """
    Calculate one-step-ahead GARCH volatility for observation `end`.

    The model is fitted only on observations 0 through end-1.
    The observation at `end` is therefore never used.
    """
    result = _fit_garch_until(
        returns,
        end,
    )

    forecast = result.forecast(
        horizon=1,
        reindex=False,
    )

    variance = float(
        forecast.variance.iloc[-1, 0]
    )

    return np.sqrt(
        variance
    ) / 100.0


def _standardized_residual_history(
    returns: pd.Series,
) -> pd.Series:
    """
    Generate standardized residuals using only past information.

    For each observation t, the GARCH model is fitted using
    observations strictly before t. The return at t is then
    standardized using its one-step-ahead conditional volatility.

    This is deliberately walk-forward and avoids look-ahead bias.
    """
    standardized = pd.Series(
        np.nan,
        index=returns.index,
        dtype=float,
    )

    minimum_observations = 30

    for t in range(
        minimum_observations,
        len(returns),
    ):
        result = _fit_garch_until(
            returns,
            t,
        )

        forecast = result.forecast(
            horizon=1,
            reindex=False,
        )

        variance = float(
            forecast.variance.iloc[-1, 0]
        )

        volatility = (
            np.sqrt(variance) / 100.0
        )

        if volatility <= 0 or not np.isfinite(volatility):
            continue

        standardized.iloc[t] = (
            returns.iloc[t] / volatility
        )

    return standardized


def standardized_residuals(
    returns: pd.DataFrame,
) -> pd.DataFrame:
    """
    Estimate walk-forward standardized residuals for each asset.

    For observation t:

        z_t = r_t / sigma_t

    where sigma_t is a one-step-ahead GARCH volatility forecast
    estimated using only information available before t.
    """
    returns = validate_returns_dataframe(
        returns
    )

    standardized = pd.DataFrame(
        index=returns.index,
        columns=returns.columns,
        dtype=float,
    )

    for asset in returns.columns:
        standardized[asset] = (
            _standardized_residual_history(
                returns[asset]
            )
        )

    return standardized


def rolling_standardized_correlation(
    returns: pd.DataFrame,
    window: int = 252,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Estimate rolling correlation matrices of standardized
    residuals using only observations available before time t.

    For time t, the estimation window is:

        t-window through t-1
    """
    returns = validate_returns_dataframe(
        returns
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

    standardized = standardized_residuals(
        returns
    )

    correlations = {}

    for t in range(
        window,
        len(returns),
    ):
        historical = standardized.iloc[
            t - window:t
        ]

        correlations[
            returns.index[t]
        ] = historical.corr()

    return correlations


def conditional_covariance_matrices(
    returns: pd.DataFrame,
    window: int = 252,
) -> dict[pd.Timestamp, pd.DataFrame]:
    """
    Construct conditional covariance matrices:

        Sigma_t = D_t R_t D_t

    where:

        D_t = diagonal matrix of one-step-ahead GARCH
              volatility forecasts

        R_t = rolling correlation matrix of standardized
              residuals estimated only from past information.

    The covariance estimate at time t uses no information
    from return t or any future return.
    """
    returns = validate_returns_dataframe(
        returns
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

    standardized = standardized_residuals(
        returns
    )

    covariance_matrices = {}

    minimum_observations = max(
        window,
        30,
    )

    for t in range(
        minimum_observations,
        len(returns),
    ):
        historical = standardized.iloc[
            t - window:t
        ]

        correlation = historical.corr()

        volatility_values = []

        for asset in returns.columns:
            volatility = (
                _one_step_ahead_garch_volatility(
                    returns[asset],
                    t,
                )
            )

            volatility_values.append(
                volatility
            )

        volatility_values = np.asarray(
            volatility_values,
            dtype=float,
        )

        diagonal = np.diag(
            volatility_values
        )

        covariance = (
            diagonal
            @ correlation.to_numpy(dtype=float)
            @ diagonal
        )

        covariance_matrices[
            returns.index[t]
        ] = pd.DataFrame(
            covariance,
            index=returns.columns,
            columns=returns.columns,
        )

    return covariance_matrices