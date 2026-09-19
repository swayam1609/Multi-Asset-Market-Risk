"""
volatility.py

Volatility analytics for the multi-asset market risk engine.

Includes:
    - Daily volatility
    - Annualized volatility
    - Rolling volatility
    - EWMA volatility
    - GARCH(1,1) estimation
    - One-step-ahead GARCH forecasting
    - Walk-forward GARCH forecasting

Important methodological principle:

All forecasting functions must use information available
strictly before the forecast date.

No look-ahead bias is allowed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from arch import arch_model


# ============================================================
# Validation
# ============================================================

def _validate_log_returns(log_returns: pd.DataFrame) -> None:
    """Validate a DataFrame of logarithmic returns."""

    if not isinstance(log_returns, pd.DataFrame):
        raise TypeError("log_returns must be a pandas DataFrame.")

    if log_returns.empty:
        raise ValueError("log_returns is empty.")

    if log_returns.isna().any().any():
        raise ValueError("log_returns contains missing values.")

    if not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError(
            "log_returns contains non-finite values."
        )


def _validate_return_series(
    returns_series: pd.Series,
) -> None:
    """Validate a single return series."""

    if not isinstance(returns_series, pd.Series):
        raise TypeError(
            "returns_series must be a pandas Series."
        )

    if returns_series.empty:
        raise ValueError(
            "returns_series is empty."
        )

    if returns_series.isna().any():
        raise ValueError(
            "returns_series contains missing values."
        )

    if not np.isfinite(
        returns_series.to_numpy()
    ).all():
        raise ValueError(
            "returns_series contains non-finite values."
        )


# ============================================================
# Existing Step 8 volatility functions
# ============================================================

def daily_volatility(
    log_returns: pd.DataFrame,
) -> pd.Series:
    """Calculate daily sample volatility."""

    _validate_log_returns(log_returns)

    return log_returns.std(ddof=1)


def annualized_volatility(
    log_returns: pd.DataFrame,
    periods: int = 252,
) -> pd.Series:
    """Annualize daily volatility."""

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    return (
        daily_volatility(log_returns)
        * np.sqrt(periods)
    )


def rolling_volatility(
    log_returns: pd.DataFrame,
    window: int = 63,
    periods: int = 252,
    annualize: bool = True,
) -> pd.DataFrame:
    """Calculate trailing rolling volatility."""

    _validate_log_returns(log_returns)

    if window <= 0:
        raise ValueError(
            "window must be positive."
        )

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    rolling_std = (
        log_returns
        .rolling(window=window)
        .std(ddof=1)
    )

    if annualize:
        return rolling_std * np.sqrt(periods)

    return rolling_std


# ============================================================
# EWMA volatility
# ============================================================

def ewma_variance(
    returns_series: pd.Series,
    lam: float = 0.94,
) -> pd.Series:
    """
    Calculate EWMA conditional variance.

    The variance reported at date t uses information
    through date t-1.

    Recursion:

        sigma_t^2 =
            lambda * sigma_(t-1)^2
            +
            (1-lambda) * r_(t-1)^2

    This prevents look-ahead bias.
    """

    _validate_return_series(returns_series)

    if not 0 < lam < 1:
        raise ValueError(
            "lam must be between 0 and 1."
        )

    variance = pd.Series(
        index=returns_series.index,
        dtype=float,
        name=returns_series.name,
    )

    # Initial variance estimated from the first
    # observed return.
    variance.iloc[0] = returns_series.iloc[0] ** 2

    for t in range(1, len(returns_series)):
        previous_variance = variance.iloc[t - 1]
        previous_return = returns_series.iloc[t - 1]

        variance.iloc[t] = (
            lam * previous_variance
            + (1 - lam) * previous_return ** 2
        )

    return variance


def ewma_volatility(
    returns_series: pd.Series,
    lam: float = 0.94,
    periods: int = 252,
) -> pd.Series:
    """
    Calculate annualized EWMA volatility.
    """

    if periods <= 0:
        raise ValueError(
            "periods must be positive."
        )

    variance = ewma_variance(
        returns_series=returns_series,
        lam=lam,
    )

    volatility = np.sqrt(variance)

    return volatility * np.sqrt(periods)


# ============================================================
# GARCH(1,1)
# ============================================================

def fit_garch(
    returns_series: pd.Series,
    scale: float = 100.0,
):
    """
    Fit a GARCH(1,1) model.

    Returns:
        fitted_model
        scaled_returns

    Returns are scaled by 100 before fitting so that the
    numerical optimization is better behaved.

    The original decimal returns are not modified.
    """

    _validate_return_series(returns_series)

    if scale <= 0:
        raise ValueError(
            "scale must be positive."
        )

    scaled_returns = returns_series * scale

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal",
        rescale=False,
    )

    fitted_model = model.fit(
        disp="off"
    )

    return fitted_model, scaled_returns


def garch_parameter_summary(
    fitted_model,
    scale: float = 100.0,
) -> pd.Series:
    """
    Extract and standardize GARCH parameters.

    omega_scaled:
        Omega in the scaled-return units.

    omega_decimal:
        Omega converted back to decimal-return
        variance units.

    alpha:
        Shock sensitivity.

    beta:
        Volatility persistence.

    persistence:
        alpha + beta.
    """

    params = fitted_model.params

    omega_scaled = float(
        params["omega"]
    )

    alpha = float(
        params["alpha[1]"]
    )

    beta = float(
        params["beta[1]"]
    )

    omega_decimal = (
        omega_scaled / (scale ** 2)
    )

    return pd.Series(
        {
            "omega_scaled": omega_scaled,
            "omega_decimal": omega_decimal,
            "alpha": alpha,
            "beta": beta,
            "persistence": alpha + beta,
        }
    )


# ============================================================
# Correct one-step GARCH forecast
# ============================================================

def garch_one_step_forecast(
    returns_series: pd.Series,
    scale: float = 100.0,
) -> float:
    """
    Fit GARCH(1,1) using all available observations and
    produce the variance forecast for the NEXT observation.

    Information set:

        returns[0], ..., returns[T-1]

    Forecast:

        variance[T]

    Therefore the forecast does not use the unknown
    return at T.
    """

    _validate_return_series(returns_series)

    fitted_model, scaled_returns = fit_garch(
        returns_series=returns_series,
        scale=scale,
    )

    params = fitted_model.params

    omega = float(params["omega"])
    alpha = float(params["alpha[1]"])
    beta = float(params["beta[1]"])

    last_return = float(
        scaled_returns.iloc[-1]
    )

    conditional_variance = float(
        fitted_model.conditional_volatility.iloc[-1] ** 2
    )

    forecast_variance_scaled = (
        omega
        + alpha * last_return ** 2
        + beta * conditional_variance
    )

    forecast_variance_decimal = (
        forecast_variance_scaled
        / (scale ** 2)
    )

    return float(
        forecast_variance_decimal
    )


# ============================================================
# Walk-forward GARCH forecasting
# ============================================================

def rolling_garch_forecast(
    returns_series: pd.Series,
    refit_frequency: int = 21,
    min_train_size: int = 756,
    scale: float = 100.0,
) -> pd.Series:
    """
    Generate one-step-ahead walk-forward GARCH forecasts.

    At forecast date t:

        model information contains returns[0:t]

    The forecast is:

        sigma_t^2 =
            omega
            + alpha * r_(t-1)^2
            + beta * sigma_(t-1)^2

    After the return at t becomes known, it can be used
    to construct the forecast for t+1.

    GARCH parameters are refitted every refit_frequency
    observations using an expanding training window.

    This implementation explicitly avoids look-ahead bias.
    """

    _validate_return_series(returns_series)

    if refit_frequency <= 0:
        raise ValueError(
            "refit_frequency must be positive."
        )

    if min_train_size <= 1:
        raise ValueError(
            "min_train_size must be greater than 1."
        )

    if len(returns_series) <= min_train_size:
        raise ValueError(
            "Not enough observations for the requested "
            "minimum training size."
        )

    forecasts = pd.Series(
        index=returns_series.index,
        dtype=float,
        name="GARCH_Forecast_Volatility",
    )

    current_params = None
    sigma2_previous = None
    last_refit = None

    for t in range(
        min_train_size,
        len(returns_series),
    ):

        # ----------------------------------------------------
        # Refit model periodically
        # ----------------------------------------------------

        if (
            current_params is None
            or last_refit is None
            or (t - last_refit) >= refit_frequency
        ):

            # Training data ends BEFORE forecast date t.
            train = returns_series.iloc[:t]

            fitted_model, scaled_train = fit_garch(
                returns_series=train,
                scale=scale,
            )

            params = fitted_model.params

            omega = float(
                params["omega"]
            )

            alpha = float(
                params["alpha[1]"]
            )

            beta = float(
                params["beta[1]"]
            )

            current_params = (
                omega,
                alpha,
                beta,
            )

            # sigma_(t-1)^2:
            # final conditional variance from
            # information available through t-1.
            sigma2_previous = float(
                fitted_model
                .conditional_volatility
                .iloc[-1] ** 2
            )

            last_refit = t

        else:
            omega, alpha, beta = current_params

        # ----------------------------------------------------
        # Forecast variance for date t
        # ----------------------------------------------------

        previous_return_scaled = (
            returns_series.iloc[t - 1]
            * scale
        )

        sigma2_forecast_scaled = (
            omega
            + alpha * previous_return_scaled ** 2
            + beta * sigma2_previous
        )

        # Convert variance from scaled-return units
        # back to decimal-return units.
        sigma2_forecast_decimal = (
            sigma2_forecast_scaled
            / (scale ** 2)
        )

        forecasts.iloc[t] = np.sqrt(
            sigma2_forecast_decimal
        ) * np.sqrt(252)

        # ----------------------------------------------------
        # Prepare sigma_t^2 for tomorrow.
        #
        # IMPORTANT:
        #
        # The return r_t is only used AFTER the forecast
        # for t has been produced.
        # ----------------------------------------------------

        current_return_scaled = (
            returns_series.iloc[t]
            * scale
        )

        sigma2_previous = (
            omega
            + alpha * current_return_scaled ** 2
            + beta * sigma2_forecast_scaled
        )

    return forecasts