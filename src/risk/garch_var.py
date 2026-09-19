from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from src.risk.conditional_var import conditional_var_normal


def validate_returns(returns: pd.Series) -> pd.Series:
    """
    Validate a return series for GARCH estimation.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError("returns must be a pandas Series.")

    if returns.empty:
        raise ValueError("returns must not be empty.")

    if returns.isna().any():
        raise ValueError("returns must not contain NaN values.")

    values = returns.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "returns must contain only finite values."
        )

    return returns.astype(float)


def fit_garch11(
    returns: pd.Series,
):
    """
    Fit a GARCH(1,1) model to a return series.

    Returns are scaled by 100 for numerical stability when
    estimating the ARCH model.
    """
    returns = validate_returns(returns)

    scaled_returns = returns * 100.0

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=0,
        q=1,
        dist="normal",
        rescale=False,
    )

    result = model.fit(
        disp="off"
    )

    return result


def garch_conditional_volatility(
    returns: pd.Series,
) -> pd.Series:
    """
    Estimate in-sample conditional volatility from GARCH(1,1).

    Volatility is returned in the original return scale.
    """
    result = fit_garch11(returns)

    volatility = result.conditional_volatility / 100.0

    volatility = pd.Series(
        volatility,
        index=returns.index,
        name="GARCH_Volatility",
    )

    return volatility


def garch_conditional_var(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
) -> pd.Series:
    """
    Calculate GARCH(1,1)-based conditional normal VaR.

    The conditional volatility is estimated from GARCH(1,1)
    and converted into a positive dollar loss threshold.
    """
    volatility = garch_conditional_volatility(
        returns
    )

    return conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
    )


def garch_parameters(
    returns: pd.Series,
) -> pd.Series:
    """
    Return the estimated GARCH(1,1) parameters.
    """
    result = fit_garch11(returns)

    return result.params.rename(
        "parameter"
    )


def garch_persistence(
    returns: pd.Series,
) -> float:
    """
    Calculate GARCH volatility persistence:

        alpha_1 + beta_1
    """
    result = fit_garch11(returns)

    alpha = float(
        result.params.get("alpha[1]", 0.0)
    )

    beta = float(
        result.params.get("beta[1]", 0.0)
    )

    return alpha + beta