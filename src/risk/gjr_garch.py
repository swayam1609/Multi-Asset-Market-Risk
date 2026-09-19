from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model

from src.risk.conditional_var import conditional_var_normal
from src.risk.garch_var import validate_returns


def fit_gjr_garch(
    returns: pd.Series,
):
    """
    Fit a GJR-GARCH(1,1) model with normal innovations.

    The asymmetric term captures the possibility that negative
    shocks affect future volatility differently from positive shocks.
    """
    returns = validate_returns(returns)

    scaled_returns = returns * 100.0

    model = arch_model(
        scaled_returns,
        mean="Constant",
        vol="GARCH",
        p=1,
        o=1,
        q=1,
        dist="normal",
        rescale=False,
    )

    result = model.fit(
        disp="off"
    )

    return result


def gjr_garch_conditional_volatility(
    returns: pd.Series,
) -> pd.Series:
    """
    Estimate in-sample conditional volatility from GJR-GARCH(1,1).

    Volatility is returned in the original return scale.
    """
    result = fit_gjr_garch(returns)

    volatility = result.conditional_volatility / 100.0

    return pd.Series(
        volatility,
        index=returns.index,
        name="GJR_GARCH_Volatility",
    )


def gjr_garch_conditional_var(
    returns: pd.Series,
    portfolio_value: float,
    confidence_level: float,
) -> pd.Series:
    """
    Calculate GJR-GARCH conditional normal VaR.
    """
    volatility = (
        gjr_garch_conditional_volatility(
            returns
        )
    )

    return conditional_var_normal(
        conditional_volatility=volatility,
        portfolio_value=portfolio_value,
        confidence_level=confidence_level,
    )


def gjr_garch_parameters(
    returns: pd.Series,
) -> pd.Series:
    """
    Return estimated GJR-GARCH parameters.
    """
    result = fit_gjr_garch(returns)

    return result.params.rename(
        "parameter"
    )


def gjr_garch_asymmetry(
    returns: pd.Series,
) -> float:
    """
    Return the GJR-GARCH asymmetric coefficient.

    The coefficient is represented by gamma[1].
    """
    result = fit_gjr_garch(returns)

    return float(
        result.params.get(
            "gamma[1]",
            0.0,
        )
    )