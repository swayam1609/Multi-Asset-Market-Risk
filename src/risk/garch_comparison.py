from __future__ import annotations

import numpy as np
import pandas as pd

from src.risk.garch_var import fit_garch11
from src.risk.gjr_garch import fit_gjr_garch


def compare_garch_models(
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Compare GARCH(1,1) and GJR-GARCH(1,1).

    The comparison reports log-likelihood, AIC, BIC,
    and the number of estimated parameters.

    Lower AIC/BIC indicates better fit under the
    corresponding information criterion.
    """
    garch = fit_garch11(returns)
    gjr = fit_gjr_garch(returns)

    results = []

    for model_name, result in [
        ("GARCH(1,1)", garch),
        ("GJR-GARCH(1,1)", gjr),
    ]:
        results.append(
            {
                "model": model_name,
                "log_likelihood": float(
                    result.loglikelihood
                ),
                "AIC": float(result.aic),
                "BIC": float(result.bic),
                "num_parameters": int(
                    len(result.params)
                ),
            }
        )

    return pd.DataFrame(results).set_index(
        "model"
    )


def compare_conditional_volatility(
    returns: pd.Series,
) -> pd.DataFrame:
    """
    Compare the conditional volatility estimates
    produced by GARCH(1,1) and GJR-GARCH(1,1).
    """
    garch = fit_garch11(returns)
    gjr = fit_gjr_garch(returns)

    garch_volatility = (
        garch.conditional_volatility / 100.0
    )

    gjr_volatility = (
        gjr.conditional_volatility / 100.0
    )

    comparison = pd.DataFrame(
        {
            "GARCH_Volatility": garch_volatility,
            "GJR_GARCH_Volatility": gjr_volatility,
        },
        index=returns.index,
    )

    comparison["Volatility_Difference"] = (
        comparison["GJR_GARCH_Volatility"]
        - comparison["GARCH_Volatility"]
    )

    return comparison


def volatility_difference_summary(
    returns: pd.Series,
) -> dict[str, float]:
    """
    Summarize the difference between GJR-GARCH and
    standard GARCH conditional volatility.
    """
    comparison = compare_conditional_volatility(
        returns
    )

    difference = comparison[
        "Volatility_Difference"
    ].dropna()

    return {
        "mean_difference": float(
            difference.mean()
        ),
        "mean_absolute_difference": float(
            difference.abs().mean()
        ),
        "max_absolute_difference": float(
            difference.abs().max()
        ),
        "correlation": float(
            comparison[
                [
                    "GARCH_Volatility",
                    "GJR_GARCH_Volatility",
                ]
            ].corr().iloc[0, 1]
        ),
    }