from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.risk.common import validate_confidence_level


def validate_exceptions(exceptions: pd.Series) -> pd.Series:
    """
    Validate a VaR exception series.

    Exceptions must be binary:
        1 = VaR breach
        0 = no VaR breach
    """
    if not isinstance(exceptions, pd.Series):
        raise TypeError("exceptions must be a pandas Series.")

    if exceptions.empty:
        raise ValueError("exceptions must not be empty.")

    if exceptions.isna().any():
        raise ValueError("exceptions must not contain NaN values.")

    values = exceptions.to_numpy()

    if not np.isin(values, [0, 1, False, True]).all():
        raise ValueError("exceptions must contain only 0/1 values.")

    return exceptions.astype(int)


def kupiec_pof(
    exceptions: pd.Series,
    confidence_level: float,
) -> dict[str, float]:
    """
    Calculate the Kupiec Proportion of Failures (POF) test.

    The test compares the observed VaR exception frequency with
    the theoretically expected exception probability.

    Parameters
    ----------
    exceptions : pd.Series
        Binary series where 1 indicates a VaR exception.
    confidence_level : float
        VaR confidence level, e.g. 0.95 or 0.99.

    Returns
    -------
    dict
        Number of observations, exceptions, observed exception rate,
        expected exception rate, likelihood-ratio statistic, and p-value.
    """
    exceptions = validate_exceptions(exceptions)
    validate_confidence_level(confidence_level)

    n = len(exceptions)
    exception_count = int(exceptions.sum())

    expected_exception_rate = 1.0 - confidence_level
    observed_exception_rate = exception_count / n

    # Unrestricted maximum-likelihood exception probability.
    unrestricted_rate = observed_exception_rate

    def log_likelihood(x: int, total: int, probability: float) -> float:
        """
        Calculate Bernoulli log-likelihood while handling
        zero-probability boundary cases safely.
        """
        if probability == 0.0:
            return 0.0 if x == 0 else -np.inf

        if probability == 1.0:
            return 0.0 if x == total else -np.inf

        return (
            x * np.log(probability)
            + (total - x) * np.log(1.0 - probability)
        )

    log_likelihood_null = log_likelihood(
        exception_count,
        n,
        expected_exception_rate,
    )

    log_likelihood_unrestricted = log_likelihood(
        exception_count,
        n,
        unrestricted_rate,
    )

    lr_statistic = -2.0 * (
        log_likelihood_null
        - log_likelihood_unrestricted
    )

    # Numerical protection for boundary cases.
    lr_statistic = max(float(lr_statistic), 0.0)

    p_value = float(chi2.sf(lr_statistic, df=1))

    return {
        "observations": float(n),
        "exceptions": float(exception_count),
        "observed_exception_rate": float(observed_exception_rate),
        "expected_exception_rate": float(expected_exception_rate),
        "lr_statistic": lr_statistic,
        "p_value": p_value,
    }


def kupiec_pof_report(
    exceptions: pd.Series,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Generate Kupiec POF results for multiple confidence levels.

    The supplied exception series should correspond to the VaR
    confidence level being tested.
    """
    results = []

    for confidence_level in confidence_levels:
        result = kupiec_pof(
            exceptions=exceptions,
            confidence_level=confidence_level,
        )

        result["confidence_level"] = confidence_level
        results.append(result)

    return pd.DataFrame(results).set_index("confidence_level")