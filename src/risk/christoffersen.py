from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2

from src.risk.common import validate_confidence_level
from src.risk.kupiec import validate_exceptions


def transition_counts(
    exceptions: pd.Series,
) -> dict[str, int]:
    """
    Calculate one-step transition counts for a binary exception series.

    The four transition counts are:

        n00 : no exception -> no exception
        n01 : no exception -> exception
        n10 : exception -> no exception
        n11 : exception -> exception
    """
    exceptions = validate_exceptions(exceptions)

    values = exceptions.to_numpy(dtype=int)

    n00 = 0
    n01 = 0
    n10 = 0
    n11 = 0

    for previous, current in zip(values[:-1], values[1:]):
        if previous == 0 and current == 0:
            n00 += 1
        elif previous == 0 and current == 1:
            n01 += 1
        elif previous == 1 and current == 0:
            n10 += 1
        elif previous == 1 and current == 1:
            n11 += 1

    return {
        "n00": n00,
        "n01": n01,
        "n10": n10,
        "n11": n11,
    }


def _log_likelihood_binomial(
    successes: int,
    trials: int,
    probability: float,
) -> float:
    """
    Numerically safe Bernoulli/binomial log-likelihood.
    """
    if trials == 0:
        return 0.0

    if probability == 0.0:
        return 0.0 if successes == 0 else -np.inf

    if probability == 1.0:
        return 0.0 if successes == trials else -np.inf

    return (
        successes * np.log(probability)
        + (trials - successes) * np.log(1.0 - probability)
    )


def christoffersen_independence(
    exceptions: pd.Series,
) -> dict[str, float]:
    """
    Perform Christoffersen's likelihood-ratio independence test.

    The null hypothesis is that VaR exceptions are independently
    distributed over time.
    """
    counts = transition_counts(exceptions)

    n00 = counts["n00"]
    n01 = counts["n01"]
    n10 = counts["n10"]
    n11 = counts["n11"]

    total_transitions = n00 + n01 + n10 + n11

    if total_transitions == 0:
        raise ValueError(
            "exceptions must contain at least two observations."
        )

    total_exceptions = n01 + n11
    total_non_exceptions = n00 + n10

    # Unconditional exception probability.
    if total_transitions > 0:
        pi = total_exceptions / total_transitions
    else:
        pi = 0.0

    # Conditional transition probabilities.
    denominator_0 = n00 + n01
    denominator_1 = n10 + n11

    pi01 = (
        n01 / denominator_0
        if denominator_0 > 0
        else 0.0
    )

    pi11 = (
        n11 / denominator_1
        if denominator_1 > 0
        else 0.0
    )

    # Likelihood under independence.
    log_likelihood_independence = (
        _log_likelihood_binomial(
            total_exceptions,
            total_transitions,
            pi,
        )
    )

    # Likelihood under the unrestricted first-order Markov model.
    log_likelihood_conditional = (
        _log_likelihood_binomial(
            n01,
            denominator_0,
            pi01,
        )
        + _log_likelihood_binomial(
            n11,
            denominator_1,
            pi11,
        )
    )

    lr_statistic = -2.0 * (
        log_likelihood_independence
        - log_likelihood_conditional
    )

    if not np.isfinite(lr_statistic):
        lr_statistic = 0.0

    lr_statistic = max(float(lr_statistic), 0.0)

    p_value = float(chi2.sf(lr_statistic, df=1))

    return {
        "n00": float(n00),
        "n01": float(n01),
        "n10": float(n10),
        "n11": float(n11),
        "pi": float(pi),
        "pi01": float(pi01),
        "pi11": float(pi11),
        "lr_independence": lr_statistic,
        "p_value_independence": p_value,
    }


def christoffersen_conditional_coverage(
    exceptions: pd.Series,
    confidence_level: float,
) -> dict[str, float]:
    """
    Perform Christoffersen's conditional coverage test.

    Conditional coverage combines:

        1. Kupiec unconditional coverage
        2. Christoffersen independence

    The resulting statistic has two degrees of freedom.
    """
    from src.risk.kupiec import kupiec_pof

    validate_confidence_level(confidence_level)

    independence = christoffersen_independence(
        exceptions,
    )

    kupiec = kupiec_pof(
        exceptions,
        confidence_level,
    )

    lr_coverage = (
        kupiec["lr_statistic"]
        + independence["lr_independence"]
    )

    lr_coverage = max(float(lr_coverage), 0.0)

    p_value_coverage = float(
        chi2.sf(lr_coverage, df=2)
    )

    return {
        **independence,
        "lr_coverage": lr_coverage,
        "p_value_coverage": p_value_coverage,
    }


def christoffersen_report(
    exceptions: pd.Series,
    confidence_levels: tuple[float, ...] = (0.95, 0.99),
) -> pd.DataFrame:
    """
    Generate Christoffersen backtesting results for
    multiple confidence levels.
    """
    results = []

    for confidence_level in confidence_levels:
        result = christoffersen_conditional_coverage(
            exceptions=exceptions,
            confidence_level=confidence_level,
        )

        result["confidence_level"] = confidence_level
        results.append(result)

    return pd.DataFrame(results).set_index(
        "confidence_level"
    )