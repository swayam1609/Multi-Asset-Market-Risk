"""
regimes.py

Volatility-regime and historical-period analysis for the
multi-asset quantitative market risk engine.

Includes:
    - Quantile-based volatility regimes
    - Regime thresholds
    - Regime summaries
    - Historical reference-period comparisons

Important methodological principle:

Regime thresholds are derived from the supplied data.
They are not hard-coded.

Historical periods are reference windows for analysis,
not assumptions about which regime they must belong to.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Validation
# ============================================================

def _validate_series(
    series: pd.Series,
) -> None:
    """Validate a numerical pandas Series."""

    if not isinstance(series, pd.Series):
        raise TypeError(
            "series must be a pandas Series."
        )

    if series.empty:
        raise ValueError(
            "series is empty."
        )

    if not np.isfinite(
        series.dropna().to_numpy()
    ).all():
        raise ValueError(
            "series contains non-finite values."
        )


# ============================================================
# Regime thresholds
# ============================================================

def volatility_regime_thresholds(
    volatility: pd.Series,
) -> pd.Series:
    """
    Calculate data-driven volatility regime thresholds.

    Low-volatility threshold:
        33rd percentile

    High-volatility threshold:
        67th percentile

    Values between the two thresholds form the
    medium-volatility regime.
    """

    _validate_series(volatility)

    clean = volatility.dropna()

    if clean.empty:
        raise ValueError(
            "volatility contains no valid observations."
        )

    low_threshold = float(
        clean.quantile(1 / 3)
    )

    high_threshold = float(
        clean.quantile(2 / 3)
    )

    if low_threshold > high_threshold:
        raise ValueError(
            "Low threshold cannot exceed high threshold."
        )

    return pd.Series(
        {
            "low_threshold": low_threshold,
            "high_threshold": high_threshold,
        },
        dtype=float,
    )


# ============================================================
# Regime classification
# ============================================================

def classify_volatility_regimes(
    volatility: pd.Series,
) -> pd.Series:
    """
    Classify volatility observations into:

        Low
        Medium
        High

    using the 33rd and 67th percentile thresholds.
    """

    _validate_series(volatility)

    thresholds = volatility_regime_thresholds(
        volatility
    )

    low_threshold = thresholds[
        "low_threshold"
    ]

    high_threshold = thresholds[
        "high_threshold"
    ]

    regimes = pd.Series(
        index=volatility.index,
        dtype="object",
        name="Volatility_Regime",
    )

    valid = volatility.notna()

    regimes.loc[
        valid
        & (
            volatility
            < low_threshold
        )
    ] = "Low"

    regimes.loc[
        valid
        & (
            volatility
            >= low_threshold
        )
        & (
            volatility
            <= high_threshold
        )
    ] = "Medium"

    regimes.loc[
        valid
        & (
            volatility
            > high_threshold
        )
    ] = "High"

    return regimes


# ============================================================
# Regime summary
# ============================================================

def regime_summary(
    volatility: pd.Series,
) -> pd.DataFrame:
    """
    Summarize observations by volatility regime.

    Returns:
        regime
        count
        percentage
        mean_volatility
        median_volatility
        min_volatility
        max_volatility
    """

    _validate_series(volatility)

    regimes = classify_volatility_regimes(
        volatility
    )

    data = pd.DataFrame(
        {
            "volatility": volatility,
            "regime": regimes,
        }
    ).dropna(
        subset=["volatility", "regime"]
    )

    if data.empty:
        raise ValueError(
            "No valid observations available."
        )

    summary = (
        data
        .groupby("regime")["volatility"]
        .agg(
            count="count",
            mean_volatility="mean",
            median_volatility="median",
            min_volatility="min",
            max_volatility="max",
        )
    )

    summary["percentage"] = (
        summary["count"]
        / summary["count"].sum()
        * 100
    )

    desired_order = [
        "Low",
        "Medium",
        "High",
    ]

    summary = (
        summary
        .reindex(desired_order)
        .dropna(how="all")
    )

    return summary[
        [
            "count",
            "percentage",
            "mean_volatility",
            "median_volatility",
            "min_volatility",
            "max_volatility",
        ]
    ]


# ============================================================
# Historical reference periods
# ============================================================

REFERENCE_PERIODS = {
    "Global Financial Crisis": (
        "2008-01-01",
        "2009-06-30",
    ),
    "COVID-19 Shock": (
        "2020-02-01",
        "2020-05-31",
    ),
    "2022 Rate and Inflation Period": (
        "2022-01-01",
        "2022-12-31",
    ),
}


def compare_reference_periods(
    volatility: pd.Series,
    periods: dict | None = None,
) -> pd.DataFrame:
    """
    Compare volatility statistics across historical
    reference periods.

    Parameters
    ----------
    volatility:
        Volatility time series.

    periods:
        Optional dictionary:

            {
                "Period Name": (
                    "YYYY-MM-DD",
                    "YYYY-MM-DD"
                )
            }

    Returns
    -------
    DataFrame containing:

        start
        end
        observations
        mean_volatility
        median_volatility
        max_volatility
        min_volatility
    """

    _validate_series(volatility)

    if periods is None:
        periods = REFERENCE_PERIODS

    rows = []

    for period_name, (
        start_date,
        end_date,
    ) in periods.items():

        period_data = volatility.loc[
            start_date:end_date
        ].dropna()

        if period_data.empty:
            rows.append(
                {
                    "period": period_name,
                    "start": start_date,
                    "end": end_date,
                    "observations": 0,
                    "mean_volatility": np.nan,
                    "median_volatility": np.nan,
                    "max_volatility": np.nan,
                    "min_volatility": np.nan,
                }
            )

            continue

        rows.append(
            {
                "period": period_name,
                "start": start_date,
                "end": end_date,
                "observations": len(
                    period_data
                ),
                "mean_volatility": (
                    period_data.mean()
                ),
                "median_volatility": (
                    period_data.median()
                ),
                "max_volatility": (
                    period_data.max()
                ),
                "min_volatility": (
                    period_data.min()
                ),
            }
        )

    return pd.DataFrame(rows)