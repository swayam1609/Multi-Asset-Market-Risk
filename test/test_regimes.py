import numpy as np
import pandas as pd

from src.analytics.regimes import (
    volatility_regime_thresholds,
    classify_volatility_regimes,
    regime_summary,
    compare_reference_periods,
)


def make_volatility_series(
    n=300,
    seed=42,
):
    rng = np.random.default_rng(seed)

    values = rng.uniform(
        0.05,
        0.40,
        size=n,
    )

    index = pd.date_range(
        "2005-01-01",
        periods=n,
        freq="B",
    )

    return pd.Series(
        values,
        index=index,
        name="Volatility",
    )


def test_regime_thresholds():
    volatility = make_volatility_series()

    thresholds = (
        volatility_regime_thresholds(
            volatility
        )
    )

    assert (
        thresholds["low_threshold"]
        <= thresholds["high_threshold"]
    )

    assert np.isfinite(
        thresholds.to_numpy()
    ).all()


def test_regime_classification():
    volatility = make_volatility_series()

    regimes = classify_volatility_regimes(
        volatility
    )

    valid_regimes = {
        "Low",
        "Medium",
        "High",
    }

    assert regimes.notna().all()

    assert set(
        regimes.unique()
    ).issubset(valid_regimes)


def test_regime_summary():
    volatility = make_volatility_series()

    summary = regime_summary(
        volatility
    )

    assert not summary.empty

    assert (
        summary["count"].sum()
        == len(volatility)
    )

    assert np.isclose(
        summary["percentage"].sum(),
        100.0,
    )

    assert (
        summary["mean_volatility"]
        >= 0
    ).all()


def test_reference_period_comparison():
    index = pd.date_range(
        "2008-01-01",
        "2023-12-31",
        freq="B",
    )

    rng = np.random.default_rng(42)

    volatility = pd.Series(
        rng.uniform(
            0.05,
            0.40,
            size=len(index),
        ),
        index=index,
        name="Volatility",
    )

    result = compare_reference_periods(
        volatility
    )

    assert len(result) == 3

    assert set(
        result["period"]
    ) == {
        "Global Financial Crisis",
        "COVID-19 Shock",
        "2022 Rate and Inflation Period",
    }

    assert (
        result["observations"] > 0
    ).all()

    assert np.isfinite(
        result[
            "mean_volatility"
        ].to_numpy()
    ).all()


def test_reference_period_can_be_customized():
    volatility = make_volatility_series(
        n=500
    )

    custom_periods = {
        "Test Period": (
            "2010-01-01",
            "2010-12-31",
        ),
    }

    result = compare_reference_periods(
        volatility,
        periods=custom_periods,
    )

    assert len(result) == 1

    assert (
        result.iloc[0]["period"]
        == "Test Period"
    )