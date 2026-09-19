import numpy as np
import pandas as pd
import pytest

from src.risk.rolling_var import (
    rolling_historical_var,
    rolling_historical_var_report,
    validate_rolling_window,
)


def test_validate_rolling_window_accepts_valid_window():
    assert validate_rolling_window(30) == 30
    assert validate_rolling_window(756) == 756


def test_validate_rolling_window_rejects_non_integer():
    with pytest.raises(TypeError):
        validate_rolling_window(30.5)


def test_validate_rolling_window_rejects_too_small_window():
    with pytest.raises(ValueError):
        validate_rolling_window(29)


def test_rolling_historical_var_returns_correct_length():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    result = rolling_historical_var(
        losses,
        confidence_level=0.95,
        window=30,
    )

    assert len(result) == len(losses)


def test_rolling_historical_var_has_nan_before_first_window():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    result = rolling_historical_var(
        losses,
        confidence_level=0.95,
        window=30,
    )

    assert result.iloc[:30].isna().all()
    assert result.iloc[30:].notna().all()


def test_rolling_historical_var_is_finite_after_window():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    result = rolling_historical_var(
        losses,
        confidence_level=0.95,
        window=30,
    )

    assert np.isfinite(result.iloc[30:].to_numpy()).all()


def test_rolling_var_does_not_use_current_observation():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    original = rolling_historical_var(
        losses,
        confidence_level=0.99,
        window=30,
    )

    modified_losses = losses.copy()
    modified_losses.iloc[50] = 1_000_000_000

    modified = rolling_historical_var(
        modified_losses,
        confidence_level=0.99,
        window=30,
    )

    # The VaR estimated for observation 50 must not change,
    # because observation 50 itself is excluded from its window.
    assert original.iloc[50] == modified.iloc[50]


def test_rolling_var_responds_to_new_information_after_current_day():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    original = rolling_historical_var(
        losses,
        confidence_level=0.99,
        window=30,
    )

    modified_losses = losses.copy()
    modified_losses.iloc[50] = 1_000_000_000

    modified = rolling_historical_var(
        modified_losses,
        confidence_level=0.99,
        window=30,
    )

    # The altered observation becomes available to later windows.
    assert modified.iloc[51] != original.iloc[51]


def test_rolling_historical_var_rejects_nan():
    losses = pd.Series(
        [1.0, 2.0, np.nan, 4.0],
        name="Loss",
    )

    with pytest.raises(ValueError):
        rolling_historical_var(
            losses,
            confidence_level=0.95,
            window=30,
        )


def test_rolling_historical_var_rejects_insufficient_data():
    losses = pd.Series(
        np.arange(1, 31, dtype=float),
        name="Loss",
    )

    with pytest.raises(ValueError):
        rolling_historical_var(
            losses,
            confidence_level=0.95,
            window=30,
        )


def test_rolling_historical_var_report_contains_both_confidence_levels():
    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        name="Loss",
    )

    report = rolling_historical_var_report(
        losses,
        confidence_levels=(0.95, 0.99),
        window=30,
    )

    assert "VaR_95" in report.columns
    assert "VaR_99" in report.columns


def test_rolling_historical_var_report_preserves_index():
    index = pd.date_range(
        "2025-01-01",
        periods=100,
        freq="D",
    )

    losses = pd.Series(
        np.arange(1, 101, dtype=float),
        index=index,
        name="Loss",
    )

    report = rolling_historical_var_report(
        losses,
        confidence_levels=(0.95, 0.99),
        window=30,
    )

    pd.testing.assert_index_equal(
        report.index,
        losses.index,
    )