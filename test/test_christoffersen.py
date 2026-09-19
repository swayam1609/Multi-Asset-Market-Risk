import numpy as np
import pandas as pd
import pytest

from src.risk.christoffersen import (
    transition_counts,
    christoffersen_independence,
    christoffersen_conditional_coverage,
    christoffersen_report,
)


def test_transition_counts_are_correct():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 1, 0]
    )

    result = transition_counts(exceptions)

    assert result["n00"] == 1
    assert result["n01"] == 2
    assert result["n10"] == 2
    assert result["n11"] == 1


def test_transition_counts_rejects_invalid_series():
    with pytest.raises(TypeError):
        transition_counts([0, 1, 0])


def test_christoffersen_independence_returns_expected_fields():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 1, 0]
    )

    result = christoffersen_independence(
        exceptions
    )

    expected_keys = {
        "n00",
        "n01",
        "n10",
        "n11",
        "pi",
        "pi01",
        "pi11",
        "lr_independence",
        "p_value_independence",
    }

    assert set(result.keys()) == expected_keys


def test_christoffersen_independence_is_finite():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 1, 0]
    )

    result = christoffersen_independence(
        exceptions
    )

    assert np.isfinite(
        result["lr_independence"]
    )

    assert np.isfinite(
        result["p_value_independence"]
    )


def test_independence_handles_zero_exceptions():
    exceptions = pd.Series(
        np.zeros(100, dtype=int)
    )

    result = christoffersen_independence(
        exceptions
    )

    assert result["n01"] == 0.0
    assert result["n11"] == 0.0
    assert np.isfinite(
        result["lr_independence"]
    )
    assert np.isfinite(
        result["p_value_independence"]
    )


def test_independence_handles_all_exceptions():
    exceptions = pd.Series(
        np.ones(100, dtype=int)
    )

    result = christoffersen_independence(
        exceptions
    )

    assert result["n10"] == 0.0
    assert result["n00"] == 0.0
    assert np.isfinite(
        result["lr_independence"]
    )
    assert np.isfinite(
        result["p_value_independence"]
    )


def test_conditional_coverage_contains_both_components():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 0, 0, 1] * 20
    )

    result = christoffersen_conditional_coverage(
        exceptions,
        confidence_level=0.95,
    )

    assert "lr_independence" in result
    assert "lr_coverage" in result
    assert "p_value_independence" in result
    assert "p_value_coverage" in result


def test_conditional_coverage_is_finite():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 0, 0, 1] * 20
    )

    result = christoffersen_conditional_coverage(
        exceptions,
        confidence_level=0.95,
    )

    assert np.isfinite(
        result["lr_coverage"]
    )

    assert np.isfinite(
        result["p_value_coverage"]
    )


def test_conditional_coverage_handles_zero_exceptions():
    exceptions = pd.Series(
        np.zeros(100, dtype=int)
    )

    result = christoffersen_conditional_coverage(
        exceptions,
        confidence_level=0.95,
    )

    assert np.isfinite(
        result["lr_coverage"]
    )

    assert np.isfinite(
        result["p_value_coverage"]
    )


def test_conditional_coverage_rejects_single_observation():
    exceptions = pd.Series([0])

    with pytest.raises(ValueError):
        christoffersen_conditional_coverage(
            exceptions,
            confidence_level=0.95,
        )


def test_christoffersen_report_contains_both_levels():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 0, 0, 1] * 20
    )

    report = christoffersen_report(
        exceptions,
        confidence_levels=(0.95, 0.99),
    )

    assert list(report.index) == [
        0.95,
        0.99,
    ]

    assert "lr_independence" in report.columns
    assert "lr_coverage" in report.columns
    assert "p_value_coverage" in report.columns


def test_christoffersen_report_preserves_expected_levels():
    exceptions = pd.Series(
        [0, 0, 1, 0, 1, 0, 0, 1] * 20
    )

    report = christoffersen_report(
        exceptions,
        confidence_levels=(0.95, 0.99),
    )

    assert report.loc[
        0.95,
        "lr_coverage",
    ] >= 0.0

    assert report.loc[
        0.99,
        "lr_coverage",
    ] >= 0.0