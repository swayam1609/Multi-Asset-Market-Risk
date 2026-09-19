import numpy as np
import pandas as pd
import pytest

from src.risk.kupiec import (
    validate_exceptions,
    kupiec_pof,
    kupiec_pof_report,
)


def test_validate_exceptions_accepts_binary_series():
    exceptions = pd.Series([0, 1, 0, 0, 1])

    result = validate_exceptions(exceptions)

    assert result.dtype == int
    assert result.tolist() == [0, 1, 0, 0, 1]


def test_validate_exceptions_rejects_non_series():
    with pytest.raises(TypeError):
        validate_exceptions([0, 1, 0])


def test_validate_exceptions_rejects_empty_series():
    with pytest.raises(ValueError):
        validate_exceptions(pd.Series(dtype=int))


def test_validate_exceptions_rejects_nan():
    exceptions = pd.Series([0, 1, np.nan])

    with pytest.raises(ValueError):
        validate_exceptions(exceptions)


def test_validate_exceptions_rejects_non_binary_values():
    exceptions = pd.Series([0, 1, 2, 0])

    with pytest.raises(ValueError):
        validate_exceptions(exceptions)


def test_kupiec_pof_returns_expected_fields():
    exceptions = pd.Series([0, 0, 0, 0, 1] * 20)

    result = kupiec_pof(
        exceptions,
        confidence_level=0.95,
    )

    expected_keys = {
        "observations",
        "exceptions",
        "observed_exception_rate",
        "expected_exception_rate",
        "lr_statistic",
        "p_value",
    }

    assert set(result.keys()) == expected_keys


def test_kupiec_pof_exception_rate_is_calculated_correctly():
    exceptions = pd.Series([0, 0, 0, 0, 1] * 20)

    result = kupiec_pof(
        exceptions,
        confidence_level=0.95,
    )

    assert result["observations"] == 100.0
    assert result["exceptions"] == 20.0
    assert result["observed_exception_rate"] == pytest.approx(0.20)
    assert result["expected_exception_rate"] == pytest.approx(0.05)


def test_kupiec_pof_is_finite():
    exceptions = pd.Series([0, 0, 0, 0, 1] * 20)

    result = kupiec_pof(
        exceptions,
        confidence_level=0.95,
    )

    assert np.isfinite(result["lr_statistic"])
    assert np.isfinite(result["p_value"])


def test_kupiec_handles_zero_exceptions():
    exceptions = pd.Series(np.zeros(100, dtype=int))

    result = kupiec_pof(
        exceptions,
        confidence_level=0.95,
    )

    assert result["exceptions"] == 0.0
    assert result["observed_exception_rate"] == 0.0
    assert np.isfinite(result["lr_statistic"])
    assert np.isfinite(result["p_value"])


def test_kupiec_handles_all_exceptions():
    exceptions = pd.Series(np.ones(100, dtype=int))

    result = kupiec_pof(
        exceptions,
        confidence_level=0.95,
    )

    assert result["exceptions"] == 100.0
    assert result["observed_exception_rate"] == 1.0
    assert np.isfinite(result["lr_statistic"])
    assert np.isfinite(result["p_value"])


def test_kupiec_pof_report_contains_both_confidence_levels():
    exceptions = pd.Series([0, 0, 0, 0, 1] * 20)

    report = kupiec_pof_report(
        exceptions,
        confidence_levels=(0.95, 0.99),
    )

    assert list(report.index) == [0.95, 0.99]
    assert "lr_statistic" in report.columns
    assert "p_value" in report.columns


def test_kupiec_pof_report_preserves_confidence_levels():
    exceptions = pd.Series([0, 0, 0, 0, 1] * 20)

    report = kupiec_pof_report(
        exceptions,
        confidence_levels=(0.95, 0.99),
    )

    assert report.loc[0.95, "expected_exception_rate"] == pytest.approx(0.05)
    assert report.loc[0.99, "expected_exception_rate"] == pytest.approx(0.01)