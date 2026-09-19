"""
Tests for Step 10D Parametric portfolio VaR and
Expected Shortfall.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.parametric_portfolio import (
    parametric_portfolio_var,
    parametric_portfolio_expected_shortfall,
    parametric_portfolio_var_es,
    parametric_portfolio_report,
)


def make_log_returns():

    return pd.DataFrame(
        {
            "SPY": [
                0.020,
                0.010,
                0.005,
                0.000,
                -0.005,
                -0.010,
                -0.020,
                -0.030,
                -0.040,
                -0.050,
            ],
            "EURUSD=X": [
                0.010,
                0.005,
                0.002,
                0.000,
                -0.002,
                -0.005,
                -0.010,
                -0.015,
                -0.020,
                -0.025,
            ],
            "TLT": [
                0.005,
                0.003,
                0.002,
                0.000,
                -0.003,
                -0.005,
                -0.008,
                -0.010,
                -0.015,
                -0.020,
            ],
        }
    )


def make_weights():

    return pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )


def test_parametric_portfolio_var_is_positive():

    var = parametric_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert var > 0
    assert np.isfinite(var)


def test_parametric_portfolio_es_is_positive():

    es = parametric_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert es > 0
    assert np.isfinite(es)


def test_parametric_portfolio_es_at_least_var():

    var = parametric_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    es = parametric_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert es >= var


def test_parametric_portfolio_var_es_returns_both():

    result = parametric_portfolio_var_es(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert "VaR" in result
    assert "Expected_Shortfall" in result

    assert result["VaR"] > 0
    assert result["Expected_Shortfall"] > 0

    assert np.isfinite(
        result["VaR"]
    )

    assert np.isfinite(
        result["Expected_Shortfall"]
    )


def test_99_percent_var_not_less_than_95_percent_var():

    var_95 = parametric_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    var_99 = parametric_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.99,
    )

    assert var_99 >= var_95


def test_99_percent_es_not_less_than_95_percent_es():

    es_95 = parametric_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    es_99 = parametric_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.99,
    )

    assert es_99 >= es_95


def test_parametric_report_has_two_confidence_levels():

    report = parametric_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert len(report) == 2

    assert set(
        report["Confidence_Level"]
    ) == {0.95, 0.99}


def test_parametric_report_contains_required_columns():

    report = parametric_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    required_columns = {
        "Confidence_Level",
        "VaR",
        "Expected_Shortfall",
        "Mean_Return",
        "Volatility",
        "Observations",
    }

    assert required_columns.issubset(
        report.columns
    )


def test_parametric_report_values_are_finite():

    report = parametric_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert np.isfinite(
        report[
            [
                "VaR",
                "Expected_Shortfall",
                "Mean_Return",
                "Volatility",
            ]
        ].to_numpy()
    ).all()


def test_zero_volatility_is_rejected():

    log_returns = pd.DataFrame(
        {
            "SPY": [0.01] * 10,
            "EURUSD=X": [0.01] * 10,
            "TLT": [0.01] * 10,
        }
    )

    with pytest.raises(ValueError):

        parametric_portfolio_var(
            log_returns,
            make_weights(),
            1_000_000,
            confidence_level=0.95,
        )


def test_invalid_portfolio_value_is_rejected():

    with pytest.raises(ValueError):

        parametric_portfolio_var(
            make_log_returns(),
            make_weights(),
            0,
            confidence_level=0.95,
        )