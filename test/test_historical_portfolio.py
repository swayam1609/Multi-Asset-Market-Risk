"""
Tests for Step 10C historical portfolio VaR and
Expected Shortfall.
"""

import numpy as np
import pandas as pd

from src.risk.portfolio import (
    portfolio_losses,
)

from src.risk.historical_portfolio import (
    historical_portfolio_var,
    historical_portfolio_expected_shortfall,
    historical_portfolio_var_es,
    historical_portfolio_report,
)


def make_log_returns():

    return pd.DataFrame(
        {
            "SPY": [
                0.02,
                0.01,
                0.005,
                0.0,
                -0.005,
                -0.01,
                -0.02,
                -0.03,
                -0.04,
                -0.05,
            ],
            "EURUSD=X": [
                0.01,
                0.005,
                0.002,
                0.0,
                -0.002,
                -0.005,
                -0.01,
                -0.015,
                -0.02,
                -0.025,
            ],
            "TLT": [
                0.005,
                0.003,
                0.002,
                0.0,
                -0.003,
                -0.005,
                -0.008,
                -0.01,
                -0.015,
                -0.02,
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


def test_historical_portfolio_var_is_positive():

    var = historical_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert var > 0
    assert np.isfinite(var)


def test_historical_portfolio_es_is_positive():

    es = historical_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert es > 0
    assert np.isfinite(es)


def test_historical_portfolio_es_at_least_var():

    var = historical_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    es = historical_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    assert es >= var


def test_historical_portfolio_var_es_returns_both():

    result = historical_portfolio_var_es(
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

    var_95 = historical_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    var_99 = historical_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.99,
    )

    assert var_99 >= var_95


def test_99_percent_es_not_less_than_95_percent_es():

    es_95 = historical_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
    )

    es_99 = historical_portfolio_expected_shortfall(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.99,
    )

    assert es_99 >= es_95


def test_report_contains_95_and_99_percent():

    report = historical_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert len(report) == 2

    assert set(
        report["Confidence_Level"]
    ) == {0.95, 0.99}


def test_report_contains_required_columns():

    report = historical_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    required_columns = {
        "Confidence_Level",
        "VaR",
        "Expected_Shortfall",
        "Observations",
    }

    assert required_columns.issubset(
        report.columns
    )


def test_report_values_are_finite():

    report = historical_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert np.isfinite(
        report[
            [
                "VaR",
                "Expected_Shortfall",
            ]
        ].to_numpy()
    ).all()


def test_portfolio_loss_series_is_consistent():

    losses = portfolio_losses(
        make_log_returns(),
        make_weights(),
        1_000_000,
    )

    assert len(losses) == len(
        make_log_returns()
    )

    assert np.isfinite(
        losses.to_numpy()
    ).all()