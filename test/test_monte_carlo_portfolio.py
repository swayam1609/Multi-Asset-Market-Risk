"""
Tests for Step 10E Monte Carlo portfolio risk.
"""

import numpy as np
import pandas as pd
import pytest

from src.risk.monte_carlo import (
    simulate_multivariate_normal_returns,
    simulated_portfolio_returns,
    simulated_losses,
)

from src.risk.monte_carlo_portfolio import (
    monte_carlo_portfolio_losses,
    monte_carlo_portfolio_var,
    monte_carlo_portfolio_expected_shortfall,
    monte_carlo_portfolio_var_es,
    monte_carlo_portfolio_report,
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
                0.015,
                -0.012,
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
                0.008,
                -0.006,
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
                0.004,
                -0.007,
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


def test_simulation_has_expected_shape():

    simple_returns = (
        np.exp(
            make_log_returns()
        ) - 1.0
    )

    mean_returns = (
        simple_returns.mean()
    )

    covariance = (
        simple_returns.cov()
    )

    simulated = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations=1_000,
            random_seed=42,
        )
    )

    assert simulated.shape == (
        1_000,
        3,
    )

    assert list(
        simulated.columns
    ) == [
        "SPY",
        "EURUSD=X",
        "TLT",
    ]


def test_simulation_is_reproducible():

    simple_returns = (
        np.exp(
            make_log_returns()
        ) - 1.0
    )

    mean_returns = (
        simple_returns.mean()
    )

    covariance = (
        simple_returns.cov()
    )

    first = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations=1_000,
            random_seed=42,
        )
    )

    second = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations=1_000,
            random_seed=42,
        )
    )

    pd.testing.assert_frame_equal(
        first,
        second,
    )


def test_different_seed_changes_simulation():

    simple_returns = (
        np.exp(
            make_log_returns()
        ) - 1.0
    )

    mean_returns = (
        simple_returns.mean()
    )

    covariance = (
        simple_returns.cov()
    )

    first = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations=500,
            random_seed=42,
        )
    )

    second = (
        simulate_multivariate_normal_returns(
            mean_returns,
            covariance,
            n_simulations=500,
            random_seed=123,
        )
    )

    assert not np.array_equal(
        first.to_numpy(),
        second.to_numpy(),
    )


def test_simulated_portfolio_returns_are_finite():

    simple_returns = (
        np.exp(
            make_log_returns()
        ) - 1.0
    )

    simulated = pd.DataFrame(
        np.repeat(
            simple_returns.mean()
            .to_numpy()
            .reshape(1, -1),
            100,
            axis=0,
        ),
        columns=simple_returns.columns,
    )

    portfolio = (
        simulated_portfolio_returns(
            simulated,
            make_weights(),
        )
    )

    assert len(portfolio) == 100

    assert np.isfinite(
        portfolio.to_numpy()
    ).all()


def test_simulated_losses_follow_loss_convention():

    simulated_returns = pd.Series(
        [0.01, -0.02],
        name="Portfolio",
    )

    losses = simulated_losses(
        simulated_returns,
        1_000_000,
    )

    expected = pd.Series(
    [-10_000.0, 20_000.0],
    name="Loss",
)

    pd.testing.assert_series_equal(
        losses,
        expected,
    )


def test_monte_carlo_losses_have_requested_count():

    losses = monte_carlo_portfolio_losses(
        make_log_returns(),
        make_weights(),
        1_000_000,
        n_simulations=2_000,
        random_seed=42,
    )

    assert len(losses) == 2_000

    assert np.isfinite(
        losses.to_numpy()
    ).all()


def test_monte_carlo_var_is_positive():

    var = monte_carlo_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
        n_simulations=5_000,
        random_seed=42,
    )

    assert var > 0
    assert np.isfinite(var)


def test_monte_carlo_es_is_positive():

    es = (
        monte_carlo_portfolio_expected_shortfall(
            make_log_returns(),
            make_weights(),
            1_000_000,
            confidence_level=0.95,
            n_simulations=5_000,
            random_seed=42,
        )
    )

    assert es > 0
    assert np.isfinite(es)


def test_monte_carlo_es_at_least_var():

    result = (
        monte_carlo_portfolio_var_es(
            make_log_returns(),
            make_weights(),
            1_000_000,
            confidence_level=0.95,
            n_simulations=5_000,
            random_seed=42,
        )
    )

    assert (
        result["Expected_Shortfall"]
        >= result["VaR"]
    )


def test_99_percent_var_not_less_than_95_percent_var():

    var_95 = monte_carlo_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.95,
        n_simulations=5_000,
        random_seed=42,
    )

    var_99 = monte_carlo_portfolio_var(
        make_log_returns(),
        make_weights(),
        1_000_000,
        confidence_level=0.99,
        n_simulations=5_000,
        random_seed=42,
    )

    assert var_99 >= var_95


def test_report_contains_both_confidence_levels():

    report = monte_carlo_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
        n_simulations=5_000,
        random_seed=42,
    )

    assert len(report) == 2

    assert set(
        report["Confidence_Level"]
    ) == {0.95, 0.99}


def test_report_records_simulation_settings():

    report = monte_carlo_portfolio_report(
        make_log_returns(),
        make_weights(),
        1_000_000,
        n_simulations=5_000,
        random_seed=42,
    )

    assert (
        report["Simulations"] == 5_000
    ).all()

    assert (
        report["Random_Seed"] == 42
    ).all()


def test_zero_simulations_are_rejected():

    simple_returns = (
        np.exp(
            make_log_returns()
        ) - 1.0
    )

    with pytest.raises(ValueError):

        simulate_multivariate_normal_returns(
            simple_returns.mean(),
            simple_returns.cov(),
            n_simulations=0,
            random_seed=42,
        )