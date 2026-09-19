import numpy as np
import pandas as pd
import pytest

from src.risk.hypothetical_stress import (
    DEFAULT_HYPOTHETICAL_SCENARIOS,
    validate_scenarios,
    hypothetical_stress_return,
    hypothetical_stress_loss,
    hypothetical_stress_report,
)


@pytest.fixture
def weights():
    return pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.30,
        }
    )


@pytest.fixture
def scenarios():
    return {
        "scenario_1": {
            "SPY": -0.10,
            "EURUSD=X": -0.05,
            "TLT": -0.08,
        },
        "scenario_2": {
            "SPY": -0.15,
            "EURUSD=X": -0.10,
            "TLT": -0.10,
        },
    }


def test_validate_scenarios(
    weights,
    scenarios,
):
    result = validate_scenarios(
        scenarios,
        weights.index,
    )

    assert set(result.keys()) == {
        "scenario_1",
        "scenario_2",
    }


def test_validate_scenarios_rejects_missing_asset(
    weights,
):
    invalid = {
        "scenario": {
            "SPY": -0.10,
            "EURUSD=X": -0.05,
        }
    }

    with pytest.raises(ValueError):
        validate_scenarios(
            invalid,
            weights.index,
        )


def test_validate_scenarios_rejects_non_numeric_shock(
    weights,
):
    invalid = {
        "scenario": {
            "SPY": "bad",
            "EURUSD=X": -0.05,
            "TLT": -0.08,
        }
    }

    with pytest.raises(TypeError):
        validate_scenarios(
            invalid,
            weights.index,
        )


def test_validate_scenarios_rejects_negative_100_percent(
    weights,
):
    invalid = {
        "scenario": {
            "SPY": -1.0,
            "EURUSD=X": -0.05,
            "TLT": -0.08,
        }
    }

    with pytest.raises(ValueError):
        validate_scenarios(
            invalid,
            weights.index,
        )


def test_hypothetical_stress_return(
    weights,
):
    scenario = {
        "SPY": -0.10,
        "EURUSD=X": -0.05,
        "TLT": -0.08,
    }

    result = hypothetical_stress_return(
        scenario,
        weights,
    )

    expected = (
        0.50 * -0.10
        + 0.20 * -0.05
        + 0.30 * -0.08
    )

    assert np.isclose(
        result,
        expected,
    )

    assert np.isclose(
        result,
        -0.084,
    )


def test_hypothetical_stress_loss(
    weights,
):
    scenario = {
        "SPY": -0.10,
        "EURUSD=X": -0.05,
        "TLT": -0.08,
    }

    result = hypothetical_stress_loss(
        scenario,
        weights,
        portfolio_value=1_000_000.0,
    )

    assert np.isclose(
        result,
        84_000.0,
    )


def test_hypothetical_stress_loss_scales_with_portfolio_value(
    weights,
):
    scenario = {
        "SPY": -0.10,
        "EURUSD=X": -0.05,
        "TLT": -0.08,
    }

    loss_1 = hypothetical_stress_loss(
        scenario,
        weights,
        1_000_000.0,
    )

    loss_2 = hypothetical_stress_loss(
        scenario,
        weights,
        2_000_000.0,
    )

    assert np.isclose(
        loss_2,
        2 * loss_1,
    )


def test_hypothetical_stress_report(
    weights,
    scenarios,
):
    result = hypothetical_stress_report(
        weights,
        scenarios,
        portfolio_value=1_000_000.0,
    )

    assert list(result.index) == [
        "scenario_1",
        "scenario_2",
    ]

    assert (
        "portfolio_return"
        in result.columns
    )

    assert (
        "portfolio_loss"
        in result.columns
    )

    assert (
        "portfolio_value"
        in result.columns
    )

    assert np.isfinite(
        result.to_numpy(dtype=float)
    ).all()


def test_hypothetical_stress_report_loss_sign(
    weights,
    scenarios,
):
    result = hypothetical_stress_report(
        weights,
        scenarios,
    )

    assert (
        result["portfolio_loss"] > 0
    ).all()

    assert np.allclose(
        result["portfolio_loss"],
        -1_000_000.0
        * result["portfolio_return"],
    )


def test_hypothetical_stress_report_custom_scenario(
    weights,
):
    custom = {
        "custom": {
            "SPY": -0.05,
            "EURUSD=X": 0.02,
            "TLT": -0.03,
        }
    }

    result = hypothetical_stress_report(
        weights,
        custom,
    )

    expected_return = (
        0.50 * -0.05
        + 0.20 * 0.02
        + 0.30 * -0.03
    )

    assert np.isclose(
        result.loc[
            "custom",
            "portfolio_return",
        ],
        expected_return,
    )


def test_default_scenarios():
    assert set(
        DEFAULT_HYPOTHETICAL_SCENARIOS.keys()
    ) == {
        "moderate_shock",
        "severe_shock",
    }


def test_hypothetical_stress_report_rejects_invalid_weights():
    weights = pd.Series(
        {
            "SPY": 0.50,
            "EURUSD=X": 0.20,
            "TLT": 0.20,
        }
    )

    with pytest.raises(ValueError):
        hypothetical_stress_report(
            weights,
        )


def test_hypothetical_stress_report_rejects_invalid_portfolio_value(
    weights,
):
    with pytest.raises(ValueError):
        hypothetical_stress_report(
            weights,
            portfolio_value=-1.0,
        )