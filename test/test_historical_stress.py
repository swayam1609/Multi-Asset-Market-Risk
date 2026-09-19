import numpy as np
import pandas as pd
import pytest

from src.risk.historical_stress import (
    STRESS_PERIODS,
    validate_stress_periods,
    stress_period_returns,
    cumulative_return,
    maximum_drawdown,
    stress_period_summary,
    historical_stress_report,
)


@pytest.fixture
def portfolio_returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=1000,
    )

    return pd.Series(
        values,
        index=pd.date_range(
            "2007-01-01",
            periods=1000,
            freq="D",
        ),
    )


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(1000, 3),
    )

    return pd.DataFrame(
        values,
        index=pd.date_range(
            "2007-01-01",
            periods=1000,
            freq="D",
        ),
        columns=[
            "SPY",
            "EURUSD=X",
            "TLT",
        ],
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


def test_validate_stress_periods():
    result = validate_stress_periods(
        STRESS_PERIODS
    )

    assert len(result) == 3

    for start, end in result.values():
        assert start <= end


def test_validate_stress_periods_rejects_invalid():
    with pytest.raises(TypeError):
        validate_stress_periods(
            [("2008", "2009")]
        )


def test_stress_period_returns(
    portfolio_returns,
):
    result = stress_period_returns(
        portfolio_returns,
        "2008-01-01",
        "2008-03-01",
    )

    assert len(result) > 0

    assert (
        result.index.min()
        >= pd.Timestamp("2008-01-01")
    )

    assert (
        result.index.max()
        <= pd.Timestamp("2008-03-01")
    )


def test_stress_period_returns_rejects_empty_period(
    portfolio_returns,
):
    with pytest.raises(ValueError):
        stress_period_returns(
            portfolio_returns,
            "2015-01-01",
            "2015-02-01",
        )


def test_cumulative_return():
    returns = pd.Series(
        [0.10, -0.10]
    )

    expected = (
        1.10
        * 0.90
        - 1.0
    )

    assert np.isclose(
        cumulative_return(returns),
        expected,
    )


def test_maximum_drawdown():
    returns = pd.Series(
        [0.10, -0.20, 0.05]
    )

    expected = (
        (1.10 * 0.80 / 1.10)
        - 1.0
    )

    assert np.isclose(
        maximum_drawdown(returns),
        expected,
    )


def test_stress_period_summary(
    portfolio_returns,
):
    result = stress_period_summary(
        portfolio_returns,
        "2008-01-01",
        "2008-03-01",
    )

    assert result["observations"] > 0

    assert np.isfinite(
        result["cumulative_return"]
    )

    assert np.isfinite(
        result["annualized_volatility"]
    )

    assert np.isfinite(
        result["worst_daily_loss"]
    )

    assert np.isfinite(
        result["maximum_drawdown"]
    )


def test_stress_period_summary_worst_loss(
    portfolio_returns,
):
    result = stress_period_summary(
        portfolio_returns,
        "2008-01-01",
        "2008-03-01",
    )

    assert np.isclose(
        result["worst_daily_loss"],
        -result["worst_daily_return"],
    )


def test_historical_stress_report(
    returns,
    weights,
):
    periods = {
        "period_1": (
            "2008-01-01",
            "2008-03-01",
        ),
        "period_2": (
            "2009-01-01",
            "2009-03-01",
        ),
    }

    result = historical_stress_report(
        returns,
        weights,
        portfolio_value=1_000_000.0,
        stress_periods=periods,
    )

    assert list(result.index) == [
        "period_1",
        "period_2",
    ]

    required_columns = [
        "observations",
        "cumulative_return",
        "annualized_volatility",
        "worst_daily_return",
        "worst_daily_loss",
        "maximum_drawdown",
        "cumulative_pnl",
        "worst_daily_loss_dollars",
        "maximum_drawdown_dollars",
    ]

    for column in required_columns:
        assert column in result.columns

    assert (
        result["observations"] > 0
    ).all()

    assert np.isfinite(
        result.to_numpy(dtype=float)
    ).all()


def test_historical_stress_report_uses_portfolio_value(
    returns,
    weights,
):
    periods = {
        "period": (
            "2008-01-01",
            "2008-03-01",
        )
    }

    result_1 = historical_stress_report(
        returns,
        weights,
        portfolio_value=1_000_000.0,
        stress_periods=periods,
    )

    result_2 = historical_stress_report(
        returns,
        weights,
        portfolio_value=2_000_000.0,
        stress_periods=periods,
    )

    assert np.isclose(
        result_2.loc[
            "period",
            "worst_daily_loss_dollars",
        ],
        2
        * result_1.loc[
            "period",
            "worst_daily_loss_dollars",
        ],
    )


def test_historical_stress_report_rejects_invalid_portfolio_value(
    returns,
    weights,
):
    with pytest.raises(ValueError):
        historical_stress_report(
            returns,
            weights,
            portfolio_value=-1.0,
            stress_periods={
                "period": (
                    "2008-01-01",
                    "2008-03-01",
                )
            },
        )


def test_historical_stress_report_rejects_nan(
    returns,
    weights,
):
    modified = returns.copy()

    modified.iloc[10, 0] = np.nan

    with pytest.raises(ValueError):
        historical_stress_report(
            modified,
            weights,
            stress_periods={
                "period": (
                    "2008-01-01",
                    "2008-03-01",
                )
            },
        )


def test_historical_stress_report_default_periods_constant():
    assert len(STRESS_PERIODS) == 3