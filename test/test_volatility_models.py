import numpy as np
import pandas as pd

from src.analytics.volatility import (
    daily_volatility,
    annualized_volatility,
    rolling_volatility,
    ewma_variance,
    ewma_volatility,
    fit_garch,
    garch_parameter_summary,
    garch_one_step_forecast,
    rolling_garch_forecast,
)


def make_test_returns(n=1000, seed=42):
    rng = np.random.default_rng(seed)

    values = rng.normal(
        loc=0.0,
        scale=0.01,
        size=n,
    )

    index = pd.date_range(
        "2010-01-01",
        periods=n,
        freq="B",
    )

    return pd.Series(
        values,
        index=index,
        name="TEST",
    )


def test_daily_volatility():
    returns = pd.DataFrame(
        {
            "A": make_test_returns(),
            "B": make_test_returns(seed=7),
        }
    )

    result = daily_volatility(returns)

    assert len(result) == 2
    assert np.isfinite(result).all()
    assert (result > 0).all()


def test_annualized_volatility():
    returns = pd.DataFrame(
        {
            "A": make_test_returns(),
        }
    )

    daily = daily_volatility(returns)
    annualized = annualized_volatility(returns)

    expected = daily["A"] * np.sqrt(252)

    assert np.isclose(
        annualized["A"],
        expected,
    )


def test_rolling_volatility():
    returns = pd.DataFrame(
        {
            "A": make_test_returns(),
        }
    )

    result = rolling_volatility(
        returns,
        window=63,
    )

    assert len(result) == len(returns)
    assert result.iloc[:62].isna().all().all()
    assert result.iloc[62:].notna().all().all()


def test_ewma_variance():
    returns = make_test_returns(n=100)

    result = ewma_variance(
        returns,
        lam=0.94,
    )

    assert len(result) == len(returns)
    assert result.notna().all()
    assert (result >= 0).all()


def test_ewma_volatility():
    returns = make_test_returns(n=100)

    result = ewma_volatility(
        returns,
        lam=0.94,
    )

    assert len(result) == len(returns)
    assert result.notna().all()
    assert (result >= 0).all()


def test_garch_fit():
    returns = make_test_returns(n=1000)

    fitted_model, scaled_returns = fit_garch(
        returns
    )

    assert len(scaled_returns) == len(returns)

    summary = garch_parameter_summary(
        fitted_model
    )

    assert np.isfinite(
        summary["omega_scaled"]
    )

    assert np.isfinite(
        summary["omega_decimal"]
    )

    assert 0 <= summary["alpha"] <= 1
    assert 0 <= summary["beta"] <= 1

    assert summary["persistence"] >= 0


def test_garch_one_step_forecast():
    returns = make_test_returns(n=1000)

    forecast = garch_one_step_forecast(
        returns
    )

    assert np.isfinite(forecast)
    assert forecast > 0


def test_garch_walk_forward_forecast():
    returns = make_test_returns(n=1000)

    forecast = rolling_garch_forecast(
        returns,
        refit_frequency=50,
        min_train_size=500,
    )

    assert len(forecast) == len(returns)

    assert forecast.iloc[:500].isna().all()

    assert forecast.iloc[500:].notna().all()

    assert np.isfinite(
        forecast.iloc[500:]
    ).all()

    assert (
        forecast.iloc[500:] > 0
    ).all()


def test_ewma_no_look_ahead():
    returns = make_test_returns(n=300)

    original = ewma_volatility(returns)

    modified = returns.copy()

    # Change observation 200.
    modified.iloc[200] *= 5

    changed = ewma_volatility(modified)

    # Volatility at 200 must NOT use r_200.
    assert np.isclose(
        original.iloc[200],
        changed.iloc[200],
    )

    # Volatility at 201 SHOULD react to r_200.
    assert not np.isclose(
        original.iloc[201],
        changed.iloc[201],
    )


def test_garch_forecast_no_look_ahead():
    returns = make_test_returns(n=850)

    original = rolling_garch_forecast(
        returns,
        refit_frequency=50,
        min_train_size=500,
    )

    modified = returns.copy()

    # Change observation 600.
    modified.iloc[600] *= 5

    changed = rolling_garch_forecast(
        modified,
        refit_frequency=50,
        min_train_size=500,
    )

    # Forecast at 600 must NOT use r_600.
    assert np.isclose(
        original.iloc[600],
        changed.iloc[600],
    )

    # Forecast at 601 SHOULD react to r_600.
    assert not np.isclose(
        original.iloc[601],
        changed.iloc[601],
    )