from __future__ import annotations

import numpy as np
import pandas as pd

from src.analytics.portfolio import (
    portfolio_simple_returns,
    validate_weights,
)


STRESS_PERIODS = {
    "2008_financial_crisis": (
        "2008-01-01",
        "2009-03-31",
    ),
    "2020_covid_crash": (
        "2020-02-01",
        "2020-05-31",
    ),
    "2022_market_shock": (
        "2022-01-01",
        "2022-12-31",
    ),
}


def validate_stress_periods(
    stress_periods: dict[str, tuple[str, str]],
) -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Validate and convert stress-period date ranges.
    """
    if not isinstance(stress_periods, dict):
        raise TypeError(
            "stress_periods must be a dictionary."
        )

    if not stress_periods:
        raise ValueError(
            "stress_periods must not be empty."
        )

    validated = {}

    for name, period in stress_periods.items():

        if not isinstance(name, str):
            raise TypeError(
                "Stress-period names must be strings."
            )

        if (
            not isinstance(period, tuple)
            or len(period) != 2
        ):
            raise ValueError(
                "Each stress period must contain "
                "a start and end date."
            )

        start = pd.Timestamp(period[0])
        end = pd.Timestamp(period[1])

        if start > end:
            raise ValueError(
                "Stress-period start date must not "
                "be after the end date."
            )

        validated[name] = (
            start,
            end,
        )

    return validated


def stress_period_returns(
    portfolio_returns: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
) -> pd.Series:
    """
    Extract portfolio returns for a historical stress period.
    """
    if not isinstance(
        portfolio_returns,
        pd.Series,
    ):
        raise TypeError(
            "portfolio_returns must be a pandas Series."
        )

    if portfolio_returns.empty:
        raise ValueError(
            "portfolio_returns must not be empty."
        )

    if portfolio_returns.isna().any():
        raise ValueError(
            "portfolio_returns must not contain NaN values."
        )

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)

    if start > end:
        raise ValueError(
            "start_date must not be after end_date."
        )

    result = portfolio_returns.loc[
        (portfolio_returns.index >= start)
        & (portfolio_returns.index <= end)
    ]

    if result.empty:
        raise ValueError(
            "No portfolio returns found "
            "inside the requested stress period."
        )

    return result


def cumulative_return(
    returns: pd.Series,
) -> float:
    """
    Calculate cumulative simple return.
    """
    if not isinstance(returns, pd.Series):
        raise TypeError(
            "returns must be a pandas Series."
        )

    return float(
        (1.0 + returns).prod() - 1.0
    )


def maximum_drawdown(
    returns: pd.Series,
) -> float:
    """
    Calculate maximum drawdown from simple returns.
    """
    wealth = (
        1.0 + returns
    ).cumprod()

    running_peak = wealth.cummax()

    drawdown = (
        wealth / running_peak
    ) - 1.0

    return float(
        drawdown.min()
    )


def stress_period_summary(
    portfolio_returns: pd.Series,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    annualization_factor: int = 252,
) -> dict[str, float | int]:
    """
    Calculate summary statistics for one stress period.
    """
    returns = stress_period_returns(
        portfolio_returns,
        start_date,
        end_date,
    )

    if not isinstance(
        annualization_factor,
        (int, np.integer),
    ):
        raise TypeError(
            "annualization_factor must be an integer."
        )

    if annualization_factor <= 0:
        raise ValueError(
            "annualization_factor must be positive."
        )

    volatility = float(
        returns.std(ddof=1)
        * np.sqrt(annualization_factor)
    )

    worst_daily_return = float(
        returns.min()
    )

    worst_daily_loss = -worst_daily_return

    return {
        "observations": int(len(returns)),
        "cumulative_return": cumulative_return(
            returns
        ),
        "annualized_volatility": volatility,
        "worst_daily_return": worst_daily_return,
        "worst_daily_loss": worst_daily_loss,
        "maximum_drawdown": maximum_drawdown(
            returns
        ),
    }


def historical_stress_report(
    returns: pd.DataFrame,
    weights: pd.Series,
    portfolio_value: float = 1_000_000.0,
    stress_periods: dict[str, tuple[str, str]]
    | None = None,
) -> pd.DataFrame:
    """
    Generate a historical stress-testing report.

    Portfolio returns are calculated using simple returns
    derived from the supplied log-return DataFrame.

    Dollar losses are based on the fixed portfolio value.
    """
    if not isinstance(
        returns,
        pd.DataFrame,
    ):
        raise TypeError(
            "returns must be a pandas DataFrame."
        )

    if returns.empty:
        raise ValueError(
            "returns must not be empty."
        )

    if returns.isna().any().any():
        raise ValueError(
            "returns must not contain NaN values."
        )

    if not np.isfinite(
        returns.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "returns must contain only finite values."
        )

    if not np.isfinite(
        portfolio_value
    ):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be positive."
        )

    weights = validate_weights(
        weights,
        returns.columns,
    )

    if stress_periods is None:
        stress_periods = STRESS_PERIODS

    validated_periods = validate_stress_periods(
        stress_periods
    )

    portfolio_returns = (
        portfolio_simple_returns(
            returns,
            weights,
        )
    )

    rows = []

    for name, (
        start_date,
        end_date,
    ) in validated_periods.items():

        summary = stress_period_summary(
            portfolio_returns,
            start_date,
            end_date,
        )

        summary["period"] = name

        summary[
            "cumulative_pnl"
        ] = (
            portfolio_value
            * summary["cumulative_return"]
        )

        summary[
            "worst_daily_loss_dollars"
        ] = (
            portfolio_value
            * summary["worst_daily_loss"]
        )

        summary[
            "maximum_drawdown_dollars"
        ] = (
            portfolio_value
            * summary["maximum_drawdown"]
        )

        rows.append(summary)

    report = pd.DataFrame(
        rows
    )

    return report.set_index(
        "period"
    )