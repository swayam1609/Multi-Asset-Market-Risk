from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd


# =====================================================================
# Constants
# =====================================================================

DEFAULT_PORTFOLIO_VALUE = 1_000_000.0

DEFAULT_HYPOTHETICAL_SCENARIOS: dict[str, dict[str, float]] = {
    "moderate_shock": {
        "SPY": -0.10,
        "EURUSD=X": -0.05,
        "TLT": -0.08,
    },
    "severe_shock": {
        "SPY": -0.15,
        "EURUSD=X": -0.10,
        "TLT": -0.10,
    },
}


# =====================================================================
# Validation
# =====================================================================

def validate_portfolio_value(
    portfolio_value: float,
) -> float:
    """
    Validate portfolio market value.

    Portfolio value must be a positive finite number.
    """

    if not isinstance(
        portfolio_value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(
            "portfolio_value must be a numeric value."
        )

    portfolio_value = float(portfolio_value)

    if not np.isfinite(portfolio_value):
        raise ValueError(
            "portfolio_value must be finite."
        )

    if portfolio_value <= 0:
        raise ValueError(
            "portfolio_value must be greater than zero."
        )

    return portfolio_value


def validate_weights(
    weights: pd.Series,
) -> pd.Series:
    """
    Validate portfolio weights.

    Requirements:
    - pandas Series
    - non-empty
    - numeric
    - finite
    - no NaN values
    - weights sum to 1
    """

    if not isinstance(weights, pd.Series):
        raise TypeError(
            "weights must be a pandas Series."
        )

    if weights.empty:
        raise ValueError(
            "weights must not be empty."
        )

    try:
        weights = weights.astype(float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "weights must contain numeric values."
        ) from exc

    if weights.isna().any():
        raise ValueError(
            "weights must not contain NaN values."
        )

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError(
            "weights must contain only finite values."
        )

    if not np.isclose(
        values.sum(),
        1.0,
        rtol=0.0,
        atol=1e-10,
    ):
        raise ValueError(
            "weights must sum to 1."
        )

    return weights


def validate_scenario(
    scenario: Mapping[str, float] | pd.Series,
    assets: pd.Index,
) -> pd.Series:
    """
    Validate a single hypothetical stress scenario.

    Scenario shocks must:
    - contain exactly the portfolio assets
    - be numeric
    - be finite
    - be greater than -100%

    Both dictionaries/mappings and pandas Series are accepted.
    """

    if not isinstance(
        scenario,
        (Mapping, pd.Series),
    ):
        raise TypeError(
            "scenario must be a mapping or pandas Series."
        )

    assets = pd.Index(assets)

    if assets.empty:
        raise ValueError(
            "assets must not be empty."
        )

    scenario_assets = pd.Index(
        scenario.keys()
    )

    missing_assets = assets.difference(
        scenario_assets
    )

    extra_assets = scenario_assets.difference(
        assets
    )

    if len(missing_assets) > 0:
        raise ValueError(
            f"scenario is missing assets: "
            f"{list(missing_assets)}."
        )

    if len(extra_assets) > 0:
        raise ValueError(
            f"scenario contains unknown assets: "
            f"{list(extra_assets)}."
        )

    shocks = []

    for asset in assets:
        shock = scenario[asset]

        if not isinstance(
            shock,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError(
                f"shock for '{asset}' must be numeric."
            )

        shock = float(shock)

        if not np.isfinite(shock):
            raise ValueError(
                f"shock for '{asset}' must be finite."
            )

        if shock <= -1.0:
            raise ValueError(
                f"shock for '{asset}' must be greater than -100%."
            )

        shocks.append(shock)

    return pd.Series(
        shocks,
        index=assets,
        dtype=float,
    )


def validate_scenarios(
    scenarios: Mapping[str, Mapping[str, float]],
    assets: pd.Index,
) -> dict[str, pd.Series]:
    """
    Validate and normalize multiple hypothetical scenarios.
    """

    if not isinstance(scenarios, Mapping):
        raise TypeError(
            "scenarios must be a mapping."
        )

    if len(scenarios) == 0:
        raise ValueError(
            "scenarios must contain at least one scenario."
        )

    validated = {}

    for scenario_name, scenario in scenarios.items():

        if not isinstance(scenario_name, str):
            raise TypeError(
                "scenario names must be strings."
            )

        if not scenario_name.strip():
            raise ValueError(
                "scenario names must not be empty."
            )

        validated[scenario_name] = validate_scenario(
            scenario=scenario,
            assets=assets,
        )

    return validated


# =====================================================================
# Core hypothetical stress return
# =====================================================================

def hypothetical_stress_return(
    scenario: Mapping[str, float],
    weights: pd.Series,
) -> float:
    """
    Calculate portfolio return under a hypothetical stress scenario.

    Parameters
    ----------
    scenario:
        Asset-level hypothetical returns.

    weights:
        Portfolio weights.

    Returns
    -------
    float
        Portfolio stress return.

    Formula
    -------
    R_p = sum(w_i * shock_i)
    """

    weights = validate_weights(
        weights
    )

    shocks = validate_scenario(
        scenario=scenario,
        assets=weights.index,
    )

    portfolio_return = float(
        np.dot(
            weights.to_numpy(dtype=float),
            shocks.to_numpy(dtype=float),
        )
    )

    if not np.isfinite(portfolio_return):
        raise ValueError(
            "calculated portfolio stress return "
            "must be finite."
        )

    return portfolio_return


# =====================================================================
# Core hypothetical stress loss
# =====================================================================

def hypothetical_stress_loss(
    scenario: Mapping[str, float],
    weights: pd.Series,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> float:
    """
    Calculate dollar portfolio loss under a hypothetical scenario.

    Parameters
    ----------
    scenario:
        Asset-level hypothetical returns.

    weights:
        Portfolio weights.

    portfolio_value:
        Initial portfolio market value.

    Returns
    -------
    float
        Dollar loss.

    Formula
    -------
    L = -V_0 * R_p

    Positive values represent losses.
    Negative values represent gains.
    """

    portfolio_value = validate_portfolio_value(
        portfolio_value
    )

    portfolio_return = hypothetical_stress_return(
        scenario=scenario,
        weights=weights,
    )

    loss = -portfolio_value * portfolio_return

    if not np.isfinite(loss):
        raise ValueError(
            "calculated portfolio stress loss "
            "must be finite."
        )

    return float(loss)


# =====================================================================
# Scenario-level report
# =====================================================================

def hypothetical_stress_report(
    weights: pd.Series,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> pd.DataFrame:
    """
    Generate a hypothetical stress-testing report.

    Parameters
    ----------
    weights:
        Portfolio weights.

    scenarios:
        Mapping of scenario names to asset shocks.

        If None, the default moderate and severe scenarios
        are used.

    portfolio_value:
        Initial portfolio market value.

    Returns
    -------
    pandas.DataFrame
        Scenario-level stress results.

    Columns
    -------
    scenario
    portfolio_return
    portfolio_loss
    portfolio_value_before
    portfolio_value_after
    loss_percentage
    """

    # Validate these FIRST.
    # This is important because an invalid portfolio should fail
    # before scenario calculations are performed.
    weights = validate_weights(
        weights
    )

    portfolio_value = validate_portfolio_value(
        portfolio_value
    )

    if scenarios is None:
        scenarios = DEFAULT_HYPOTHETICAL_SCENARIOS

    validated_scenarios = validate_scenarios(
        scenarios=scenarios,
        assets=weights.index,
    )

    rows = []

    for scenario_name, shocks in validated_scenarios.items():

        portfolio_return = hypothetical_stress_return(
            scenario=shocks,
            weights=weights,
        )

        portfolio_loss = hypothetical_stress_loss(
            scenario=shocks,
            weights=weights,
            portfolio_value=portfolio_value,
        )

        portfolio_value_after = (
            portfolio_value
            * (1.0 + portfolio_return)
        )

        loss_percentage = (
            portfolio_loss
            / portfolio_value
        )

        rows.append(
    {
        "scenario": scenario_name,
        "portfolio_return": portfolio_return,
        "portfolio_loss": portfolio_loss,
        "portfolio_value": portfolio_value,
        "portfolio_value_before": portfolio_value,
        "portfolio_value_after": portfolio_value_after,
        "loss_percentage": loss_percentage,
    }
)
    report = pd.DataFrame(
        rows
    )

    if report.empty:
        raise ValueError(
            "stress report contains no scenarios."
        )

    return report.set_index(
        "scenario"
    )


# =====================================================================
# Asset-level contribution analysis
# =====================================================================

def hypothetical_stress_contributions(
    scenario: Mapping[str, float],
    weights: pd.Series,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> pd.DataFrame:
    """
    Calculate asset-level contributions to a hypothetical stress.

    Return contribution:

        w_i * shock_i

    Dollar loss contribution:

        -V_0 * w_i * shock_i

    The dollar loss contributions sum to the total portfolio
    stress loss.
    """

    weights = validate_weights(
        weights
    )

    portfolio_value = validate_portfolio_value(
        portfolio_value
    )

    shocks = validate_scenario(
        scenario=scenario,
        assets=weights.index,
    )

    return_contributions = (
        weights * shocks
    )

    loss_contributions = (
        -portfolio_value
        * return_contributions
    )

    total_loss = float(
        loss_contributions.sum()
    )

    if np.isclose(
        total_loss,
        0.0,
        atol=1e-12,
    ):
        contribution_percentages = pd.Series(
            np.nan,
            index=weights.index,
            dtype=float,
        )
    else:
        contribution_percentages = (
            loss_contributions
            / total_loss
        )

    return pd.DataFrame(
        {
            "weight": weights,
            "shock": shocks,
            "return_contribution": return_contributions,
            "loss_contribution": loss_contributions,
            "loss_contribution_pct": contribution_percentages,
        }
    )


# =====================================================================
# Complete analysis convenience function
# =====================================================================

def run_hypothetical_stress_analysis(
    weights: pd.Series,
    scenarios: Mapping[str, Mapping[str, float]] | None = None,
    portfolio_value: float = DEFAULT_PORTFOLIO_VALUE,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Run the complete hypothetical stress analysis.

    Returns
    -------
    report:
        Scenario-level stress report.

    contributions:
        Dictionary containing asset-level contribution tables.
    """

    weights = validate_weights(
        weights
    )

    portfolio_value = validate_portfolio_value(
        portfolio_value
    )

    if scenarios is None:
        scenarios = DEFAULT_HYPOTHETICAL_SCENARIOS

    validated_scenarios = validate_scenarios(
        scenarios=scenarios,
        assets=weights.index,
    )

    report = hypothetical_stress_report(
        weights=weights,
        scenarios=validated_scenarios,
        portfolio_value=portfolio_value,
    )

    contributions = {}

    for scenario_name, scenario in validated_scenarios.items():

        contributions[scenario_name] = (
            hypothetical_stress_contributions(
                scenario=scenario,
                weights=weights,
                portfolio_value=portfolio_value,
            )
        )

    return report, contributions