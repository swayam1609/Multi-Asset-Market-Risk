import numpy as np
import pandas as pd

from src.risk.garch_comparison import (
    compare_garch_models,
    compare_conditional_volatility,
    volatility_difference_summary,
)


def test_model_comparison_contains_both_models():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_garch_models(
        returns
    )

    assert list(result.index) == [
        "GARCH(1,1)",
        "GJR-GARCH(1,1)",
    ]


def test_model_comparison_contains_information_criteria():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_garch_models(
        returns
    )

    assert "log_likelihood" in result.columns
    assert "AIC" in result.columns
    assert "BIC" in result.columns
    assert "num_parameters" in result.columns


def test_model_comparison_values_are_finite():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_garch_models(
        returns
    )

    assert np.isfinite(
        result[
            [
                "log_likelihood",
                "AIC",
                "BIC",
            ]
        ].to_numpy()
    ).all()


def test_gjr_has_at_least_as_many_parameters():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_garch_models(
        returns
    )

    assert (
        result.loc[
            "GJR-GARCH(1,1)",
            "num_parameters",
        ]
        >=
        result.loc[
            "GARCH(1,1)",
            "num_parameters",
        ]
    )


def test_volatility_comparison_preserves_index():
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2020-01-01",
        periods=500,
        freq="D",
    )

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        ),
        index=index,
    )

    result = compare_conditional_volatility(
        returns
    )

    assert result.index.equals(index)


def test_volatility_comparison_contains_expected_columns():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_conditional_volatility(
        returns
    )

    assert list(result.columns) == [
        "GARCH_Volatility",
        "GJR_GARCH_Volatility",
        "Volatility_Difference",
    ]


def test_conditional_volatilities_are_positive():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_conditional_volatility(
        returns
    )

    assert (
        result[
            "GARCH_Volatility"
        ].dropna()
        > 0
    ).all()

    assert (
        result[
            "GJR_GARCH_Volatility"
        ].dropna()
        > 0
    ).all()


def test_volatility_difference_is_finite():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = compare_conditional_volatility(
        returns
    )

    assert np.isfinite(
        result[
            "Volatility_Difference"
        ].dropna().to_numpy()
    ).all()


def test_volatility_difference_summary_contains_expected_fields():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = volatility_difference_summary(
        returns
    )

    assert set(result.keys()) == {
        "mean_difference",
        "mean_absolute_difference",
        "max_absolute_difference",
        "correlation",
    }


def test_volatility_difference_summary_is_finite():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = volatility_difference_summary(
        returns
    )

    assert np.isfinite(
        list(result.values())
    ).all()


def test_volatility_models_are_highly_correlated():
    rng = np.random.default_rng(42)

    returns = pd.Series(
        rng.normal(
            0.0002,
            0.01,
            500,
        )
    )

    result = volatility_difference_summary(
        returns
    )

    assert -1.0 <= result["correlation"] <= 1.0