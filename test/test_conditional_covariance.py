import numpy as np
import pandas as pd
import pytest

from src.risk.conditional_covariance import (
    validate_returns_dataframe,
    standardized_residuals,
    rolling_standardized_correlation,
    conditional_covariance_matrices,
)


@pytest.fixture
def returns():
    rng = np.random.default_rng(42)

    values = rng.normal(
        loc=0.0002,
        scale=0.01,
        size=(500, 3),
    )

    return pd.DataFrame(
        values,
        index=pd.date_range(
            "2020-01-01",
            periods=500,
            freq="D",
        ),
        columns=[
            "SPY",
            "EURUSD=X",
            "TLT",
        ],
    )


def test_validate_returns_dataframe_accepts_valid_data(
    returns,
):
    result = validate_returns_dataframe(
        returns
    )

    assert result.equals(
        returns.astype(float)
    )


def test_validate_returns_dataframe_rejects_non_dataframe():
    with pytest.raises(TypeError):
        validate_returns_dataframe(
            np.ones((10, 3))
        )


def test_validate_returns_dataframe_rejects_single_asset():
    returns = pd.DataFrame(
        {"SPY": np.ones(100)}
    )

    with pytest.raises(ValueError):
        validate_returns_dataframe(
            returns
        )


def test_validate_returns_dataframe_rejects_nan(
    returns,
):
    modified = returns.copy()
    modified.iloc[10, 0] = np.nan

    with pytest.raises(ValueError):
        validate_returns_dataframe(
            modified
        )


def test_standardized_residuals_preserves_shape(
    returns,
):
    result = standardized_residuals(
        returns
    )

    assert result.shape == returns.shape
    assert result.index.equals(
        returns.index
    )
    assert list(result.columns) == list(
        returns.columns
    )


def test_standardized_residuals_is_finite(
    returns,
):
    result = standardized_residuals(
        returns
    )

    valid = result.dropna()

    assert np.isfinite(
        valid.to_numpy()
    ).all()


def test_rolling_correlation_returns_matrices(
    returns,
):
    result = rolling_standardized_correlation(
        returns,
        window=100,
    )

    assert len(result) > 0

    first = next(
        iter(result.values())
    )

    assert first.shape == (3, 3)
    assert list(first.index) == list(
        returns.columns
    )


def test_rolling_correlation_matrix_is_symmetric(
    returns,
):
    result = rolling_standardized_correlation(
        returns,
        window=100,
    )

    for matrix in result.values():
        assert np.allclose(
            matrix.to_numpy(),
            matrix.to_numpy().T,
        )


def test_conditional_covariance_returns_matrices(
    returns,
):
    result = conditional_covariance_matrices(
        returns,
        window=100,
    )

    assert len(result) > 0

    first = next(
        iter(result.values())
    )

    assert first.shape == (3, 3)


def test_conditional_covariance_is_symmetric(
    returns,
):
    result = conditional_covariance_matrices(
        returns,
        window=100,
    )

    for matrix in result.values():
        assert np.allclose(
            matrix.to_numpy(),
            matrix.to_numpy().T,
        )


def test_conditional_covariance_diagonal_is_positive(
    returns,
):
    result = conditional_covariance_matrices(
        returns,
        window=100,
    )

    for matrix in result.values():
        diagonal = np.diag(
            matrix.to_numpy()
        )

        assert (diagonal > 0).all()


def test_conditional_covariance_is_finite(
    returns,
):
    result = conditional_covariance_matrices(
        returns,
        window=100,
    )

    for matrix in result.values():
        assert np.isfinite(
            matrix.to_numpy()
        ).all()


def test_conditional_covariance_no_look_ahead(
    returns,
):
    original = conditional_covariance_matrices(
        returns,
        window=100,
    )

    modified = returns.copy()

    # Change an observation well after the first
    # covariance estimate.
    modified.iloc[200, 0] = 1_000.0

    changed = conditional_covariance_matrices(
        modified,
        window=100,
    )

    first_timestamp = list(
        original.keys()
    )[50]

    assert np.allclose(
        original[first_timestamp].to_numpy(),
        changed[first_timestamp].to_numpy(),
    )