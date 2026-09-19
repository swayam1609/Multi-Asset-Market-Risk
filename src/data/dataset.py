"""
dataset.py

Computes daily log returns from the cleaned price dataset,
validates the results, and saves the returns to data/processed/.
"""

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def compute_log_returns(
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute daily logarithmic returns.
    """

    returns = np.log(
        prices / prices.shift(1)
    )

    returns = returns.dropna()

    returns.index.name = "Date"

    return returns


def validate_returns(
    returns: pd.DataFrame,
) -> None:
    """
    Validate the calculated return dataset.
    """

    assert not returns.empty, (
        "Returns DataFrame is empty."
    )

    assert returns.isna().sum().sum() == 0, (
        "Returns contains missing values."
    )

    assert np.isfinite(
        returns.to_numpy()
    ).all(), (
        "Returns contains non-finite values."
    )


def build_returns(
    prices_path: Path = None,
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """
    Load cleaned prices, calculate log returns,
    validate them, and save the result.
    """

    processed_dir = Path(processed_dir)

    if prices_path is None:

        prices_path = (
            processed_dir / "prices.csv"
        )

    else:

        prices_path = Path(prices_path)

    if not prices_path.exists():

        raise FileNotFoundError(
            f"Processed prices file not found: "
            f"{prices_path}. "
            "Run clean_prices() first."
        )

    prices = pd.read_csv(
        prices_path,
        index_col=0,
        parse_dates=True,
    )

    returns = compute_log_returns(
        prices
    )

    validate_returns(
        returns
    )

    output_path = (
        processed_dir / "returns.csv"
    )

    returns.to_csv(
        output_path
    )

    print(
        f"[saved] daily log returns -> "
        f"{output_path}"
    )

    return returns


if __name__ == "__main__":
    build_returns()