"""
cleaner.py

Loads raw per-asset market data from data/raw/,
extracts adjusted closing prices, aligns the assets,
handles missing observations, and saves the cleaned
price dataset into data/processed/.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


ASSET_NAMES = [
    "SPY",
    "EURUSD",
    "TLT",
]


def _read_raw_csv(path: Path) -> pd.DataFrame:
    """
    Read one raw CSV file and make sure the Close column
    is available.
    """

    df = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
    )

    if "Close" not in df.columns:
        raise ValueError(
            f"'Close' column not found in {path}. "
            f"Columns present: {list(df.columns)}"
        )

    return df


def load_raw_prices(
    raw_dir: Path = RAW_DATA_DIR,
) -> dict:
    """
    Load the raw price data for all project assets.
    """

    raw_dir = Path(raw_dir)

    series_dict = {}

    for asset_name in ASSET_NAMES:

        path = raw_dir / f"{asset_name}.csv"

        if not path.exists():
            raise FileNotFoundError(
                f"Raw data file not found: {path}. "
                "Run download_raw_data() first."
            )

        df = _read_raw_csv(path)

        series_dict[asset_name] = (
            df["Close"]
            .rename(asset_name)
        )

    return series_dict


def clean_prices(
    raw_dir: Path = RAW_DATA_DIR,
    processed_dir: Path = PROCESSED_DATA_DIR,
    max_ffill_days: int = 3,
) -> pd.DataFrame:
    """
    Combine asset prices, align dates, handle short
    missing-data gaps, and save the cleaned dataset.
    """

    series_dict = load_raw_prices(raw_dir)

    prices = pd.concat(
        series_dict.values(),
        axis=1,
    )

    prices.columns = ASSET_NAMES

    prices = prices.sort_index()

    prices = prices[
        ~prices.index.duplicated(
            keep="first"
        )
    ]

    # Fill only short gaps caused by calendar differences.
    prices = prices.ffill(
        limit=max_ffill_days
    )

    # Remove rows that still contain missing values.
    prices = prices.dropna()

    prices.index.name = "Date"

    processed_dir = Path(processed_dir)

    processed_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        processed_dir / "prices.csv"
    )

    prices.to_csv(output_path)

    print(
        f"[saved] clean prices -> {output_path}"
    )

    return prices


if __name__ == "__main__":
    clean_prices()