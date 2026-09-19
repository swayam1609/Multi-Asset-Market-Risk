"""
downloader.py

Downloads historical market data from Yahoo Finance and saves
the downloaded datasets into data/raw/.
"""

from pathlib import Path

import pandas as pd
import yfinance as yf


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


TICKERS = {
    "SPY": "SPY",
    "EURUSD": "EURUSD=X",
    "TLT": "TLT",
}


DEFAULT_START = "2005-01-01"


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten Yahoo Finance MultiIndex columns if present."""

    if isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)

    return df


def download_asset(
    ticker_symbol: str,
    start: str = DEFAULT_START,
    end: str = None,
) -> pd.DataFrame:
    """Download historical data for one asset."""

    df = yf.download(
        ticker_symbol,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
    )

    if df is None or df.empty:
        raise RuntimeError(
            f"No data returned for '{ticker_symbol}'. "
            "Check the ticker symbol or internet connection."
        )

    df = _flatten_columns(df)

    df.index.name = "Date"

    return df


def download_raw_data(
    start: str = DEFAULT_START,
    end: str = None,
    raw_dir: Path = RAW_DATA_DIR,
    overwrite: bool = False,
) -> dict:
    """Download and save raw data for all project assets."""

    raw_dir = Path(raw_dir)

    raw_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    raw_data = {}

    for clean_name, ticker_symbol in TICKERS.items():

        out_path = raw_dir / f"{clean_name}.csv"

        if out_path.exists() and not overwrite:

            print(
                f"[skip] {out_path} already exists. "
                "Set overwrite=True to re-download."
            )

            raw_data[clean_name] = pd.read_csv(
                out_path,
                index_col=0,
                parse_dates=True,
            )

            continue

        print(
            f"[download] {ticker_symbol} -> {out_path}"
        )

        df = download_asset(
            ticker_symbol,
            start=start,
            end=end,
        )

        df.to_csv(out_path)

        raw_data[clean_name] = df

    return raw_data


if __name__ == "__main__":
    download_raw_data()