"""Snapshotted daily price downloads for the experiment scripts.

Yahoo Finance revises adjusted prices retroactively: every new dividend
rescales the whole adjusted history of that stock, so the same 2020-2024 period
downloaded a month apart gives slightly different price levels. Pooled models
that use raw price levels as features (the Random Forest in particular) then
produce slightly different results, which makes figures drift between runs.

The experiments therefore read from a local snapshot. The first call for a
ticker downloads it and writes a CSV under data/raw/; later calls read that
file, so every script in one analysis sees identical inputs. Delete the files
(or pass refresh=True) to take a fresh snapshot.

The live web application does not use this module: it needs current prices.
"""

import os

import pandas as pd

#: Directory holding the CSV snapshots. Ignored by git: the data belong to
#: Yahoo Finance and are re-downloaded rather than redistributed.
RAW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

#: Study period shared by every experiment.
START_DATE = "2020-01-01"
END_DATE = "2024-12-31"


def snapshot_path(ticker: str, start: str = START_DATE, end: str = END_DATE,
                  cache_dir: str = RAW_DIR) -> str:
    """Location of the CSV snapshot for one ticker and period."""
    return os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")


def load_prices(ticker: str, start: str = START_DATE, end: str = END_DATE,
                cache_dir: str = RAW_DIR, refresh: bool = False) -> pd.DataFrame:
    """Daily OHLCV prices for a ticker, from the snapshot when one exists.

    Prices are dividend- and split-adjusted (auto_adjust is pinned, because
    its yfinance default has changed between releases).

    Args:
        ticker: Exchange symbol as Yahoo Finance spells it (e.g. BRK-B).
        start: First date, inclusive.
        end: Last date, exclusive (yfinance convention).
        cache_dir: Directory for snapshot files.
        refresh: Download again even if a snapshot exists.

    Returns:
        DataFrame indexed by date with single-level OHLCV columns. Empty if
        Yahoo Finance has no data for the ticker in the period; empty results
        are not cached, so a transient failure is retried on the next call.
    """
    path = snapshot_path(ticker, start, end, cache_dir)
    if not refresh and os.path.exists(path):
        return pd.read_csv(path, index_col=0, parse_dates=True)

    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, auto_adjust=True,
                     progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.columns.name = None

    if len(df):
        os.makedirs(cache_dir, exist_ok=True)
        df.to_csv(path)
    return df
