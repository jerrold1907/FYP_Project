"""Unit tests for the price snapshot used by the experiment scripts.

yfinance is replaced with stubs throughout, so these tests never touch the
network.
"""

import numpy as np
import pandas as pd
import pytest

from src.market_data import load_prices, snapshot_path


def _frame(n_rows=5):
    index = pd.bdate_range("2021-03-01", periods=n_rows, name="Date")
    close = 100 + np.arange(n_rows) / 3          # values with long decimals
    return pd.DataFrame({"Close": close, "High": close + 1, "Low": close - 1,
                         "Open": close, "Volume": 1_000_000 + np.arange(n_rows)},
                        index=index)


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("yfinance must not be called")
    monkeypatch.setattr("yfinance.download", refuse)


def test_existing_snapshot_is_read_without_downloading(tmp_path, no_network):
    original = _frame()
    original.to_csv(snapshot_path("TEST", cache_dir=str(tmp_path)))

    loaded = load_prices("TEST", cache_dir=str(tmp_path))

    pd.testing.assert_frame_equal(loaded, original, check_freq=False,
                                  check_names=False)


def test_first_call_downloads_and_saves(tmp_path, monkeypatch):
    calls = []

    def fake_download(ticker, **kwargs):
        calls.append((ticker, kwargs))
        return _frame()

    monkeypatch.setattr("yfinance.download", fake_download)
    first = load_prices("TEST", cache_dir=str(tmp_path))
    second = load_prices("TEST", cache_dir=str(tmp_path))

    assert len(calls) == 1                         # second call hit the file
    assert calls[0][1]["auto_adjust"] is True      # pinned, not the default
    pd.testing.assert_frame_equal(first, second, check_freq=False,
                                  check_names=False)


def test_refresh_downloads_again(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr("yfinance.download",
                        lambda ticker, **kwargs: calls.append(1) or _frame())
    load_prices("TEST", cache_dir=str(tmp_path))
    load_prices("TEST", cache_dir=str(tmp_path), refresh=True)
    assert len(calls) == 2


def test_empty_download_is_not_cached(tmp_path, monkeypatch):
    monkeypatch.setattr("yfinance.download",
                        lambda ticker, **kwargs: pd.DataFrame())
    assert load_prices("GONE", cache_dir=str(tmp_path)).empty
    assert not (tmp_path / "GONE_2020-01-01_2024-12-31.csv").exists()


def test_multiindex_columns_are_flattened(tmp_path, monkeypatch):
    frame = _frame()
    frame.columns = pd.MultiIndex.from_product([frame.columns, ["TEST"]],
                                               names=["Price", "Ticker"])
    monkeypatch.setattr("yfinance.download", lambda ticker, **kwargs: frame)
    loaded = load_prices("TEST", cache_dir=str(tmp_path))
    assert list(loaded.columns) == ["Close", "High", "Low", "Open", "Volume"]
