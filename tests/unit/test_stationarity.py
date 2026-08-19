"""Unit tests for stationarity diagnostics.

Test series are constructed so the correct verdict is known in advance:
white noise is stationary, a random walk is not.
"""

import numpy as np
import pandas as pd
import pytest

from src.stationarity import check_feature_frame, check_stationarity, summarise

pytest.importorskip("statsmodels", reason="statsmodels not installed")


def white_noise(n: int = 500, seed: int = 0) -> np.ndarray:
    """Stationary by construction."""
    return np.random.default_rng(seed).normal(0, 1, n)


def random_walk(n: int = 500, seed: int = 0) -> np.ndarray:
    """Non-stationary by construction: has a unit root."""
    return np.cumsum(np.random.default_rng(seed).normal(0, 1, n))


class TestStationarityDetection:
    """Verdicts must match the known behaviour of the test series."""

    def test_white_noise_is_stationary(self):
        result = check_stationarity(white_noise(), "noise")
        assert result.adf_says_stationary
        assert result.kpss_says_stationary
        assert result.verdict == "stationary"
        assert not result.problematic

    def test_random_walk_is_non_stationary(self):
        result = check_stationarity(random_walk(), "walk")
        assert not result.adf_says_stationary
        assert result.verdict == "non-stationary"
        assert result.problematic

    def test_differencing_a_random_walk_restores_stationarity(self):
        """The first difference of a random walk is white noise."""
        differenced = np.diff(random_walk())
        assert check_stationarity(differenced, "diff").verdict == "stationary"

    def test_result_records_name_and_sample_size(self):
        result = check_stationarity(white_noise(300), "my_feature")
        assert result.name == "my_feature"
        assert result.n_observations == 300

    def test_long_series_is_subsampled_to_cap(self):
        result = check_stationarity(white_noise(20_000), "long",
                                   max_observations=2000)
        assert result.n_observations <= 2000

    def test_too_short_series_rejected(self):
        with pytest.raises(ValueError, match="at least 20 finite observations"):
            check_stationarity([1.0, 2.0, 3.0], "tiny")

    def test_constant_series_rejected(self):
        with pytest.raises(ValueError, match="constant"):
            check_stationarity([5.0] * 100, "flat")

    def test_non_finite_values_are_dropped(self):
        values = white_noise(200).tolist() + [np.nan, np.inf]
        assert check_stationarity(values, "with_nan").n_observations == 200


class TestFeatureFrameTesting:
    """Frame-level helper must handle grouping and column selection."""

    @staticmethod
    def _frame() -> pd.DataFrame:
        return pd.DataFrame({
            "stationary_col": white_noise(400, seed=1),
            "trending_col": random_walk(400, seed=2),
            "ticker": ["AAA"] * 400,
        })

    def test_tests_every_requested_column(self):
        results = check_feature_frame(
            self._frame(), columns=["stationary_col", "trending_col"])
        assert {r.name for r in results} == {"stationary_col", "trending_col"}

    def test_identifies_the_non_stationary_column(self):
        results = check_feature_frame(
            self._frame(), columns=["stationary_col", "trending_col"])
        by_name = {r.name: r for r in results}
        assert not by_name["stationary_col"].problematic
        assert by_name["trending_col"].problematic

    def test_defaults_to_numeric_columns_excluding_group(self):
        results = check_feature_frame(self._frame(), group_column="ticker")
        assert "ticker" not in {r.name for r in results}
        assert len(results) == 2

    def test_uses_largest_group_when_grouping(self):
        frame = pd.DataFrame({
            "value": np.concatenate([white_noise(300, 3), white_noise(50, 4)]),
            "ticker": ["BIG"] * 300 + ["SMALL"] * 50,
        })
        results = check_feature_frame(frame, columns=["value"],
                                     group_column="ticker")
        assert results[0].n_observations == 300

    def test_empty_frame_rejected(self):
        with pytest.raises(ValueError, match="empty DataFrame"):
            check_feature_frame(pd.DataFrame({"a": []}))

    def test_unknown_column_rejected(self):
        with pytest.raises(ValueError, match="Columns not found"):
            check_feature_frame(self._frame(), columns=["nonexistent"])

    def test_unknown_group_column_rejected(self):
        with pytest.raises(ValueError, match="Group column"):
            check_feature_frame(self._frame(), columns=["stationary_col"],
                               group_column="missing")

    def test_constant_column_is_skipped_not_fatal(self):
        frame = self._frame()
        frame["flat"] = 1.0
        results = check_feature_frame(
            frame, columns=["stationary_col", "flat"])
        assert {r.name for r in results} == {"stationary_col"}


class TestSummaryTable:
    """The reporting table must be well formed."""

    def test_one_row_per_result_with_expected_columns(self):
        results = [check_stationarity(white_noise(200, s), f"f{s}")
                   for s in range(3)]
        table = summarise(results)
        assert len(table) == 3
        for column in ["Feature", "ADF p", "KPSS p", "Verdict", "n"]:
            assert column in table.columns
