"""Stationarity diagnostics for the engineered feature set.

Most statistical learning methods, including Random Forest and logistic
regression, assume the joint distribution of the data is stable. A
non-stationary feature breaks that assumption: a model trained on 2021 price
levels is being asked to generalise to 2024 price levels it never saw.

This module tests each feature for stationarity so the assumption can be
examined rather than assumed. It supplies concrete evidence for two claims the
project needs to make:

  - Which features are non-stationary, and therefore why raw price levels are
    poor inputs while returns and bounded oscillators are better.
  - Whether the label distribution shifts across market regimes, which would
    explain regime-dependent accuracy.

Two complementary tests are used because they have opposite null hypotheses,
and agreement between them is stronger evidence than either alone:

  - ADF (Augmented Dickey-Fuller): H0 = a unit root is present, i.e. the series
    is non-stationary. Rejecting H0 supports stationarity.
  - KPSS: H0 = the series is stationary. Rejecting H0 supports non-stationarity.

References:
    Dickey, D.A. and Fuller, W.A. (1979) 'Distribution of the estimators for
        autoregressive time series with a unit root', JASA 74(366), pp. 427-431.
    Kwiatkowski, D. et al. (1992) 'Testing the null hypothesis of stationarity
        against the alternative of a unit root', Journal of Econometrics 54,
        pp. 159-178.
"""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import pandas as pd


@dataclass
class StationarityResult:
    """Combined ADF and KPSS verdict for one series."""
    name: str
    adf_statistic: float
    adf_p_value: float
    kpss_statistic: float
    kpss_p_value: float
    n_observations: int
    alpha: float = 0.05

    @property
    def adf_says_stationary(self) -> bool:
        """ADF rejects its unit-root null, supporting stationarity."""
        return self.adf_p_value < self.alpha

    @property
    def kpss_says_stationary(self) -> bool:
        """KPSS fails to reject its stationarity null."""
        return self.kpss_p_value >= self.alpha

    @property
    def verdict(self) -> str:
        """Plain-language conclusion combining both tests."""
        if self.adf_says_stationary and self.kpss_says_stationary:
            return "stationary"
        if not self.adf_says_stationary and not self.kpss_says_stationary:
            return "non-stationary"
        if self.adf_says_stationary and not self.kpss_says_stationary:
            return "conflicting (possibly trend-stationary)"
        return "conflicting (possibly difference-stationary)"

    @property
    def problematic(self) -> bool:
        """True if the series is not clearly stationary."""
        return self.verdict != "stationary"


def check_stationarity(series: Sequence[float], name: str = "series",
                       alpha: float = 0.05,
                       max_observations: int = 5000) -> StationarityResult:
    """Run ADF and KPSS tests on a single series.

    Args:
        series: Observations in time order.
        name: Label for reporting.
        alpha: Significance level for both tests.
        max_observations: Cap on series length; longer series are subsampled
            evenly, since KPSS is slow and both tests are already decisive at
            this size.

    Returns:
        StationarityResult with both test outcomes and a combined verdict.

    Raises:
        ImportError: If statsmodels is unavailable.
        ValueError: If fewer than 20 finite observations remain, or the series
            is constant.
    """
    try:
        from statsmodels.tsa.stattools import adfuller, kpss
    except ImportError as exc:
        raise ImportError(
            "statsmodels is required for stationarity testing. "
            "Install with: pip install statsmodels"
        ) from exc

    values = np.asarray(series, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 20:
        raise ValueError(
            f"Need at least 20 finite observations, got {len(values)}")
    if np.std(values) == 0:
        raise ValueError(f"Series '{name}' is constant; tests are undefined")

    if len(values) > max_observations:
        step = len(values) // max_observations
        values = values[::step]

    adf_statistic, adf_p, *_ = adfuller(values, autolag="AIC")

    # KPSS warns when the p-value is outside its interpolation table; the
    # boundary value it returns is still the correct conservative reading.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        kpss_statistic, kpss_p, *_ = kpss(values, regression="c", nlags="auto")

    return StationarityResult(
        name=name,
        adf_statistic=float(adf_statistic),
        adf_p_value=float(adf_p),
        kpss_statistic=float(kpss_statistic),
        kpss_p_value=float(kpss_p),
        n_observations=len(values),
        alpha=alpha,
    )


def check_feature_frame(frame: pd.DataFrame,
                        columns: Optional[Sequence[str]] = None,
                        group_column: Optional[str] = None,
                        alpha: float = 0.05) -> list[StationarityResult]:
    """Test each feature column for stationarity.

    When `group_column` is given (ticker), each feature is tested on a single
    group rather than the pooled series. Pooling several stocks end to end would
    create artificial level jumps at the joins and make every feature look
    non-stationary regardless of its true behaviour.

    Args:
        frame: DataFrame of engineered features, ordered by date within group.
        columns: Feature columns to test. Defaults to all numeric columns
            except the group column.
        group_column: Column identifying the series each row belongs to.
        alpha: Significance level.

    Returns:
        One StationarityResult per tested column.

    Raises:
        ValueError: If the frame is empty or a requested column is absent.
    """
    if len(frame) == 0:
        raise ValueError("Cannot test an empty DataFrame")

    if columns is None:
        columns = [c for c in frame.select_dtypes(include=[np.number]).columns
                   if c != group_column]

    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"Columns not found in frame: {missing}")

    if group_column is not None:
        if group_column not in frame.columns:
            raise ValueError(f"Group column '{group_column}' not found")
        # Use the largest group for the cleanest single-series test.
        largest = frame[group_column].value_counts().idxmax()
        subset = frame[frame[group_column] == largest]
    else:
        subset = frame

    results = []
    for column in columns:
        try:
            results.append(check_stationarity(
                subset[column].to_numpy(), name=column, alpha=alpha))
        except ValueError:
            # Constant or too-short columns are skipped rather than aborting.
            continue

    return results


def summarise(results: Sequence[StationarityResult]) -> pd.DataFrame:
    """Assemble stationarity results into a reporting table."""
    return pd.DataFrame([{
        "Feature": r.name,
        "ADF stat": round(r.adf_statistic, 4),
        "ADF p": round(r.adf_p_value, 4),
        "KPSS stat": round(r.kpss_statistic, 4),
        "KPSS p": round(r.kpss_p_value, 4),
        "Verdict": r.verdict,
        "n": r.n_observations,
    } for r in results])
