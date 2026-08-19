"""Unit tests for the shared stock universe and ticker matching.

The universe is the single source of truth for every component, so these tests
guard against metadata drift and against the substring-matching hazards that
short tickers introduce.
"""

import pytest

from src.chatbot import STOCK_INFO as CHATBOT_INFO, classify_intent
from src.stock_universe import (
    STOCK_INFO,
    STOCK_NAMES,
    TICKERS,
    UNIVERSE,
    get,
    sectors,
)


class TestUniverseIntegrity:
    """The universe must be internally consistent and complete."""

    def test_contains_twenty_stocks(self):
        assert len(UNIVERSE) == 20
        assert len(TICKERS) == 20

    def test_tickers_are_unique(self):
        assert len(set(TICKERS)) == len(TICKERS)

    def test_includes_micron(self):
        micron = get("MU")
        assert micron.name == "Micron Technology"
        assert micron.sector == "Technology"
        assert micron.industry == "Semiconductors"

    def test_every_stock_has_complete_metadata(self):
        for stock in UNIVERSE:
            assert stock.ticker.isupper() and stock.ticker.isalpha()
            assert stock.name and stock.sector and stock.industry
            assert len(stock.description) > 20, stock.ticker

    def test_derived_mappings_agree_with_universe(self):
        assert set(STOCK_NAMES) == set(TICKERS)
        assert set(STOCK_INFO) == set(TICKERS)
        for stock in UNIVERSE:
            assert STOCK_NAMES[stock.ticker] == stock.name
            assert STOCK_INFO[stock.ticker]["sector"] == stock.sector

    def test_stock_is_immutable(self):
        """Frozen dataclass prevents accidental mutation of shared state."""
        with pytest.raises(Exception):
            UNIVERSE[0].ticker = "CHANGED"


class TestSectorDiversification:
    """Sector spread is the reason for expanding the universe."""

    def test_spans_at_least_eight_sectors(self):
        assert len(sectors()) >= 8

    def test_every_ticker_appears_in_exactly_one_sector(self):
        allocated = [t for tickers in sectors().values() for t in tickers]
        assert sorted(allocated) == sorted(TICKERS)

    def test_includes_defensive_and_cyclical_sectors(self):
        """Regime analysis needs sectors that behave differently."""
        present = set(sectors())
        for required in ["Healthcare", "Utilities", "Energy", "Industrials"]:
            assert required in present, f"{required} missing"

    def test_no_sector_dominates_more_than_half(self):
        largest = max(len(t) for t in sectors().values())
        assert largest <= len(TICKERS) / 2


class TestLookup:
    """Ticker lookup must be forgiving of case but strict about membership."""

    def test_lookup_is_case_insensitive(self):
        assert get("mu").ticker == "MU"
        assert get("Mu").ticker == "MU"

    def test_unknown_ticker_raises_with_guidance(self):
        with pytest.raises(KeyError, match="not in the universe"):
            get("NOTREAL")


class TestChatbotSynchronisation:
    """The chatbot must never drift out of step with the universe."""

    def test_chatbot_uses_the_shared_metadata(self):
        assert CHATBOT_INFO is STOCK_INFO


class TestTickerMatching:
    """Word-boundary matching prevents false ticker attribution."""

    def test_detects_new_tickers(self):
        _, found = classify_intent("Compare MU and NVDA")
        assert set(found) == {"MU", "NVDA"}

    def test_short_ticker_not_matched_inside_a_word(self):
        """MU must not match 'much' \u2014 the hazard that motivated the fix."""
        _, found = classify_intent("How much should I invest?")
        assert "MU" not in found

    def test_hd_not_matched_inside_a_word(self):
        _, found = classify_intent("What are the THD levels?")
        assert "HD" not in found

    def test_micron_query_does_not_match_advanced_micro_devices(self):
        """'Micro' in AMD's registered name must not capture Micron queries."""
        _, found = classify_intent("Tell me about Micron")
        assert found == ["MU"], f"expected only MU, got {found}"

    def test_amd_still_matched_by_its_own_ticker(self):
        _, found = classify_intent("Is AMD risky?")
        assert "AMD" in found

    def test_generic_corporate_words_match_nothing(self):
        """'Technology' alone cannot identify one company among several."""
        _, found = classify_intent("I like technology companies")
        assert found == []

    def test_distinctive_company_names_still_resolve(self):
        for query, expected in [
            ("Tell me about Caterpillar", "CAT"),
            ("What about Walmart", "WMT"),
            ("How is Chevron doing", None),      # not in universe
            ("Thoughts on NextEra", "NEE"),
        ]:
            _, found = classify_intent(query)
            if expected:
                assert expected in found, f"{query} -> {found}"
            else:
                assert found == [], f"{query} -> {found}"

    def test_lowercase_tickers_are_matched(self):
        _, found = classify_intent("compare mu and jnj")
        assert set(found) == {"MU", "JNJ"}

    def test_ticker_adjacent_to_punctuation_is_matched(self):
        _, found = classify_intent("Is XOM, or CAT, a better buy?")
        assert set(found) == {"XOM", "CAT"}
