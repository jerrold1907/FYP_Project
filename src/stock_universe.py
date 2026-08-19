"""Single source of truth for the stock universe.

Previously the ticker list was duplicated across app.py, chatbot.py, the
training script, three experiment scripts and a template, which made expanding
the universe error-prone. Everything now imports from here.

The universe was expanded from 10 to 20 equities to address a limitation
identified during evaluation: the original set was concentrated in large-cap US
technology, so any claim about generalisation was untestable. The additions
deliberately span sectors that behave differently across the market cycle —
healthcare and utilities are defensive, energy and industrials are cyclical,
semiconductors are highly volatile — which makes the regime analysis in
experiments/exp03 considerably more informative.

Sector labels follow the Global Industry Classification Standard groupings as
reported by Yahoo Finance.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Stock:
    """Reference metadata for one equity in the universe.

    Attributes:
        ticker: Exchange symbol used for data retrieval.
        name: Company name shown in the interface.
        sector: GICS sector, used for diversification analysis.
        industry: Narrower industry grouping.
        description: One-line summary shown by the chatbot.
    """
    ticker: str
    name: str
    sector: str
    industry: str
    description: str


#: The full universe. Order is stable so results are reproducible.
UNIVERSE: tuple[Stock, ...] = (
    # --- Technology: software, internet and consumer devices ---
    Stock("AAPL", "Apple Inc.", "Technology", "Consumer Electronics",
          "Designs and sells smartphones, computers, tablets and digital services."),
    Stock("MSFT", "Microsoft Corp.", "Technology", "Software",
          "Develops operating systems, cloud services and enterprise software."),
    Stock("GOOGL", "Alphabet Inc.", "Technology", "Internet Services",
          "Operates Google search, cloud computing, YouTube and AI research."),
    Stock("META", "Meta Platforms Inc.", "Technology", "Social Media",
          "Operates Facebook, Instagram and WhatsApp, and invests in VR."),

    # --- Technology: semiconductors, the most volatile sub-sector held ---
    Stock("NVDA", "NVIDIA Corp.", "Technology", "Semiconductors",
          "Designs GPUs and AI computing hardware for gaming and data centres."),
    Stock("AMD", "Advanced Micro Devices", "Technology", "Semiconductors",
          "Produces CPUs, GPUs and adaptive computing chips for PCs and servers."),
    Stock("MU", "Micron Technology", "Technology", "Semiconductors",
          "Manufactures DRAM and NAND memory and storage products. Highly "
          "cyclical, tracking the memory supply cycle."),

    # --- Consumer cyclical ---
    Stock("AMZN", "Amazon.com Inc.", "Consumer Cyclical", "E-Commerce",
          "Runs an e-commerce marketplace, AWS cloud computing and streaming."),
    Stock("TSLA", "Tesla Inc.", "Consumer Cyclical", "Automobiles",
          "Manufactures electric vehicles, battery storage and solar products."),
    Stock("HD", "The Home Depot Inc.", "Consumer Cyclical", "Home Improvement",
          "Operates home improvement retail stores across North America."),

    # --- Consumer defensive ---
    Stock("KO", "The Coca-Cola Co.", "Consumer Defensive", "Beverages",
          "Manufactures and distributes non-alcoholic beverages worldwide."),
    Stock("PEP", "PepsiCo Inc.", "Consumer Defensive", "Beverages & Snacks",
          "Produces beverages and snack foods including Pepsi and Lay's."),
    Stock("WMT", "Walmart Inc.", "Consumer Defensive", "Discount Retail",
          "Operates hypermarkets, discount stores and grocery retail."),

    # --- Financial services ---
    Stock("JPM", "JPMorgan Chase & Co.", "Financial Services", "Banking",
          "Provides investment banking, consumer banking and asset management."),

    # --- Healthcare: defensive, low correlation with technology ---
    Stock("JNJ", "Johnson & Johnson", "Healthcare", "Pharmaceuticals",
          "Develops pharmaceuticals and medical devices."),
    Stock("UNH", "UnitedHealth Group Inc.", "Healthcare", "Health Insurance",
          "Provides health insurance and healthcare services."),

    # --- Energy: cyclical, often moves against growth equities ---
    Stock("XOM", "Exxon Mobil Corp.", "Energy", "Oil & Gas",
          "Explores, produces and refines crude oil and natural gas."),

    # --- Industrials: sensitive to the economic cycle ---
    Stock("CAT", "Caterpillar Inc.", "Industrials", "Heavy Machinery",
          "Manufactures construction and mining equipment and engines."),

    # --- Communication services ---
    Stock("DIS", "The Walt Disney Co.", "Communication Services", "Entertainment",
          "Operates media networks, theme parks and streaming services."),

    # --- Utilities: the lowest-volatility sector represented ---
    Stock("NEE", "NextEra Energy Inc.", "Utilities", "Renewable Utilities",
          "Generates and distributes electricity, with large renewable assets."),
)

#: Ticker symbols in universe order. Used by training and experiment scripts.
TICKERS: list[str] = [stock.ticker for stock in UNIVERSE]

#: Ticker to display name, used by the recommendation endpoint.
STOCK_NAMES: dict[str, str] = {s.ticker: s.name for s in UNIVERSE}

#: Ticker to full metadata, used by the chatbot for informational queries.
STOCK_INFO: dict[str, dict[str, str]] = {
    s.ticker: {
        "name": s.name,
        "sector": s.sector,
        "industry": s.industry,
        "description": s.description,
    }
    for s in UNIVERSE
}


def sectors() -> dict[str, list[str]]:
    """Group tickers by sector.

    Returns:
        Mapping of sector name to the tickers it contains, useful for
        diversification reporting and for sector-filtered recommendations.
    """
    grouped: dict[str, list[str]] = {}
    for stock in UNIVERSE:
        grouped.setdefault(stock.sector, []).append(stock.ticker)
    return grouped


def get(ticker: str) -> Stock:
    """Look up a stock by ticker.

    Args:
        ticker: Exchange symbol, case-insensitive.

    Returns:
        The matching Stock.

    Raises:
        KeyError: If the ticker is not in the universe.
    """
    symbol = ticker.upper()
    for stock in UNIVERSE:
        if stock.ticker == symbol:
            return stock
    raise KeyError(
        f"'{ticker}' is not in the universe. Available: {', '.join(TICKERS)}")
