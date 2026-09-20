"""AI Chatbot Module for the Stock Advisor.

Provides a conversational interface where users can ask questions about
stocks, recommendations, and comparisons. The chatbot uses stock data
from the recommendation system and can optionally format responses
through a local LLM (Ollama) for natural language output.

Supported question types:
- "Why did you recommend X?" / "Why is X suitable?"
- "Is X risky?" / "How risky is X?"
- "Compare X and Y" / "X vs Y"
- "What industry is X in?"
- "Tell me about X" / "What do you think of X?"
- "What should I buy?" / "Best stock for me?"

Architecture:
    User question → Intent classification (keyword matching)
        → Route to handler (retrieves relevant data)
        → Format response (Ollama LLM or template-based fallback)
"""

import re
from typing import Optional

# Company metadata comes from the shared universe definition so the chatbot can
# never drift out of step with the set of stocks the advisor actually analyses.
from src.stock_universe import STOCK_INFO


#: Company-name words too generic to identify a stock on their own. Matching on
#: these would attribute a question to the wrong company.
_AMBIGUOUS_NAME_WORDS = frozenset({
    "INC", "CORP", "CO", "GROUP", "PLATFORMS", "TECHNOLOGY", "TECHNOLOGIES",
    "COMPANY", "HOLDINGS", "THE", "AND", "ENERGY", "MICRO", "DEVICES",
    "ADVANCED", "SERVICES", "SYSTEMS", "INTERNATIONAL",
})


def classify_intent(message: str) -> tuple[str, list[str]]:
    """Classify user message intent and extract stock tickers mentioned.

    Ticker and company-name matching both use word boundaries. Plain substring
    matching is unsafe with this universe: the ticker MU would match inside
    "how much", and AMD's registered name "Advanced Micro Devices" would match
    a question about Micron. Generic corporate words are also excluded, since
    "Technology" alone cannot identify one company among several.

    Args:
        message: Raw user message.

    Returns:
        Tuple of (intent_type, tickers mentioned in the message).
    """
    msg = message.upper().strip()

    # Tickers, matched as whole words only.
    tickers_found = [t for t in STOCK_INFO
                     if re.search(rf"\b{re.escape(t)}\b", msg)]

    # Company names, matched on distinctive words at word boundaries.
    for ticker, info in STOCK_INFO.items():
        if ticker in tickers_found:
            continue
        distinctive = [w for w in re.findall(r"[A-Z]+", info["name"].upper())
                       if len(w) > 3 and w not in _AMBIGUOUS_NAME_WORDS]
        if any(re.search(rf"\b{re.escape(word)}\b", msg) for word in distinctive):
            tickers_found.append(ticker)

    # Intent classification by keyword matching
    msg_lower = message.lower()

    if any(w in msg_lower for w in ["compare", " vs ", "versus", "difference between", "which is better"]):
        return "compare", tickers_found

    if any(w in msg_lower for w in ["why", "reason", "explain", "how come", "based on", "what makes", "how is"]):
        if any(w in msg_lower for w in ["recommend", "suitable", "suggest", "pick"]):
            return "explain_recommendation", tickers_found
        # Follow-up: "why is X high risk?", "this is based on?", "explain the risk"
        if tickers_found or any(w in msg_lower for w in ["risk", "risky", "high risk", "low risk", "rating"]):
            return "explain_risk", tickers_found

    if any(w in msg_lower for w in ["risk", "risky", "dangerous", "volatile", "safe"]):
        return "risk_analysis", tickers_found

    if any(w in msg_lower for w in ["industry", "sector", "what does", "what is", "tell me about", "info"]):
        return "stock_info", tickers_found

    if any(w in msg_lower for w in ["best", "should i buy", "what should", "recommend me", "top pick", "suggestion"]):
        return "best_pick", tickers_found

    if any(w in msg_lower for w in ["portfolio", "diversif", "all stocks", "overall"]):
        return "portfolio_overview", tickers_found

    # Default: if ticker mentioned, give info; otherwise general help
    if tickers_found:
        return "stock_info", tickers_found

    return "general_help", tickers_found


#: Intents whose answers depend on the user's risk profile. These cannot be
#: answered until the risk questionnaire has been completed, because doing so
#: would mean assuming a risk tolerance the user never stated.
PERSONALISED_INTENTS = {
    "best_pick",
    "portfolio_overview",
    "explain_recommendation",
}

_ASSESSMENT_PROMPT = (
    "**I don't know your risk profile yet.**\n\n"
    "I can't answer that without knowing your risk tolerance — recommending "
    "stocks without it would mean guessing how much risk you're comfortable "
    "taking.\n\n"
    "Please complete the **5-question risk assessment** on the "
    "Recommendations page first. It takes under a minute, and then I can tell "
    "you which stocks suit you and why.\n\n"
    "In the meantime I can still help with:\n"
    "- \"Tell me about NVDA\" — company details and current signal\n"
    "- \"Is TSLA risky?\" — volatility and momentum analysis\n"
    "- \"Compare AAPL and MSFT\" — side-by-side metrics"
)


def generate_response(
    intent: str,
    tickers: list[str],
    stock_data: list[dict],
    risk_category: Optional[str],
) -> str:
    """Generate the factual, data-grounded response for an intent.

    This layer is deterministic: every figure comes from the ML model output
    or the stock metadata table. The optional LLM layer (see
    `summarise_with_llm`) only summarises this text — it never invents data.

    Args:
        intent: Classified intent type
        tickers: Stock tickers mentioned in the query
        stock_data: List of stock analysis dicts from /api/recommend
        risk_category: User's risk category, or None if the risk questionnaire
            has not been completed. Never substitute a default value here.

    Returns:
        Response string
    """
    # Refuse to answer profile-dependent questions without a real assessment.
    if intent in PERSONALISED_INTENTS and risk_category is None:
        return _ASSESSMENT_PROMPT

    # Build context from stock data
    stock_map = {s["ticker"]: s for s in stock_data}

    if intent == "compare":
        return _handle_compare(tickers, stock_map, risk_category)
    elif intent == "explain_recommendation":
        return _handle_explain(tickers, stock_map, risk_category)
    elif intent == "explain_risk":
        return _handle_explain_risk(tickers, stock_map)
    elif intent == "risk_analysis":
        return _handle_risk(tickers, stock_map)
    elif intent == "stock_info":
        return _handle_info(tickers, stock_map)
    elif intent == "best_pick":
        return _handle_best_pick(stock_data, risk_category)
    elif intent == "portfolio_overview":
        return _handle_portfolio(stock_data, risk_category)
    else:
        return _handle_help()


def _handle_compare(tickers: list[str], stock_map: dict,
                    risk_category: Optional[str]) -> str:
    """Compare two or more stocks side by side."""
    if len(tickers) < 2:
        return ("Please mention two stocks to compare. For example: "
                "\"Compare AAPL and MSFT\" or \"NVDA vs GOOGL\".")

    # No cached analysis available at all — point at the dedicated page, which
    # fetches live data without needing a completed risk assessment.
    if not stock_map:
        pair = " and ".join(tickers[:2])
        return (f"I don't have current analysis loaded for {pair} yet.\n\n"
                "Use the **Compare** page for a live side-by-side comparison "
                "with overlaid price charts, or complete the risk assessment on "
                "the **Recommendations** page so I can analyse all stocks and "
                "discuss them here.")

    stocks = [stock_map[t] for t in tickers[:3] if t in stock_map]
    if len(stocks) < 2:
        missing = [t for t in tickers if t not in stock_map]
        return (f"I only have analysis for some of those. Missing: "
                f"{', '.join(missing)}. Try the **Compare** page for live data "
                "on any pair.")

    lines = [f"**Comparison: {' vs '.join(s['ticker'] for s in stocks)}**\n"]
    lines.append("| Metric | " + " | ".join(s["ticker"] for s in stocks) + " |")
    lines.append("|--------|" + "|".join(["--------"] * len(stocks)) + "|")
    lines.append("| Price | " + " | ".join(f"${s['current_price']:.2f}" for s in stocks) + " |")
    lines.append("| 30d Change | " + " | ".join(f"{s['price_change_30d']:+.2f}%" for s in stocks) + " |")
    lines.append("| Signal | " + " | ".join(s["signal"] for s in stocks) + " |")
    lines.append("| RSI | " + " | ".join(f"{s['rsi']:.1f}" for s in stocks) + " |")
    lines.append("| MACD | " + " | ".join(f"{s['macd']:.2f}" for s in stocks) + " |")
    lines.append("| Volatility | " + " | ".join(f"{s['volatility']:.2f}%" for s in stocks) + " |")
    # The suitability rating is profile-dependent, so only show it when known.
    if risk_category is not None:
        lines.append("| Suitability | " + " | ".join(
            s["suitability"]["rating"] for s in stocks) + " |")

    # Profile-specific verdict, only when a real assessment exists.
    if risk_category is None:
        lines.append("\n*Complete the risk assessment on the Recommendations page "
                     "and I can tell you which of these suits your profile.*")
    else:
        suitable_stocks = [s for s in stocks if s["suitability"]["rating"] == "Suitable"]
        if suitable_stocks:
            best = max(suitable_stocks, key=lambda s: -s["volatility"])
            lines.append(f"\n**For your {risk_category} profile:** {best['ticker']} appears most aligned — "
                         f"it has a {best['signal']} signal with {best['volatility']:.2f}% volatility.")
        else:
            lines.append(f"\n**For your {risk_category} profile:** None of these are rated Suitable. "
                         "Consider reviewing stocks with stronger signals.")

    return "\n".join(lines)


def _handle_explain(tickers: list[str], stock_map: dict,
                    risk_category: Optional[str]) -> str:
    """Explain why a stock was recommended or not."""
    if not tickers:
        return "Which stock would you like me to explain? Try: \"Why is AAPL recommended?\""

    ticker = tickers[0]
    stock = stock_map.get(ticker)
    if not stock:
        return (f"I don't have current analysis for {ticker}. Please run the "
                "recommendations on the **Recommendations** page first.")

    rating = stock["suitability"]["rating"]
    explanation = stock["suitability"]["explanation"]
    signal = stock["signal"]

    lines = [f"**Why {ticker} is rated \"{rating}\" for you:**\n"]
    lines.append(f"The ML model analysed {ticker}'s technical indicators and predicted a "
                 f"**{signal}** signal. Combined with your **{risk_category}** risk profile, "
                 f"the suitability engine determined this is **{rating}**.\n")
    lines.append(f"*Detailed reasoning:* {explanation}\n")

    # Add indicator context
    lines.append("**Key indicators driving this:**")
    if stock["rsi"] > 70:
        lines.append(f"• RSI at {stock['rsi']:.1f} — overbought territory (potential pullback)")
    elif stock["rsi"] < 30:
        lines.append(f"• RSI at {stock['rsi']:.1f} — oversold (potential rebound opportunity)")
    else:
        lines.append(f"• RSI at {stock['rsi']:.1f} — neutral momentum zone")

    if stock["macd"] > 0:
        lines.append(f"• MACD positive ({stock['macd']:.2f}) — upward trend momentum")
    else:
        lines.append(f"• MACD negative ({stock['macd']:.2f}) — downward pressure")

    lines.append(f"• 30-day price change: {stock['price_change_30d']:+.2f}%")
    lines.append(f"• Volatility: {stock['volatility']:.2f}%")

    return "\n".join(lines)


def _handle_explain_risk(tickers: list[str], stock_map: dict) -> str:
    """Explain in plain language why a stock received its risk rating."""
    if not tickers:
        return (
            "Which stock's risk rating would you like me to explain? "
            "Try: \"Why is MU high risk?\""
        )

    ticker = tickers[0]
    stock = stock_map.get(ticker)
    name = STOCK_INFO.get(ticker, {}).get("name", ticker)

    if not stock:
        return (
            f"I don't have current data for {ticker}. "
            "Please run the recommendations first so I have live figures to work from."
        )

    vol = stock["volatility"]
    rsi = stock["rsi"]
    change = stock["price_change_30d"]
    signal = stock["signal"]

    risk_level = "LOW" if vol < 1.5 else "MODERATE" if vol < 3.0 else "HIGH"

    reasons = []

    # Volatility explanation
    if vol >= 3.0:
        reasons.append(
            f"**Volatility ({vol:.2f}%/day)** is the biggest factor — "
            "the stock moves more than 3% in a typical day, meaning large "
            "gains or losses can happen very quickly."
        )
    elif vol >= 1.5:
        reasons.append(
            f"**Volatility ({vol:.2f}%/day)** is moderate — noticeable swings "
            "but not extreme."
        )
    else:
        reasons.append(
            f"**Volatility ({vol:.2f}%/day)** is low — the stock price is "
            "relatively stable day-to-day."
        )

    # RSI explanation
    if rsi > 70:
        reasons.append(
            f"**RSI ({rsi:.1f})** is in overbought territory, which means the "
            "stock may have risen faster than fundamentals justify — a pullback "
            "is more likely."
        )
    elif rsi < 30:
        reasons.append(
            f"**RSI ({rsi:.1f})** signals the stock is oversold — it has fallen "
            "sharply and may continue declining or be due a bounce."
        )
    else:
        reasons.append(
            f"**RSI ({rsi:.1f})** is in the neutral range (30–70), so momentum "
            "alone doesn't raise a flag."
        )

    # 30-day trend explanation
    if change <= -5:
        reasons.append(
            f"**30-day price trend ({change:+.2f}%)** is negative — the stock "
            "has been falling steadily, adding to downside risk."
        )
    elif change >= 10:
        reasons.append(
            f"**30-day trend ({change:+.2f}%)** is a strong uptrend, but rapid "
            "gains can also mean a sharper correction if sentiment shifts."
        )
    else:
        reasons.append(
            f"**30-day trend ({change:+.2f}%)** is relatively flat — no strong "
            "directional signal from recent price action."
        )

    # Model signal explanation
    signal_note = {
        "Buy":   "The model sees positive momentum and favourable features.",
        "Hold":  "The model rates it neutral — not compelling enough to buy or avoid.",
        "Avoid": "The model predicts unfavourable conditions, which contributes to a higher risk rating.",
    }.get(signal, "")
    if signal_note:
        reasons.append(f"**Model signal ({signal}):** {signal_note}")

    lines = [
        f"**Why is {ticker} ({name}) rated {risk_level} risk?**\n",
        "Here's what drives that rating:\n",
    ]
    lines += [f"{i+1}. {r}" for i, r in enumerate(reasons)]
    lines.append(
        f"\nThe overall rating is **{risk_level}** because "
        + (
            "the daily volatility alone puts it in the high-risk bucket — "
            "even if other signals are mixed, that level of price movement "
            "is significant for most investors."
            if risk_level == "HIGH"
            else "the combination of signals above places it in the moderate range."
            if risk_level == "MODERATE"
            else "all signals point to a stable, lower-risk profile."
        )
    )
    return "\n".join(lines)


def _handle_risk(tickers: list[str], stock_map: dict) -> str:
    """Analyse the risk level of a stock."""
    if not tickers:
        return "Which stock would you like me to assess for risk? Try: \"Is TSLA risky?\""

    ticker = tickers[0]
    stock = stock_map.get(ticker)
    if not stock:
        return f"I don't have current data for {ticker}."

    vol = stock["volatility"]
    rsi = stock["rsi"]
    change = stock["price_change_30d"]

    risk_level = "LOW" if vol < 1.5 else "MODERATE" if vol < 3.0 else "HIGH"

    lines = [f"**Risk Assessment for {ticker} ({STOCK_INFO.get(ticker, {}).get('name', '')}):**\n"]
    lines.append(f"Overall Risk Level: **{risk_level}**\n")
    lines.append(f"• **Volatility:** {vol:.2f}% daily — {'stable and predictable' if vol < 1.5 else 'moderate swings expected' if vol < 3 else 'large price swings likely'}")
    lines.append(f"• **RSI:** {rsi:.1f} — {'oversold (higher risk of continued decline)' if rsi < 30 else 'overbought (risk of pullback)' if rsi > 70 else 'neutral range'}")
    lines.append(f"• **30-day trend:** {change:+.2f}% — {'declining' if change < -5 else 'relatively flat' if abs(change) < 5 else 'strong uptrend'}")
    lines.append(f"• **Model signal:** {stock['signal']} — {'positive outlook' if stock['signal'] == 'Buy' else 'neutral outlook' if stock['signal'] == 'Hold' else 'negative outlook'}")

    return "\n".join(lines)


def _handle_info(tickers: list[str], stock_map: dict) -> str:
    """Provide basic info about a stock."""
    if not tickers:
        return "Which stock would you like to know about? I cover: " + ", ".join(STOCK_INFO.keys())

    ticker = tickers[0]
    info = STOCK_INFO.get(ticker)
    stock = stock_map.get(ticker)

    if not info:
        return f"I don't have information on {ticker}. Available: {', '.join(STOCK_INFO.keys())}"

    lines = [f"**{ticker} — {info['name']}**\n"]
    lines.append(f"• **Sector:** {info['sector']}")
    lines.append(f"• **Industry:** {info['industry']}")
    lines.append(f"• **Description:** {info['description']}")

    if stock:
        lines.append(f"\n**Current Analysis:**")
        lines.append(f"• Price: ${stock['current_price']:.2f}")
        lines.append(f"• 30-day change: {stock['price_change_30d']:+.2f}%")
        lines.append(f"• Signal: {stock['signal']}")
        lines.append(f"• Suitability: {stock['suitability']['rating']}")

    return "\n".join(lines)


def _handle_best_pick(stock_data: list[dict], risk_category: str) -> str:
    """Suggest the best stock for the user's profile.

    Reached only with a real risk_category (see PERSONALISED_INTENTS guard).
    """
    if not stock_data:
        return ("I don't have any stock analysis loaded yet. Please run the "
                "recommendations on the **Recommendations** page first.")

    suitable = [s for s in stock_data if s["suitability"]["rating"] == "Suitable"]

    if not suitable:
        return (f"Based on your {risk_category} profile, no stocks currently have a 'Suitable' "
                "rating. Consider waiting for better market conditions or reviewing stocks rated "
                "'Use Caution' with careful position sizing.")

    # Rank by: Buy signal first, then lowest volatility
    buy_stocks = [s for s in suitable if s["signal"] == "Buy"]
    if buy_stocks:
        best = min(buy_stocks, key=lambda s: s["volatility"])
        source = "Buy signal stocks"
    else:
        best = min(suitable, key=lambda s: s["volatility"])
        source = "Suitable stocks"

    lines = [f"**My top pick for your {risk_category} profile:**\n"]
    lines.append(f"**{best['ticker']}** ({STOCK_INFO.get(best['ticker'], {}).get('name', '')})")
    lines.append(f"• Signal: {best['signal']}")
    lines.append(f"• Price: ${best['current_price']:.2f}")
    lines.append(f"• Volatility: {best['volatility']:.2f}% (selected for lower risk)")
    lines.append(f"• 30-day change: {best['price_change_30d']:+.2f}%")
    lines.append(f"\n*Why this one?* From {len(suitable)} {source.lower()}, {best['ticker']} "
                 f"offers the best risk/reward balance for your profile with "
                 f"{'strong upward momentum' if best['signal'] == 'Buy' else 'stable outlook'}.")

    return "\n".join(lines)


def _handle_portfolio(stock_data: list[dict], risk_category: str) -> str:
    """Give portfolio-level overview.

    Reached only with a real risk_category (see PERSONALISED_INTENTS guard).
    """
    if not stock_data:
        return ("I don't have any stock analysis loaded yet. Please run the "
                "recommendations on the **Recommendations** page first.")

    suitable = [s for s in stock_data if s["suitability"]["rating"] == "Suitable"]
    caution = [s for s in stock_data if s["suitability"]["rating"] == "Use Caution"]
    not_suitable = [s for s in stock_data if s["suitability"]["rating"] == "Not Suitable"]

    buy_count = sum(1 for s in stock_data if s["signal"] == "Buy")
    hold_count = sum(1 for s in stock_data if s["signal"] == "Hold")
    avoid_count = sum(1 for s in stock_data if s["signal"] == "Avoid")

    lines = [f"**Portfolio Overview for {risk_category} Profile:**\n"]
    lines.append(f"**Market Signals:** {buy_count} Buy, {hold_count} Hold, {avoid_count} Avoid")
    lines.append(f"**For your profile:** {len(suitable)} Suitable, {len(caution)} Use Caution, {len(not_suitable)} Not Suitable\n")

    if suitable:
        lines.append("**Suitable stocks:** " + ", ".join(s["ticker"] for s in suitable))
    if caution:
        lines.append("**Use Caution:** " + ", ".join(s["ticker"] for s in caution))

    avg_vol = sum(s["volatility"] for s in stock_data) / len(stock_data) if stock_data else 0
    lines.append(f"\n**Average volatility:** {avg_vol:.2f}%")
    lines.append(f"**Market sentiment:** {'Bullish' if buy_count > avoid_count else 'Bearish' if avoid_count > buy_count else 'Mixed'} ({buy_count} Buy vs {avoid_count} Avoid)")

    return "\n".join(lines)


def _handle_help() -> str:
    """General help message."""
    return """**I can help you with:**

• **"Tell me about AAPL"** — Stock info and current analysis
• **"Compare AAPL and MSFT"** — Side-by-side comparison
• **"Why is META recommended?"** — Explain a recommendation
• **"Is TSLA risky?"** — Risk assessment
• **"What should I buy?"** — Best pick for your profile
• **"Portfolio overview"** — Summary of all recommendations

Just ask naturally — I understand questions about any of the 10 stocks in our universe."""


# --- Optional LLM presentation layer (Ollama) -----------------------------
#
# Design note: the LLM is deliberately used ONLY to rephrase the factual
# answer produced above. All numbers, signals and ratings originate from the
# Random Forest model and the Suitability Engine. This keeps the advice
# auditable and prevents the language model from hallucinating financial data.

OLLAMA_URL = "http://localhost:11434/api/generate"
# llama3.2:1b is used rather than the 3B variant because inference here runs on
# CPU: the 1B model returns a short summary in a few seconds instead of ~90s,
# which keeps the chat widget responsive.
OLLAMA_MODEL = "llama3.2:1b"

_SYSTEM_PROMPT = (
    "You are a financial advisor assistant. You receive a user question and a "
    "factual answer produced by a machine learning system.\n"
    "Write ONE short paragraph (2-3 sentences) that summarises the key insight "
    "of the factual answer in a natural, advisory tone.\n"
    "RULES:\n"
    "- Only use numbers that appear in the factual answer. Never invent data.\n"
    "- Do not output tables, bullet points, headings or markdown.\n"
    "- Do not add greetings, disclaimers or sign-offs.\n"
    "- Be direct and professional."
)


def is_ollama_available(timeout: float = 1.5) -> bool:
    """Check whether a local Ollama server is reachable."""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=timeout)
        return response.status_code == 200
    except Exception:
        return False


#: Terms the guard checks for mis-attribution against the structured data.
_SIGNAL_TERMS = ("buy", "hold", "avoid")
_RATING_TERMS = ("suitable", "not suitable", "use caution")


def verify_summary(summary: str, stock_data: list[dict]) -> bool:
    """Reject an LLM summary that contradicts the structured model output.

    Small language models occasionally swap attributes between entities — for
    example stating that stock B is "not suitable" when that rating belongs to
    stock A. Because this system gives financial advice, such a summary must
    never reach the user.

    The check works per sentence: if a sentence names exactly one ticker and
    asserts a suitability rating or trading signal, that assertion must match
    the value the Suitability Engine and ML model actually produced.

    Args:
        summary: Candidate LLM summary.
        stock_data: Stock analysis dicts holding the authoritative values.

    Returns:
        True if no contradiction was found, False if the summary must be dropped.
    """
    if not stock_data:
        return True

    facts = {
        s["ticker"]: {
            "rating": s.get("suitability", {}).get("rating", "").lower(),
            "signal": s.get("signal", "").lower(),
        }
        for s in stock_data
    }

    for sentence in re.split(r"(?<=[.!?])\s+", summary):
        named = [t for t in facts if re.search(rf"\b{t}\b", sentence, re.I)]
        # Ambiguous sentences (0 or 2+ tickers) cannot be attributed safely.
        if len(named) != 1:
            continue
        fact = facts[named[0]]
        lowered = sentence.lower()

        # "not suitable" must be tested before "suitable" (substring overlap).
        for term in ("not suitable", "use caution", "suitable"):
            if term in lowered:
                if fact["rating"] and term != fact["rating"]:
                    return False
                break

        for term in _SIGNAL_TERMS:
            if re.search(rf"\b{term}\b", lowered):
                if fact["signal"] and term != fact["signal"]:
                    return False
                break

    return True


def summarise_with_llm(question: str, factual_answer: str,
                       timeout: float = 20.0) -> Optional[str]:
    """Generate a short natural-language summary of a factual answer.

    The summary is intended to be shown *above* the structured answer so the
    user gets a conversational read-out while the underlying tables, figures
    and ratings remain visible and unaltered.

    Args:
        question: The user's original question.
        factual_answer: Deterministic answer built from model output.
        timeout: Seconds to wait for the LLM before giving up.

    Returns:
        A short advisory paragraph, or None if Ollama is unavailable, too slow,
        or produced an unusable generation.
    """
    # Help text and short error messages gain nothing from a summary.
    if len(factual_answer) < 80:
        return None

    prompt = (
        f"Question: {question}\n\n"
        f"Factual answer:\n{factual_answer}\n\n"
        "Write your 2-3 sentence summary now."
    )

    try:
        import requests
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "system": _SYSTEM_PROMPT,
                "stream": False,
                "options": {
                    "temperature": 0.2,
                    "num_predict": 110,   # cap length to keep latency low
                    "top_k": 20,
                },
            },
            timeout=timeout,
        )
        if response.status_code != 200:
            return None

        text = (response.json().get("response") or "").strip()
        # Strip any markdown the model may still emit.
        text = text.replace("**", "").replace("#", "").strip()
        # Reject empty, truncated or suspiciously long generations.
        if not (40 < len(text) < 700):
            return None
        return text
    except Exception:
        return None
