"""Suitability Engine Module.

This module implements the Suitability Engine — the component responsible for
combining investor Risk Category (from the Risk Classifier) and Stock Signal
(from the Stock Recommender) into a final Suitability Rating.

The engine uses a deterministic 3×3 lookup table that maps every valid
(Risk_Category, Stock_Signal) pair to one of three ratings:
  - Suitable: The stock aligns well with the investor's risk profile.
  - Use Caution: The stock may be appropriate but warrants careful consideration.
  - Not Suitable: The stock does not match the investor's risk tolerance.

Each mapping also carries a human-readable explanation describing WHY the
recommendation was made, helping investors understand the reasoning.
"""

from src.validation import validate_risk_category, validate_stock_signal


class SuitabilityEngine:
    """Combines Risk Category and Stock Signal into a Suitability Rating.

    The engine validates inputs, looks up the rating from a predefined mapping,
    and returns both the rating and a human-readable explanation.
    """

    # MAPPING: Dictionary mapping (Risk_Category, Stock_Signal) tuples to
    # Suitability_Rating strings. Covers all 9 valid combinations from the
    # 3x3 matrix of risk categories and stock signals.
    #
    # Design rationale:
    # - Conservative investors are cautioned even on Buy signals and advised
    #   against Hold/Avoid signals, since their low risk tolerance cannot
    #   absorb potential losses.
    # - Moderate investors get full "Suitable" only on strong Buy signals,
    #   "Use Caution" on Hold (neutral outlook), and "Not Suitable" on Avoid.
    # - Aggressive investors can tolerate more risk, so Buy and Hold are both
    #   "Suitable", while Avoid still warrants caution rather than rejection.
    MAPPING: dict[tuple[str, str], str] = {
        # Conservative: very risk-averse, only cautious on strong Buy
        ("Conservative", "Buy"):   "Use Caution",
        ("Conservative", "Hold"):  "Not Suitable",
        ("Conservative", "Avoid"): "Not Suitable",
        # Moderately Conservative: slightly more open but still cautious
        ("Moderately Conservative", "Buy"):   "Use Caution",
        ("Moderately Conservative", "Hold"):  "Not Suitable",
        ("Moderately Conservative", "Avoid"): "Not Suitable",
        # Balanced (also accepts legacy "Moderate"): accepts moderate risk
        ("Balanced", "Buy"):       "Suitable",
        ("Balanced", "Hold"):      "Use Caution",
        ("Balanced", "Avoid"):     "Not Suitable",
        ("Moderate", "Buy"):       "Suitable",
        ("Moderate", "Hold"):      "Use Caution",
        ("Moderate", "Avoid"):     "Not Suitable",
        # Growth: accepts significant risk for returns
        ("Growth", "Buy"):         "Suitable",
        ("Growth", "Hold"):        "Suitable",
        ("Growth", "Avoid"):       "Use Caution",
        # Aggressive: highest risk tolerance
        ("Aggressive", "Buy"):     "Suitable",
        ("Aggressive", "Hold"):    "Suitable",
        ("Aggressive", "Avoid"):   "Use Caution",
    }

    # EXPLANATIONS: Dictionary mapping each (Risk_Category, Stock_Signal) pair
    # to a human-readable explanation string. Each explanation describes WHY the
    # particular suitability rating makes sense for the given combination,
    # referencing both the investor's risk profile characteristics and the
    # stock signal's implications.
    EXPLANATIONS: dict[tuple[str, str], str] = {
        ("Conservative", "Buy"): (
            "Conservative investor + Buy signal → Use Caution: "
            "although the stock shows positive momentum, your conservative risk "
            "profile suggests limiting exposure to high-growth stocks that may "
            "carry significant downside risk."
        ),
        ("Conservative", "Hold"): (
            "Conservative investor + Hold signal → Not Suitable: "
            "a neutral outlook combined with your low risk tolerance means "
            "the potential reward does not justify the uncertainty. Consider "
            "lower-volatility alternatives."
        ),
        ("Conservative", "Avoid"): (
            "Conservative investor + Avoid signal → Not Suitable: "
            "the stock is signaling negative prospects and your conservative "
            "profile cannot absorb potential losses. This investment is not "
            "aligned with your risk tolerance."
        ),
        ("Moderately Conservative", "Buy"): (
            "Moderately Conservative investor + Buy signal → Use Caution: "
            "the stock shows positive indicators, but your preference for "
            "stability means you should consider a smaller position size and "
            "monitor closely for signs of reversal."
        ),
        ("Moderately Conservative", "Hold"): (
            "Moderately Conservative investor + Hold signal → Not Suitable: "
            "a neutral outlook does not provide enough upside to justify the "
            "risk for your conservative-leaning profile. Prefer investments "
            "with clearer positive trajectories."
        ),
        ("Moderately Conservative", "Avoid"): (
            "Moderately Conservative investor + Avoid signal → Not Suitable: "
            "the stock shows negative indicators. Your preference for stability "
            "means this investment carries too much downside risk for your profile."
        ),
        ("Balanced", "Buy"): (
            "Balanced investor + Buy signal → Suitable: "
            "the stock shows strong positive indicators and your balanced risk "
            "profile can accommodate the associated volatility. This is a good "
            "match for your investment goals."
        ),
        ("Balanced", "Hold"): (
            "Balanced investor + Hold signal → Use Caution: "
            "the stock has a neutral outlook. While your moderate risk tolerance "
            "allows for some uncertainty, the lack of strong upside signals "
            "suggests proceeding carefully and monitoring closely."
        ),
        ("Balanced", "Avoid"): (
            "Balanced investor + Avoid signal → Not Suitable: "
            "the stock is showing negative indicators. Even with a balanced risk "
            "tolerance, the downside risk outweighs potential benefits. Consider "
            "alternatives with better prospects."
        ),
        ("Moderate", "Buy"): (
            "Moderate investor + Buy signal → Suitable: "
            "the stock shows strong positive indicators and your balanced risk "
            "profile can accommodate the associated volatility. This is a good "
            "match for your investment goals."
        ),
        ("Moderate", "Hold"): (
            "Moderate investor + Hold signal → Use Caution: "
            "the stock has a neutral outlook. While your moderate risk tolerance "
            "allows for some uncertainty, the lack of strong upside signals "
            "suggests proceeding carefully and monitoring closely."
        ),
        ("Moderate", "Avoid"): (
            "Moderate investor + Avoid signal → Not Suitable: "
            "the stock is showing negative indicators. Even with a moderate risk "
            "tolerance, the downside risk outweighs potential benefits. Consider "
            "alternatives with better prospects."
        ),
        ("Growth", "Buy"): (
            "Growth investor + Buy signal → Suitable: "
            "the stock shows strong positive momentum and your growth-oriented "
            "profile is well-positioned to capitalise on upside potential. "
            "This aligns well with your investment strategy."
        ),
        ("Growth", "Hold"): (
            "Growth investor + Hold signal → Suitable: "
            "although the stock has a neutral outlook, your growth tolerance "
            "allows you to hold through uncertainty and potentially benefit "
            "from future upside movements."
        ),
        ("Growth", "Avoid"): (
            "Growth investor + Avoid signal → Use Caution: "
            "the stock is signaling negative prospects. While your growth "
            "profile can absorb some losses, the negative indicators warrant "
            "careful consideration before committing further capital."
        ),
        ("Aggressive", "Buy"): (
            "Aggressive investor + Buy signal → Suitable: "
            "the stock shows strong positive momentum and your high risk tolerance "
            "is well-suited to capitalize on growth opportunities. This aligns "
            "with an aggressive investment strategy."
        ),
        ("Aggressive", "Hold"): (
            "Aggressive investor + Hold signal → Suitable: "
            "although the stock has a neutral outlook, your high risk tolerance "
            "allows you to hold through uncertainty and potentially benefit from "
            "future upside movements."
        ),
        ("Aggressive", "Avoid"): (
            "Aggressive investor + Avoid signal → Use Caution: "
            "the stock is signaling negative prospects. While your aggressive "
            "profile can absorb some losses, the negative indicators warrant "
            "careful consideration before committing capital."
        ),
    }

    def generate_explanation(self, risk_category: str, stock_signal: str) -> str:
        """Returns the human-readable explanation for a given combination.

        Looks up the explanation string from the EXPLANATIONS dictionary for
        the specified (risk_category, stock_signal) pair.

        Args:
            risk_category: A valid Risk_Category string
                           (Conservative, Moderate, or Aggressive).
            stock_signal: A valid Stock_Signal string (Buy, Hold, or Avoid).

        Returns:
            A human-readable explanation string describing why the suitability
            rating was assigned for this particular combination.

        Raises:
            KeyError: If the combination is not found in EXPLANATIONS
                      (should only occur if called with invalid inputs
                      without prior validation).
        """
        # Direct lookup in the explanations dictionary
        return self.EXPLANATIONS[(risk_category, stock_signal)]

    def recommend(self, risk_category: str, stock_signal: str) -> dict:
        """Produces a suitability recommendation for the given inputs.

        Validates both inputs using the shared validation functions, then
        looks up the rating and generates an explanation for valid combinations.

        Validation flow:
        1. Check if risk_category is None or empty → return missing error
        2. Check if stock_signal is None or empty → return missing error
        3. Validate risk_category against allowed values
        4. Validate stock_signal against allowed values
        5. If all valid, look up rating and generate explanation

        Args:
            risk_category: The investor's risk classification. Must be one of
                           'Conservative', 'Moderate', or 'Aggressive'.
            stock_signal: The stock's predicted action signal. Must be one of
                          'Buy', 'Hold', or 'Avoid'.

        Returns:
            A dictionary with either:
            - On success: {'rating': str, 'explanation': str}
              where 'rating' is the Suitability_Rating and 'explanation' is
              a human-readable string from generate_explanation().
            - On error: {'error': str}
              where 'error' identifies which input was invalid/missing and
              the value that was received.
        """
        # Step 1: Check for None or empty risk_category
        if risk_category is None or risk_category == "":
            # Construct error message identifying the missing input
            return {
                "error": (
                    "Missing Risk_Category: input is required but received "
                    f"{repr(risk_category)}"
                )
            }

        # Step 2: Check for None or empty stock_signal
        if stock_signal is None or stock_signal == "":
            # Construct error message identifying the missing input
            return {
                "error": (
                    "Missing Stock_Signal: input is required but received "
                    f"{repr(stock_signal)}"
                )
            }

        # Step 3: Validate risk_category against allowed values
        risk_error = validate_risk_category(risk_category)
        if risk_error is not None:
            # Return the validation error identifying the invalid input
            return {"error": risk_error}

        # Step 4: Validate stock_signal against allowed values
        signal_error = validate_stock_signal(stock_signal)
        if signal_error is not None:
            # Return the validation error identifying the invalid input
            return {"error": signal_error}

        # Step 5: Both inputs valid — look up rating from the mapping table
        rating = self.MAPPING[(risk_category, stock_signal)]

        # Generate the human-readable explanation for this combination
        explanation = self.generate_explanation(risk_category, stock_signal)

        return {"rating": rating, "explanation": explanation}
