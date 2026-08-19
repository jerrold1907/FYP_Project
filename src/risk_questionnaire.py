"""Scenario-Based Risk Tolerance Questionnaire Module.

Replaces the simple 1-10 slider with a validated, scenario-based assessment
that estimates investor risk tolerance through behavioural questions. The
approach is informed by Grable & Lytton (1999) risk tolerance instrument,
simplified to 5 scenarios covering:
  - Loss reaction (behavioural)
  - Investment goal (objective)
  - Time horizon (capacity)
  - Emotional comfort (psychological)
  - Risk/return tradeoff preference (attitudinal)

Scoring: Each answer maps to 1-4 points. Total (5-20) maps to 5 categories:
  - Conservative (5-8)
  - Moderately Conservative (9-11)
  - Balanced (12-14)
  - Growth (15-17)
  - Aggressive (18-20)
"""

from dataclasses import dataclass, field


# --- Question Definitions ---

QUESTIONS = [
    {
        "id": "loss_reaction",
        "text": (
            "If your portfolio dropped 20% in a month, "
            "what would you most likely do?"
        ),
        "options": [
            ("Sell everything to prevent further losses", 1),
            ("Sell some holdings to reduce risk", 2),
            ("Hold and wait for recovery", 3),
            ("Buy more at the lower prices", 4),
        ],
    },
    {
        "id": "investment_goal",
        "text": "What is your primary investment objective?",
        "options": [
            ("Preserve my capital — I cannot afford to lose money", 1),
            ("Generate steady income with minimal risk", 2),
            ("Balanced growth — willing to accept moderate ups and downs", 3),
            ("Aggressive growth — maximise returns, accept large swings", 4),
        ],
    },
    {
        "id": "time_horizon",
        "text": "When do you expect to need this invested money?",
        "options": [
            ("Within 1 year", 1),
            ("1 to 3 years", 2),
            ("3 to 7 years", 3),
            ("More than 7 years", 4),
        ],
    },
    {
        "id": "emotional_comfort",
        "text": (
            "How would you feel if a single stock in your "
            "portfolio lost 30% of its value?"
        ),
        "options": [
            ("I would lose sleep and want to sell immediately", 1),
            ("Very uncomfortable — I'd seriously consider selling", 2),
            ("Somewhat uncomfortable but I'd hold on", 3),
            ("Fine — losses are part of investing", 4),
        ],
    },
    {
        "id": "risk_return_preference",
        "text": (
            "Which portfolio outcome would you prefer over one year?"
        ),
        "options": [
            ("Gain +3% guaranteed, no chance of loss", 1),
            ("Likely gain +5%, possible loss of -2%", 2),
            ("Likely gain +12%, possible loss of -10%", 3),
            ("Likely gain +25%, possible loss of -20%", 4),
        ],
    },
]


# --- Risk Categories ---

RISK_CATEGORIES = [
    "Conservative",
    "Moderately Conservative",
    "Balanced",
    "Growth",
    "Aggressive",
]

# Score range boundaries: (min_score, max_score) -> category
SCORE_BOUNDARIES = [
    (5, 8, "Conservative"),
    (9, 11, "Moderately Conservative"),
    (12, 14, "Balanced"),
    (15, 17, "Growth"),
    (18, 20, "Aggressive"),
]


@dataclass
class QuestionnaireResponse:
    """Stores a user's responses to the risk questionnaire.

    Attributes:
        answers: Dict mapping question ID to selected score (1-4).
    """
    answers: dict[str, int] = field(default_factory=dict)

    @property
    def total_score(self) -> int:
        """Sum of all answer scores. Range: 5-20 when all answered."""
        return sum(self.answers.values())

    @property
    def is_complete(self) -> bool:
        """True if all 5 questions have been answered."""
        return len(self.answers) == len(QUESTIONS)


def classify_risk_from_score(total_score: int) -> str:
    """Maps a total questionnaire score (5-20) to a risk category.

    Args:
        total_score: Sum of scenario answers (each 1-4, total 5-20).

    Returns:
        One of: 'Conservative', 'Moderately Conservative', 'Balanced',
        'Growth', or 'Aggressive'.

    Raises:
        ValueError: If total_score is outside the valid range [5, 20].
    """
    if total_score < 5 or total_score > 20:
        raise ValueError(
            f"Total score must be between 5 and 20, got {total_score}"
        )

    for min_score, max_score, category in SCORE_BOUNDARIES:
        if min_score <= total_score <= max_score:
            return category

    # Should never reach here given valid input
    raise ValueError(f"No category found for score {total_score}")


def classify_questionnaire(response: QuestionnaireResponse) -> dict:
    """Classifies risk tolerance from a completed questionnaire.

    Args:
        response: A QuestionnaireResponse with all 5 answers.

    Returns:
        Dict with keys:
        - 'category': The risk category string
        - 'total_score': The numeric score (5-20)
        - 'breakdown': Dict of question_id -> score for transparency
        - 'description': Human-readable explanation of the category

    Raises:
        ValueError: If the questionnaire is incomplete.
    """
    if not response.is_complete:
        raise ValueError(
            f"Questionnaire incomplete: {len(response.answers)}/5 "
            f"questions answered"
        )

    total = response.total_score
    category = classify_risk_from_score(total)

    descriptions = {
        "Conservative": (
            "You prefer capital preservation over growth. You are uncomfortable "
            "with market volatility and prioritise safety of principal. "
            "Suitable investments: bonds, blue-chip dividends, money market funds."
        ),
        "Moderately Conservative": (
            "You prefer stability but accept small fluctuations for modest growth. "
            "You lean toward lower-risk investments with some equity exposure. "
            "Suitable investments: balanced funds, dividend stocks, investment-grade bonds."
        ),
        "Balanced": (
            "You accept moderate risk for moderate returns. You're comfortable "
            "with market cycles and have a medium-term outlook. "
            "Suitable investments: diversified index funds, mix of stocks and bonds."
        ),
        "Growth": (
            "You prioritise capital appreciation and tolerate significant volatility. "
            "You have a longer time horizon and can withstand drawdowns. "
            "Suitable investments: growth stocks, sector ETFs, small-cap funds."
        ),
        "Aggressive": (
            "You seek maximum returns and are comfortable with large portfolio swings. "
            "You have a long time horizon and high emotional resilience to losses. "
            "Suitable investments: high-growth tech stocks, leveraged positions, "
            "emerging markets."
        ),
    }

    return {
        "category": category,
        "total_score": total,
        "breakdown": dict(response.answers),
        "description": descriptions[category],
    }
