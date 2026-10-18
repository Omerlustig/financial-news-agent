"""4-Dimensional Sentiment Decomposition Engine.

Disentangles editorial tone (linguistic sentiment) from fundamental cash-flow alteration,
market expectation surprise, and directional price response.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from src.analysis.sentiment import FinancialSentimentEngine
from src.models import FourDimensionalSentiment

logger = logging.getLogger(__name__)

# Direct Cash-Flow Alteration Accretive Keywords (+0.5 to +1.0)
ACCRETIVE_KEYWORDS: Dict[str, float] = {
    "record free cash flow": 0.85,
    "free cash flow record": 0.85,
    "margin expansion": 0.65,
    "expanding margins": 0.65,
    "cost savings": 0.55,
    "cost reduction": 0.55,
    "cost reductions": 0.55,
    "raises guidance": 0.75,
    "raised guidance": 0.75,
    "raising guidance": 0.75,
    "boosts guidance": 0.75,
    "dividend hike": 0.65,
    "hiked dividend": 0.65,
    "dividend increase": 0.65,
    "accretive": 0.65,
    "highly accretive": 0.80,
    "secures contract": 0.65,
    "secured contract": 0.65,
    "securing contract": 0.65,
    "contract win": 0.65,
    "major contract win": 0.80,
    "record revenue": 0.75,
    "record earnings": 0.75,
    "share buyback": 0.60,
    "buyback authorization": 0.65,
    "operating leverage": 0.60,
    "profit surge": 0.70,
    "earnings surge": 0.70,
    "raises outlook": 0.70,
    "cash flow positive": 0.60,
    # Hebrew Accretive Terms
    "רווח שיא": 0.80,
    "העלאת תחזיות": 0.75,
    "חלוקת דיבידנד": 0.65,
    "רכישה עצמית": 0.60,
    "זכייה בחוזה": 0.65,
}

# Direct Cash-Flow Alteration Destructive Keywords (-0.5 to -1.0)
DESTRUCTIVE_KEYWORDS: Dict[str, float] = {
    "margin compression": -0.65,
    "compressing margins": -0.65,
    "terminates deal": -0.75,
    "terminated deal": -0.75,
    "terminating deal": -0.75,
    "contract termination": -0.75,
    "deal termination": -0.75,
    "slashes guidance": -0.85,
    "slashed guidance": -0.85,
    "slashing guidance": -0.85,
    "cuts guidance": -0.70,
    "cut guidance": -0.70,
    "cutting guidance": -0.70,
    "restatement": -0.90,
    "accounting restatement": -0.95,
    "restates financials": -0.90,
    "impairment": -0.70,
    "impairments": -0.70,
    "goodwill impairment": -0.80,
    "fined": -0.60,
    "penalized": -0.60,
    "hefty fine": -0.75,
    "supply chain disruption": -0.65,
    "supply chain disruptions": -0.65,
    "supply bottleneck": -0.60,
    "profit warning": -0.85,
    "accounting fraud": -0.95,
    "default": -0.85,
    "debt default": -0.90,
    "bankruptcy": -0.95,
    "chapter 11": -0.95,
    "covenant breach": -0.85,
    "covenant breached": -0.85,
    "material weakness": -0.85,
    "layoffs": -0.50,
    "revenue decline": -0.55,
    # Hebrew Destructive Terms
    "אזהרת רווח": -0.85,
    "הורדת תחזיות": -0.75,
    "הפסד נקי": -0.65,
    "פשיטת רגל": -0.95,
    "ביטול הסכם": -0.75,
    "שחיקה ברווחיות": -0.65,
}

# Deviation from Consensus Expectations — Beat Keywords (+0.6 to +1.0)
BEAT_KEYWORDS: Dict[str, float] = {
    "surprises to upside": 0.85,
    "surprised to upside": 0.85,
    "surprises to the upside": 0.85,
    "unexpected surge": 0.85,
    "beats expectations": 0.70,
    "beat expectations": 0.70,
    "beating expectations": 0.70,
    "beats consensus": 0.70,
    "beat consensus": 0.70,
    "beats estimates": 0.70,
    "beat estimates": 0.70,
    "raises outlook": 0.70,
    "raised outlook": 0.70,
    "raising outlook": 0.70,
    "above consensus": 0.70,
    "tops estimates": 0.70,
    "topped estimates": 0.70,
    "earnings beat": 0.65,
    "blowout quarter": 0.85,
    "massive beat": 0.85,
    "substantial beat": 0.80,
}

# Deviation from Consensus Expectations — Miss Keywords (-0.6 to -1.0)
MISS_KEYWORDS: Dict[str, float] = {
    "cuts guidance unexpectedly": -0.85,
    "cut guidance unexpectedly": -0.85,
    "unexpectedly cuts guidance": -0.85,
    "profit warning": -0.85,
    "misses expectations": -0.70,
    "missed expectations": -0.70,
    "missing expectations": -0.70,
    "below consensus": -0.70,
    "misses consensus": -0.70,
    "missed consensus": -0.70,
    "misses estimates": -0.70,
    "missed estimates": -0.70,
    "earnings miss": -0.65,
    "cuts outlook": -0.70,
    "cut outlook": -0.70,
    "lowers outlook unexpectedly": -0.85,
    "massive miss": -0.85,
    "substantial miss": -0.80,
    "disappoints": -0.65,
    "disappointing results": -0.65,
}

# Deviation from Consensus Expectations — In-Line Keywords (0.0)
IN_LINE_KEYWORDS: List[str] = [
    "meets expectations",
    "met expectations",
    "in line with expectations",
    "in-line with expectations",
    "in line with consensus",
    "in-line with consensus",
    "meets consensus",
    "met consensus",
    "as expected",
    "as anticipated",
    "in line",
    "in-line",
]


class FourDimensionalSentimentAnalyzer:
    """Institutional 4-dimensional sentiment decomposition engine.

    Deconstructs catalyst news into:
    1. linguistic_sentiment: Editorial lexical valence (-1.0 to 1.0)
    2. fundamental_impact: Direct cash-flow alteration (-1.0 to 1.0)
    3. expectation_surprise: Deviation from consensus expectations (-1.0 to 1.0)
    4. price_implication: Directional market price response (-1.0 to 1.0)
    5. surprise_verdict: Categorical expectation verdict (BEAT, MISS, IN_LINE, UNEXPECTED_SHOCK)
    """

    def __init__(self, sentiment_engine: Optional[FinancialSentimentEngine] = None):
        self.sentiment_engine = sentiment_engine or FinancialSentimentEngine()

    def analyze_4d_sentiment(
        self,
        title: str,
        summary: str = "",
        content: str = "",
        llm_payload: Optional[Dict[str, Any]] = None,
    ) -> FourDimensionalSentiment:
        """Analyzes text across the 4 sentiment and expectation dimensions.

        If `llm_payload` is provided and contains valid 4D fields, validates, normalizes,
        and returns them. Otherwise, computes deterministically from text.
        """
        if llm_payload is not None:
            parsed = self._try_parse_llm_payload(llm_payload)
            if parsed is not None:
                return parsed

        return self._compute_deterministic(title, summary, content)

    def _try_parse_llm_payload(
        self, payload: Dict[str, Any]
    ) -> Optional[FourDimensionalSentiment]:
        """Attempts to parse and validate 4D sentiment from LLM payload."""
        raw_4d = payload.get("four_d_sentiment", payload)
        if not isinstance(raw_4d, dict):
            return None

        # Check if at least linguistic_sentiment, fundamental_impact, or expectation_surprise are present
        keys = {"linguistic_sentiment", "fundamental_impact", "expectation_surprise"}
        if not any(k in raw_4d for k in keys):
            return None

        try:
            ling = float(raw_4d.get("linguistic_sentiment", 0.0))
            fund = float(raw_4d.get("fundamental_impact", 0.0))
            exp = float(raw_4d.get("expectation_surprise", 0.0))

            # Clamp input components
            ling = max(-1.0, min(1.0, ling))
            fund = max(-1.0, min(1.0, fund))
            exp = max(-1.0, min(1.0, exp))

            if "price_implication" in raw_4d and raw_4d["price_implication"] is not None:
                price_imp = max(-1.0, min(1.0, round(float(raw_4d["price_implication"]), 2)))
            else:
                price_imp = self._compute_price_implication(fund, exp, ling)

            verdict_raw = raw_4d.get("surprise_verdict")
            if verdict_raw and str(verdict_raw).strip().upper() in (
                "UNEXPECTED_SHOCK",
                "BEAT",
                "MISS",
                "IN_LINE",
            ):
                verdict = str(verdict_raw).strip().upper()
            else:
                verdict = self._compute_verdict(exp)

            return FourDimensionalSentiment(
                linguistic_sentiment=round(ling, 4),
                fundamental_impact=round(fund, 4),
                expectation_surprise=round(exp, 4),
                price_implication=price_imp,
                surprise_verdict=verdict,
            )
        except (ValueError, TypeError) as e:
            logger.warning(
                "Error parsing llm_payload for FourDimensionalSentiment: %s. Using deterministic fallback.",
                e,
            )
            return None

    def _compute_deterministic(
        self, title: str, summary: str = "", content: str = ""
    ) -> FourDimensionalSentiment:
        """Deterministically evaluates 4D sentiment from text."""
        combined_text = f"{title} {summary} {content}".strip().lower()

        # 1. Linguistic Sentiment using FinancialSentimentEngine
        linguistic_score = self.sentiment_engine.score_text(f"{title} {summary}").score
        linguistic_sentiment = round(max(-1.0, min(1.0, linguistic_score)), 4)

        # 2. Fundamental Impact: Direct cash-flow alteration
        fundamental_impact = self._compute_fundamental_impact(combined_text)

        # 3. Expectation Surprise: Deviation from consensus
        expectation_surprise = self._compute_expectation_surprise(combined_text)

        # 4. Surprise Verdict
        verdict = self._compute_verdict(expectation_surprise)

        # 5. Price Implication Formula
        # Formula: round(0.50 * fundamental_impact + 0.40 * expectation_surprise + 0.10 * linguistic_sentiment, 2)
        price_implication = self._compute_price_implication(
            fundamental_impact, expectation_surprise, linguistic_sentiment
        )

        return FourDimensionalSentiment(
            linguistic_sentiment=linguistic_sentiment,
            fundamental_impact=round(fundamental_impact, 4),
            expectation_surprise=round(expectation_surprise, 4),
            price_implication=price_implication,
            surprise_verdict=verdict,
        )

    def _compute_fundamental_impact(self, text: str) -> float:
        """Calculates direct cash-flow alteration (-1.0 to 1.0)."""
        pos_matches: List[float] = [
            weight for phrase, weight in ACCRETIVE_KEYWORDS.items() if phrase in text
        ]
        neg_matches: List[float] = [
            weight for phrase, weight in DESTRUCTIVE_KEYWORDS.items() if phrase in text
        ]

        if not pos_matches and not neg_matches:
            return 0.0

        if pos_matches and not neg_matches:
            # If multiple positive matches, scale towards upper bound
            base = max(pos_matches)
            boost = 0.05 * (len(pos_matches) - 1)
            return min(1.0, base + boost)

        if neg_matches and not pos_matches:
            # If multiple negative matches, scale towards lower bound
            base = min(neg_matches)
            drop = 0.05 * (len(neg_matches) - 1)
            return max(-1.0, base - drop)

        # Both present: net balance
        net = sum(pos_matches) + sum(neg_matches)
        return max(-1.0, min(1.0, net))

    def _compute_expectation_surprise(self, text: str) -> float:
        """Calculates deviation from consensus expectations (-1.0 to 1.0)."""
        # In-line check first if explicitly stated as expected / in line
        is_in_line = any(phrase in text for phrase in IN_LINE_KEYWORDS)

        beat_matches: List[float] = [
            weight for phrase, weight in BEAT_KEYWORDS.items() if phrase in text
        ]
        miss_matches: List[float] = [
            weight for phrase, weight in MISS_KEYWORDS.items() if phrase in text
        ]

        if is_in_line and not beat_matches and not miss_matches:
            return 0.0

        if beat_matches and not miss_matches:
            # Use the most extreme beat score
            return max(beat_matches)

        if miss_matches and not beat_matches:
            # Use the most extreme miss score
            return min(miss_matches)

        if beat_matches and miss_matches:
            # Mixed surprise
            net = sum(beat_matches) + sum(miss_matches)
            return max(-1.0, min(1.0, net))

        # Default neutral if no expectation keywords matched
        return 0.0

    @staticmethod
    def _compute_verdict(expectation_surprise: float) -> str:
        """Assigns expectation surprise verdict based on calibrated thresholds.

        - "UNEXPECTED_SHOCK" if abs(expectation_surprise) >= 0.8
        - "BEAT" if expectation_surprise >= 0.35
        - "MISS" if expectation_surprise <= -0.35
        - "IN_LINE" otherwise
        """
        if abs(expectation_surprise) >= 0.8:
            return "UNEXPECTED_SHOCK"
        if expectation_surprise >= 0.35:
            return "BEAT"
        if expectation_surprise <= -0.35:
            return "MISS"
        return "IN_LINE"

    @staticmethod
    def _compute_price_implication(
        fundamental_impact: float, expectation_surprise: float, linguistic_sentiment: float
    ) -> float:
        """Calculates expected directional price response.

        Formula: round(0.50 * fundamental_impact + 0.40 * expectation_surprise + 0.10 * linguistic_sentiment, 2)
        clamped to [-1.0, 1.0].
        """
        score = (
            0.50 * fundamental_impact
            + 0.40 * expectation_surprise
            + 0.10 * linguistic_sentiment
        )
        return max(-1.0, min(1.0, round(score, 2)))
