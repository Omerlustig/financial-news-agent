"""Financial Sentiment Engine combining Loughran-McDonald financial tone lexicon,
catalyst polarity weights, Hebrew corporate disclosures, and calibrated LLM fusion.
"""

import re
from typing import Dict, List, Optional, Set, Tuple, Union

from src.models import SentimentLabel, SentimentScore


# Strong Bullish (+0.8 to +1.0)
STRONG_BULLISH_KEYWORDS: Dict[str, float] = {
    "record revenue": 0.90,
    "record earnings": 0.90,
    "all-time high": 0.85,
    "all time high": 0.85,
    "guidance raised": 0.90,
    "guidance boost": 0.85,
    "massive beat": 0.90,
    "blowout quarter": 0.95,
    "breakthrough": 0.85,
    "major contract win": 0.85,
    "patent approved": 0.80,
    "patents approved": 0.80,
    "dividend hike": 0.80,
    "share buyback authorization": 0.85,
    # Hebrew Strong Bullish
    "רווח שיא": 0.90,
    "העלאת תחזיות": 0.85,
    "חלוקת דיבידנד": 0.80,
    "רכישה עצמית": 0.80,
}

# Moderate Bullish (+0.3 to +0.6)
MODERATE_BULLISH_KEYWORDS: Dict[str, float] = {
    "beat estimates": 0.50,
    "beats estimates": 0.50,
    "beating estimates": 0.50,
    "earnings beat": 0.50,
    "revenue growth": 0.45,
    "margin expansion": 0.50,
    "upgrade": 0.45,
    "upgraded": 0.45,
    "upgrades": 0.45,
    "rating raised": 0.50,
    "ratings raised": 0.50,
    "overweight": 0.40,
    "order backlog": 0.40,
    "expansion": 0.35,
    "partnership": 0.35,
    "partnerships": 0.35,
    "cost reductions": 0.35,
    "cost reduction": 0.35,
    # Hebrew Moderate Bullish
    "עלייה בהכנסות": 0.50,
    "העלאת דירוג": 0.55,
    "התרחבות": 0.40,
}

# Moderate Bearish (-0.3 to -0.6)
MODERATE_BEARISH_KEYWORDS: Dict[str, float] = {
    "missed estimates": -0.50,
    "misses estimates": -0.50,
    "missing estimates": -0.50,
    "earnings miss": -0.50,
    "revenue decline": -0.45,
    "margin compression": -0.50,
    "downgrade": -0.45,
    "downgraded": -0.45,
    "downgrades": -0.45,
    "underweight": -0.40,
    "guidance cut": -0.55,
    "layoffs": -0.40,
    "headcount reduction": -0.45,
    "headcount reductions": -0.45,
    "cost overruns": -0.45,
    "litigation": -0.40,
    # Hebrew Moderate Bearish
    "הורדת תחזיות": -0.55,
    "הפסד נקי": -0.50,
    "הורדת דירוג": -0.50,
}

# Strong Bearish (-0.8 to -1.0)
STRONG_BEARISH_KEYWORDS: Dict[str, float] = {
    "accounting fraud": -0.95,
    "accounting restatement": -0.95,
    "sec investigation": -0.90,
    "doj probe": -0.90,
    "subpoena": -0.85,
    "default": -0.90,
    "bankruptcy": -0.95,
    "chapter 11": -0.95,
    "covenant breach": -0.85,
    "covenant breached": -0.85,
    "going concern": -0.90,
    "liquidation": -0.90,
    "ceo ousted": -0.85,
    "material weakness": -0.85,
    "profit warning": -0.90,
    # Hebrew Strong Bearish
    "אזהרת רווח": -0.90,
    "אי עמידה בהתניות": -0.85,
    "חקירת רשות ניירות ערך": -0.90,
    "פשיטת רגל": -0.95,
    "חדלות פירעון": -0.95,
}

# Complete financial tone dictionary
DEFAULT_FINANCIAL_LEXICON: Dict[str, float] = {
    **STRONG_BULLISH_KEYWORDS,
    **MODERATE_BULLISH_KEYWORDS,
    **MODERATE_BEARISH_KEYWORDS,
    **STRONG_BEARISH_KEYWORDS,
}

# Extreme triggers that necessitate hard floor override (<= -0.8 VERY_BEARISH)
SEVERE_BEARISH_TRIGGERS: Set[str] = {
    "accounting fraud",
    "accounting restatement",
    "item 4.02",
    "4.02",
    "restatement of financial statements",
    "non-reliance on previously issued financial statements",
    "sec investigation",
    "doj probe",
    "bankruptcy",
    "chapter 11",
    "covenant breach",
    "covenant breached",
    "going concern",
    "liquidation",
    "פשיטת רגל",
    "חדלות פירעון",
    "חקירת רשות ניירות ערך",
}


def score_to_label(score: float) -> SentimentLabel:
    """Assigns discrete categorical SentimentLabel based on calibrated continuous score:
    - score >= 0.6: VERY_BULLISH
    - 0.15 <= score < 0.6: BULLISH
    - -0.15 <= score < 0.15: NEUTRAL
    - -0.6 < score <= -0.15: BEARISH
    - score <= -0.6: VERY_BEARISH
    """
    if score >= 0.6:
        return SentimentLabel.VERY_BULLISH
    elif score >= 0.15:
        return SentimentLabel.BULLISH
    elif score > -0.15:
        return SentimentLabel.NEUTRAL
    elif score > -0.6:
        return SentimentLabel.BEARISH
    else:
        return SentimentLabel.VERY_BEARISH


class FinancialSentimentEngine:
    """Institutional domain-specific financial sentiment analysis engine.

    Combines Loughran-McDonald financial tone scoring and market catalyst polarity
    with calibrated LLM sentiment blending and deterministic regulatory overrides.
    """

    def __init__(self, custom_lexicon: Optional[Dict[str, float]] = None):
        self.lexicon: Dict[str, float] = dict(DEFAULT_FINANCIAL_LEXICON)
        if custom_lexicon:
            self.lexicon.update(custom_lexicon)

        # Precompile regex matchers for performance and word boundary enforcement
        self._compiled_patterns: List[Tuple[re.Pattern, float]] = []
        # Sort keys by length descending to match multi-word expressions first
        sorted_items = sorted(self.lexicon.items(), key=lambda x: len(x[0]), reverse=True)
        for phrase, weight in sorted_items:
            # Check if contains Hebrew characters
            if re.search(r"[\u0590-\u05FF]", phrase):
                pat = re.compile(
                    r"(?:^|[^\w\u0590-\u05fe])" + re.escape(phrase) + r"(?:$|[^\w\u0590-\u05fe])",
                    re.IGNORECASE,
                )
            else:
                pat = re.compile(r"\b" + re.escape(phrase) + r"\b", re.IGNORECASE)
            self._compiled_patterns.append((pat, weight))

    def score_to_label(self, score: float) -> SentimentLabel:
        """Helper delegation to global score_to_label."""
        return score_to_label(score)

    def score_text(self, text: str) -> SentimentScore:
        """Scores financial text polarity using weighted financial tone keywords.

        Returns:
            SentimentScore with compound score [-1.0, 1.0], label, positive/negative/neutral
            probabilities, and confidence score.
        """
        if not text or not text.strip():
            return SentimentScore(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                confidence=0.80,
            )

        matched_weights: List[float] = []
        for pattern, weight in self._compiled_patterns:
            if pattern.search(text):
                matched_weights.append(weight)

        if not matched_weights:
            return SentimentScore(
                score=0.0,
                label=SentimentLabel.NEUTRAL,
                positive=0.0,
                negative=0.0,
                neutral=1.0,
                confidence=0.70,
            )

        pos_weights = [w for w in matched_weights if w > 0]
        neg_weights = [w for w in matched_weights if w < 0]

        pos_score = 0.0
        if pos_weights:
            max_pos = max(pos_weights)
            mean_pos = sum(pos_weights) / len(pos_weights)
            pos_score = 0.7 * max_pos + 0.3 * mean_pos

        neg_score = 0.0
        if neg_weights:
            min_neg = min(neg_weights)
            mean_neg = sum(neg_weights) / len(neg_weights)
            neg_score = 0.7 * min_neg + 0.3 * mean_neg

        raw_score = pos_score + neg_score
        clamped_score = max(-1.0, min(1.0, round(raw_score, 4)))

        # Probabilistic distribution
        if clamped_score > 0:
            positive = round(min(0.95, max(0.10, clamped_score)), 4)
            negative = round(min(0.95, max(0.0, abs(neg_score))), 4) if neg_score else 0.0
            neutral = round(max(0.0, 1.0 - positive - negative), 4)
        elif clamped_score < 0:
            negative = round(min(0.95, max(0.10, abs(clamped_score))), 4)
            positive = round(min(0.95, max(0.0, pos_score)), 4) if pos_score else 0.0
            neutral = round(max(0.0, 1.0 - positive - negative), 4)
        else:
            positive = 0.05
            negative = 0.05
            neutral = 0.90

        confidence = round(
            min(0.95, 0.70 + 0.04 * len(matched_weights) + 0.15 * abs(clamped_score)), 2
        )
        label = self.score_to_label(clamped_score)

        return SentimentScore(
            score=clamped_score,
            label=label,
            positive=positive,
            negative=negative,
            neutral=neutral,
            confidence=confidence,
        )

    def calibrate_sentiment(
        self, text: str, llm_sentiment: Optional[SentimentScore] = None
    ) -> SentimentScore:
        """Blends rule-based domain sentiment with LLM sentiment and enforces deterministic overrides.

        Args:
            text: Headline and summary text to evaluate.
            llm_sentiment: Optional sentiment output parsed from the LLM.

        Returns:
            Calibrated SentimentScore guaranteed within [-1.0, 1.0] and with deterministic
            floor overrides (<= -0.8 VERY_BEARISH) for critical adverse events.
        """
        rule_sentiment = self.score_text(text)

        if llm_sentiment is None:
            blended_score = rule_sentiment.score
            pos = rule_sentiment.positive
            neg = rule_sentiment.negative
            neu = rule_sentiment.neutral
            conf = rule_sentiment.confidence
        else:
            # 50/50 weighting blend between rule-based domain engine and LLM
            blended_score = 0.5 * rule_sentiment.score + 0.5 * llm_sentiment.score
            pos = 0.5 * rule_sentiment.positive + 0.5 * llm_sentiment.positive
            neg = 0.5 * rule_sentiment.negative + 0.5 * llm_sentiment.negative
            neu = 0.5 * rule_sentiment.neutral + 0.5 * llm_sentiment.neutral
            conf = min(1.0, 0.5 * rule_sentiment.confidence + 0.5 * llm_sentiment.confidence)

        # Enforce deterministic extreme override for severe events
        text_lower = text.lower()
        is_severe_bearish = any(trigger in text_lower for trigger in SEVERE_BEARISH_TRIGGERS)

        if is_severe_bearish:
            blended_score = min(blended_score, -0.85)
            neg = max(neg, 0.90)
            pos = min(pos, 0.05)
            neu = max(0.0, round(1.0 - pos - neg, 4))
            conf = max(conf, 0.90)

        final_score = max(-1.0, min(1.0, round(blended_score, 4)))
        label = self.score_to_label(final_score)

        return SentimentScore(
            score=final_score,
            label=label,
            positive=round(pos, 4),
            negative=round(neg, 4),
            neutral=round(neu, 4),
            confidence=round(conf, 4),
        )
