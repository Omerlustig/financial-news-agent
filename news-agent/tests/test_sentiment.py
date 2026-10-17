"""Comprehensive test suite for FinancialSentimentEngine and prompt calibration."""

import pytest

from src.models import (
    RawArticle,
    SentimentLabel,
    SentimentScore,
)
from src.analysis.sentiment import (
    FinancialSentimentEngine,
    score_to_label,
)
from src.analysis.prompts import SYSTEM_ANALYST_PROMPT, build_analyst_prompt
from src.analysis.analyzer import MarketImpactAnalyzer


class TestFinancialSentimentEngineScoring:
    """Tests keyword-based scoring across all financial severity tiers."""

    @pytest.fixture
    def engine(self):
        return FinancialSentimentEngine()

    def test_strong_bullish_english_keywords(self, engine):
        samples = [
            "NVIDIA reports record revenue and massive beat in Q3 earnings",
            "Apple raises dividend hike and announces record earnings",
            "Company shares reach all-time high following blowout quarter",
            "Biotech firm announces FDA breakthrough and patent approved",
            "Board authorizes massive share buyback authorization and guidance raised",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert res.score >= 0.6, f"Expected score >= 0.6 for '{text}', got {res.score}"
            assert res.label == SentimentLabel.VERY_BULLISH
            assert res.positive > 0.5
            assert res.negative == 0.0
            assert res.confidence >= 0.7

    def test_moderate_bullish_english_keywords(self, engine):
        samples = [
            "Microsoft beats estimates with solid revenue growth",
            "Analysts issue upgrade to overweight citing margin expansion",
            "Company secures strategic partnership and expands order backlog",
            "Efficiency initiatives drive cost reductions across all divisions",
            "Bank initiates coverage with rating raised",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert 0.15 <= res.score < 0.6, f"Expected 0.15 <= score < 0.6 for '{text}', got {res.score}"
            assert res.label == SentimentLabel.BULLISH
            assert res.positive > res.negative

    def test_neutral_text(self, engine):
        neutral_samples = [
            "Federal Reserve holds FOMC meeting in Washington",
            "Company schedules annual general meeting of shareholders for June",
            "Executive attends industry technology conference in New York",
            "",
            "   ",
        ]
        for text in neutral_samples:
            res = engine.score_text(text)
            assert -0.15 <= res.score < 0.15, f"Expected neutral score for '{text}', got {res.score}"
            assert res.label == SentimentLabel.NEUTRAL
            assert res.neutral >= 0.7

    def test_moderate_bearish_english_keywords(self, engine):
        samples = [
            "Alphabet reports earnings miss amid digital ad revenue decline",
            "Analyst issues downgrade to underweight citing margin compression",
            "Firm announces guidance cut and widespread layoffs",
            "Company faces headcount reduction and severe cost overruns",
            "Supplier entangled in protracted patent litigation",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert -0.6 < res.score <= -0.15, f"Expected -0.6 < score <= -0.15 for '{text}', got {res.score}"
            assert res.label == SentimentLabel.BEARISH
            assert res.negative > res.positive

    def test_strong_bearish_english_keywords(self, engine):
        samples = [
            "SEC launches investigation into suspected accounting fraud",
            "Audit committee discloses accounting restatement of prior financials",
            "Retailer files for Chapter 11 bankruptcy following loan default",
            "Lenders issue notice of covenant breach and demand liquidation",
            "Board issues going concern warning and confirms CEO ousted",
            "Semiconductor firm issues emergency profit warning for Q4",
            "Federal grand jury issues DOJ probe subpoena",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert res.score <= -0.6, f"Expected score <= -0.6 for '{text}', got {res.score}"
            assert res.label == SentimentLabel.VERY_BEARISH
            assert res.negative >= 0.7


class TestHebrewFinancialKeywords:
    """Tests Hebrew market catalyst and regulatory tone lexicon."""

    @pytest.fixture
    def engine(self):
        return FinancialSentimentEngine()

    def test_hebrew_strong_bullish(self, engine):
        samples = [
            "בנק הפועלים מדווח על רווח שיא ברבעון השלישי",
            "החברה מפרסמת העלאת תחזיות שנתית לצד חלוקת דיבידנד מוגדלת",
            "הדירקטוריון אישר תוכנית רכישה עצמית של מניות",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert res.score >= 0.6, f"Expected score >= 0.6 for '{text}', got {res.score}"
            assert res.label == SentimentLabel.VERY_BULLISH

    def test_hebrew_moderate_bullish(self, engine):
        samples = [
            "חברת הנדל\"ן מדווחת על עלייה בהכנסות מהשכרת נכסים",
            "מידרוג הודיעה על העלאת דירוג האג\"ח של הקבוצה",
            "רשת הקמעונאות ממשיכה בתוכנית התרחבות סניפים",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert 0.15 <= res.score < 0.6, f"Expected moderate bullish for '{text}', got {res.score}"
            assert res.label == SentimentLabel.BULLISH

    def test_hebrew_moderate_bearish(self, engine):
        samples = [
            "הנהלת החברה הודיעה על הורדת תחזיות לשנה כולה",
            "החברה רשמה הפסד נקי עקב עליית הוצאות המימון",
            "סוכנות הדירוג הודיעה על הורדת דירוג החוב",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert -0.6 < res.score <= -0.15, f"Expected moderate bearish for '{text}', got {res.score}"
            assert res.label == SentimentLabel.BEARISH

    def test_hebrew_strong_bearish(self, engine):
        samples = [
            "דוח מיידי: אזהרת רווח חמורה והפסד תפעולי צפוי",
            "הבנק הודיע לחברה על אי עמידה בהתניות פיננסיות",
            "נפתחה חקירת רשות ניירות ערך נגד נושאי משרה בכירים",
            "בית המשפט הוציא צו פשיטת רגל ואישר חדלות פירעון",
        ]
        for text in samples:
            res = engine.score_text(text)
            assert res.score <= -0.6, f"Expected strong bearish for '{text}', got {res.score}"
            assert res.label == SentimentLabel.VERY_BEARISH


class TestSentimentCalibrationAndFusion:
    """Tests rule-based and LLM sentiment blending with deterministic overrides."""

    @pytest.fixture
    def engine(self):
        return FinancialSentimentEngine()

    def test_calibration_without_llm_sentiment(self, engine):
        text = "Tesla announces record revenue and massive beat in Q3 deliveries"
        calibrated = engine.calibrate_sentiment(text, llm_sentiment=None)

        assert calibrated.score >= 0.6
        assert calibrated.label == SentimentLabel.VERY_BULLISH
        assert calibrated.positive > 0.5

    def test_calibration_with_llm_sentiment_blending(self, engine):
        text = "Company reports beat estimates on moderate revenue growth"
        rule_score = engine.score_text(text).score  # around 0.45 - 0.50

        llm_sentiment = SentimentScore(
            score=0.70,
            label=SentimentLabel.VERY_BULLISH,
            positive=0.75,
            negative=0.05,
            neutral=0.20,
            confidence=0.85,
        )

        calibrated = engine.calibrate_sentiment(text, llm_sentiment=llm_sentiment)
        expected_score = round(0.5 * rule_score + 0.5 * 0.70, 4)
        assert abs(calibrated.score - expected_score) < 0.02
        assert calibrated.label in (SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH)

    def test_deterministic_extreme_override_accounting_restatement(self, engine):
        """Even if the LLM hallucinated a bullish score, Item 4.02/restatements must be <= -0.8 VERY_BEARISH."""
        text = "Acme Corp Item 4.02 Non-Reliance on Previously Issued Financial Statements and accounting restatement"
        hallucinated_bullish_llm = SentimentScore(
            score=0.85,
            label=SentimentLabel.VERY_BULLISH,
            positive=0.90,
            negative=0.0,
            neutral=0.10,
            confidence=0.95,
        )

        calibrated = engine.calibrate_sentiment(text, llm_sentiment=hallucinated_bullish_llm)
        assert calibrated.score <= -0.80
        assert calibrated.label == SentimentLabel.VERY_BEARISH
        assert calibrated.negative >= 0.85

    def test_deterministic_extreme_override_bankruptcy(self, engine):
        text = "Retail giant files for Chapter 11 bankruptcy and liquidation"
        calibrated = engine.calibrate_sentiment(text, llm_sentiment=None)
        assert calibrated.score <= -0.80
        assert calibrated.label == SentimentLabel.VERY_BEARISH
        assert calibrated.negative >= 0.85

    def test_deterministic_extreme_override_hebrew_insolvency(self, engine):
        text = "החברה הגישה בקשה לפתיחת הליכי חדלות פירעון ופשיטת רגל"
        calibrated = engine.calibrate_sentiment(text, llm_sentiment=None)
        assert calibrated.score <= -0.80
        assert calibrated.label == SentimentLabel.VERY_BEARISH

    def test_boundary_clamping(self, engine):
        """Validates that calibrated scores are strictly clamped to [-1.0, 1.0]."""
        # Test extreme upper bound input
        extreme_bullish = SentimentScore(score=1.0, positive=1.0, negative=0.0, neutral=0.0)
        calibrated_up = engine.calibrate_sentiment(
            "record revenue blowout quarter massive beat all-time high breakthrough",
            llm_sentiment=extreme_bullish,
        )
        assert -1.0 <= calibrated_up.score <= 1.0
        assert 0.0 <= calibrated_up.positive <= 1.0
        assert 0.0 <= calibrated_up.negative <= 1.0
        assert 0.0 <= calibrated_up.neutral <= 1.0
        assert 0.0 <= calibrated_up.confidence <= 1.0

        # Test extreme lower bound input
        extreme_bearish = SentimentScore(score=-1.0, positive=0.0, negative=1.0, neutral=0.0)
        calibrated_down = engine.calibrate_sentiment(
            "accounting fraud bankruptcy liquidation default covenant breach",
            llm_sentiment=extreme_bearish,
        )
        assert -1.0 <= calibrated_down.score <= 1.0
        assert calibrated_down.score <= -0.8


class TestPromptAndAnalyzerIntegration:
    """Tests integration of sentiment engine with prompts and analyzer."""

    def test_prompt_contains_sentiment_instructions_and_schema(self):
        article = RawArticle(
            title="NVIDIA beats estimates on surging AI infrastructure demand",
            url="https://example.com/nvda-beat",
            source_name="Bloomberg",
            summary="Strong datacenter revenue drives quarterly earnings beat.",
        )
        system_prompt, user_prompt = build_analyst_prompt(article)

        assert "FINANCIAL SENTIMENT ANALYSIS" in system_prompt
        assert "Loughran-McDonald" in system_prompt or "LOUGHRAN-MCDONALD" in system_prompt
        assert '"sentiment"' in system_prompt
        assert "VERY_BULLISH" in system_prompt
        assert "VERY_BEARISH" in system_prompt
        assert "Financial Sentiment" in user_prompt

    def test_analyzer_initializes_sentiment_engine(self):
        analyzer = MarketImpactAnalyzer()
        assert hasattr(analyzer, "sentiment_engine")
        assert isinstance(analyzer.sentiment_engine, FinancialSentimentEngine)

    def test_analyzer_fallback_has_valid_calibrated_sentiment(self):
        analyzer = MarketImpactAnalyzer()
        article = RawArticle(
            title="Tech Firm Discloses Accounting Restatement of Audited Results",
            url="https://example.com/restatement",
            source_name="SEC",
            summary="Audit review requires material restatement.",
        )
        fallback = analyzer._build_fallback(article, reason="Timeout")
        assert fallback.sentiment is not None
        assert fallback.sentiment.score <= -0.6
        assert fallback.sentiment.label == SentimentLabel.VERY_BEARISH
