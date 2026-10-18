import pytest
from src.analysis.sentiment_4d import (
    FourDimensionalSentimentAnalyzer,
    ACCRETIVE_KEYWORDS,
    DESTRUCTIVE_KEYWORDS,
    BEAT_KEYWORDS,
    MISS_KEYWORDS,
    IN_LINE_KEYWORDS,
)
from src.models import FourDimensionalSentiment


@pytest.fixture
def analyzer():
    return FourDimensionalSentimentAnalyzer()


def test_fundamental_impact_accretive(analyzer):
    accretive_phrases = [
        "record free cash flow",
        "margin expansion",
        "cost savings",
        "raises guidance",
        "dividend hike",
        "accretive",
        "secures contract",
    ]
    for phrase in accretive_phrases:
        result = analyzer.analyze_4d_sentiment(
            title=f"Company reports {phrase} in quarterly release",
            summary=f"Strong balance sheet driven by {phrase}.",
        )
        assert isinstance(result, FourDimensionalSentiment)
        assert result.fundamental_impact >= 0.5, f"Failed on phrase: {phrase}"
        assert result.fundamental_impact <= 1.0


def test_fundamental_impact_destructive(analyzer):
    destructive_phrases = [
        "margin compression",
        "terminates deal",
        "slashes guidance",
        "restatement",
        "impairment",
        "fined",
        "supply chain disruption",
    ]
    for phrase in destructive_phrases:
        result = analyzer.analyze_4d_sentiment(
            title=f"Company suffers {phrase} amid difficulties",
            summary=f"Significant balance sheet hit from {phrase}.",
        )
        assert isinstance(result, FourDimensionalSentiment)
        assert result.fundamental_impact <= -0.5, f"Failed on phrase: {phrase}"
        assert result.fundamental_impact >= -1.0


def test_fundamental_impact_neutral_when_no_cash_flow_mention(analyzer):
    result = analyzer.analyze_4d_sentiment(
        title="CEO to participate in technology industry panel discussion",
        summary="Executive will share high-level vision at annual summit.",
    )
    assert result.fundamental_impact == 0.0


def test_expectation_surprise_beat(analyzer):
    # Beat that is not an extreme shock (>= 0.35 and < 0.8)
    result_beat = analyzer.analyze_4d_sentiment(
        title="Company beats expectations with strong revenue growth",
        summary="Results beat consensus across all segments.",
    )
    assert result_beat.expectation_surprise >= 0.6
    assert result_beat.surprise_verdict == "BEAT"

    # Extreme shock beat (>= 0.8)
    result_shock = analyzer.analyze_4d_sentiment(
        title="Unexpected surge in cloud demand surprises to upside",
        summary="Blowout quarter with massive beat.",
    )
    assert result_shock.expectation_surprise >= 0.8
    assert result_shock.surprise_verdict == "UNEXPECTED_SHOCK"


def test_expectation_surprise_miss(analyzer):
    # Normal miss (-0.6 to -0.7)
    result_miss = analyzer.analyze_4d_sentiment(
        title="Company misses expectations in Q2",
        summary="Revenue below consensus as enterprise demand softens.",
    )
    assert result_miss.expectation_surprise <= -0.6
    assert result_miss.surprise_verdict == "MISS"

    # Extreme shock miss (<= -0.8)
    result_shock_miss = analyzer.analyze_4d_sentiment(
        title="Company cuts guidance unexpectedly and issues profit warning",
        summary="Severe deterioration triggers abrupt profit warning.",
    )
    assert result_shock_miss.expectation_surprise <= -0.8
    assert result_shock_miss.surprise_verdict == "UNEXPECTED_SHOCK"


def test_expectation_surprise_in_line(analyzer):
    in_line_phrases = [
        "meets expectations",
        "in line with consensus",
        "as expected",
    ]
    for phrase in in_line_phrases:
        result = analyzer.analyze_4d_sentiment(
            title=f"Quarterly earnings {phrase}",
            summary=f"Financial results were {phrase} across the board.",
        )
        assert result.expectation_surprise == 0.0, f"Failed on: {phrase}"
        assert result.surprise_verdict == "IN_LINE"


def test_price_implication_formula(analyzer):
    # Formula: round(0.50 * fundamental + 0.40 * expectation + 0.10 * linguistic, 2)
    fund = 0.80
    exp = 0.70
    ling = 0.60
    expected = round(0.50 * fund + 0.40 * exp + 0.10 * ling, 2)
    assert expected == 0.74

    calc = analyzer._compute_price_implication(fund, exp, ling)
    assert calc == 0.74

    # Test clamping
    assert analyzer._compute_price_implication(1.5, 1.5, 1.5) == 1.0
    assert analyzer._compute_price_implication(-1.5, -1.5, -1.5) == -1.0


def test_surprise_verdict_thresholds(analyzer):
    assert analyzer._compute_verdict(0.85) == "UNEXPECTED_SHOCK"
    assert analyzer._compute_verdict(-0.85) == "UNEXPECTED_SHOCK"
    assert analyzer._compute_verdict(0.80) == "UNEXPECTED_SHOCK"
    assert analyzer._compute_verdict(-0.80) == "UNEXPECTED_SHOCK"
    assert analyzer._compute_verdict(0.70) == "BEAT"
    assert analyzer._compute_verdict(0.35) == "BEAT"
    assert analyzer._compute_verdict(-0.70) == "MISS"
    assert analyzer._compute_verdict(-0.35) == "MISS"
    assert analyzer._compute_verdict(0.20) == "IN_LINE"
    assert analyzer._compute_verdict(-0.20) == "IN_LINE"
    assert analyzer._compute_verdict(0.0) == "IN_LINE"


def test_llm_payload_parsing(analyzer):
    payload = {
        "linguistic_sentiment": 0.75,
        "fundamental_impact": 0.85,
        "expectation_surprise": 0.90,
        "price_implication": 0.86,
        "surprise_verdict": "UNEXPECTED_SHOCK",
    }
    result = analyzer.analyze_4d_sentiment(
        title="Headline",
        llm_payload=payload,
    )
    assert result.linguistic_sentiment == 0.75
    assert result.fundamental_impact == 0.85
    assert result.expectation_surprise == 0.90
    assert result.price_implication == 0.86
    assert result.surprise_verdict == "UNEXPECTED_SHOCK"


def test_llm_payload_nested_and_clamping(analyzer):
    payload = {
        "four_d_sentiment": {
            "linguistic_sentiment": 1.5,
            "fundamental_impact": -2.0,
            "expectation_surprise": 0.5,
        }
    }
    result = analyzer.analyze_4d_sentiment(
        title="Headline",
        llm_payload=payload,
    )
    assert result.linguistic_sentiment == 1.0  # Clamped to 1.0
    assert result.fundamental_impact == -1.0  # Clamped to -1.0
    assert result.expectation_surprise == 0.5
    assert result.surprise_verdict == "BEAT"


def test_hebrew_catalyst_decomposition(analyzer):
    # Accretive Hebrew
    result_accretive = analyzer.analyze_4d_sentiment(
        title="החברה מדווחת על רווח שיא והעלאת תחזיות שנתית",
        summary="דירקטוריון אישר חלוקת דיבידנד מוגדלת",
    )
    assert result_accretive.fundamental_impact >= 0.5

    # Destructive Hebrew
    result_destructive = analyzer.analyze_4d_sentiment(
        title="חברת הנדל\"ן מפרסמת אזהרת רווח חמורה והורדת תחזיות",
        summary="שחיקה ברווחיות בעקבות עליות ריבית",
    )
    assert result_destructive.fundamental_impact <= -0.5
