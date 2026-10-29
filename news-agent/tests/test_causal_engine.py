import pytest
from datetime import datetime, timezone
from unittest.mock import MagicMock

from src.models import (
    CatalystStage,
    TimeHorizon,
    MarketExchange,
    CausalTransmissionAnalysis,
    MarketImpactAnalysis,
    RawArticle,
)
from src.analysis.prompts import filter_valid_tickers
from src.analysis.triage import is_economically_relevant
from src.analysis.analyzer import MarketImpactAnalyzer


def test_causal_transmission_model_serialization():
    causal = CausalTransmissionAnalysis(
        catalyst_stage=CatalystStage.PREDICTIVE_ANTICIPATORY,
        trigger_event="Foreign airlines suspend Tel Aviv routes",
        transmission_mechanism="Removes 40% seat capacity; grants monopoly pricing power to flag carrier",
        time_horizon=TimeHorizon.TACTICAL_1_3_DAYS,
        market_exchange=MarketExchange.TASE,
        predicted_beneficiaries=["$ELAL"],
        unpriced_edge_thesis="Anticipatory window before traffic report discounts yield surge",
    )
    data = causal.model_dump()
    assert data["catalyst_stage"] == "PREDICTIVE_ANTICIPATORY"
    assert data["market_exchange"] == "TASE"
    assert "$ELAL" in data["predicted_beneficiaries"]


def test_tase_ticker_filtering_support():
    tickers = ["$ELAL", "$ESLT", "$LUMI", "ICL", "$NVDA", "$USA"]
    valid = filter_valid_tickers(tickers)
    assert "$ELAL" in valid
    assert "$ESLT" in valid
    assert "$LUMI" in valid
    assert "$ICL" in valid
    assert "$NVDA" in valid
    assert "$USA" not in valid  # Sovereign entity dropped


def test_triage_filter_allows_aviation_and_defense_shocks():
    article = RawArticle(
        title="Flight diverted after security alert over airspace",
        summary="Foreign carriers are suspending flights to Tel Aviv amid regional tensions.",
        url="https://ynetnews.com/article/1",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is True


def test_triage_filter_allows_hebrew_aviation_disruption():
    article = RawArticle(
        title="חברות תעופה זרות מבטלות טיסות לנתבג בעקבות המצב הביטחוני",
        summary="אל על נערכת לתגבר טיסות לנוסעים שנתקעו באירופה.",
        url="https://ynet.co.il/article/2",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is True


def test_triage_filter_drops_celebrity_and_sports_noise():
    article = RawArticle(
        title="Celebrity wedding draws thousands of fans in Tel Aviv",
        summary="Actor married singer in private ceremony with star guests.",
        url="https://ynetnews.com/article/3",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is False


def test_triage_always_allows_financial_sources():
    article = RawArticle(
        title="General market summary",
        summary="Market wrap for the day.",
        url="https://reuters.com/business/1",
        published_at=datetime.now(timezone.utc),
        source_name="Reuters Live Wire",
    )
    assert is_economically_relevant(article) is True


def test_analyzer_parses_causal_transmission():
    from unittest.mock import patch
    from datetime import datetime, timezone
    from src.analysis.analyzer import MarketImpactAnalyzer
    from src.models import RawArticle

    mock_json = '''{
        "is_market_moving": true,
        "status": "OPPORTUNITY_CANDIDATE",
        "confidence_score": 9,
        "impact_grade": "CRITICAL",
        "affected_tickers": ["$ELAL"],
        "beneficiaries": ["$ELAL"],
        "negatively_exposed": [],
        "affected_sectors": ["Airlines", "Commercial Aviation"],
        "summary": "Foreign airlines cancel Tel Aviv flights following security alert.",
        "catalyst_thesis": "40% seat capacity cut grants domestic flag carrier monopoly pricing power.",
        "recommendation": "Prioritize tactical long exposure on $ELAL before flight suspension capacity squeeze is priced in",
        "causal_transmission": {
            "catalyst_stage": "PREDICTIVE_ANTICIPATORY",
            "trigger_event": "Foreign airlines halt flights to Israel",
            "transmission_mechanism": "Eliminates international seat capacity; yields surge",
            "time_horizon": "TACTICAL_1_3_DAYS",
            "market_exchange": "TASE",
            "predicted_beneficiaries": ["$ELAL"],
            "unpriced_edge_thesis": "Traffic surge not yet reflected in consensus earnings"
        }
    }'''
    analyzer = MarketImpactAnalyzer(max_retries=1, retry_delay=0.01)
    article = RawArticle(
        title="Foreign carriers cancel Tel Aviv flights",
        summary="Major airlines suspend operations following airspace warning.",
        url="https://example.com/flights",
        published_at=datetime.now(timezone.utc),
        source_name="Test Source",
    )
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": mock_json}}]
        }
        analysis = analyzer.analyze(article)

    assert analysis.causal_transmission is not None
    assert analysis.causal_transmission.catalyst_stage == CatalystStage.PREDICTIVE_ANTICIPATORY
    assert analysis.causal_transmission.market_exchange == MarketExchange.TASE
    assert "$ELAL" in analysis.causal_transmission.predicted_beneficiaries
