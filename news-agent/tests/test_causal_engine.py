import pytest
from src.models import (
    CatalystStage,
    TimeHorizon,
    MarketExchange,
    CausalTransmissionAnalysis,
    MarketImpactAnalysis,
)
from src.analysis.prompts import filter_valid_tickers

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
    from datetime import datetime, timezone
    from src.models import RawArticle
    from src.analysis.triage import is_economically_relevant

    article = RawArticle(
        title="Flight diverted after security alert over airspace",
        summary="Foreign carriers are suspending flights to Tel Aviv amid regional tensions.",
        url="https://ynetnews.com/article/1",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is True


def test_triage_filter_allows_hebrew_aviation_disruption():
    from datetime import datetime, timezone
    from src.models import RawArticle
    from src.analysis.triage import is_economically_relevant

    article = RawArticle(
        title="חברות תעופה זרות מבטלות טיסות לנתבג בעקבות המצב הביטחוני",
        summary="אל על נערכת לתגבר טיסות לנוסעים שנתקעו באירופה.",
        url="https://ynet.co.il/article/2",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is True


def test_triage_filter_drops_celebrity_and_sports_noise():
    from datetime import datetime, timezone
    from src.models import RawArticle
    from src.analysis.triage import is_economically_relevant

    article = RawArticle(
        title="Celebrity wedding draws thousands of fans in Tel Aviv",
        summary="Actor married singer in private ceremony with star guests.",
        url="https://ynetnews.com/article/3",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is False


def test_triage_always_allows_financial_sources():
    from datetime import datetime, timezone
    from src.models import RawArticle
    from src.analysis.triage import is_economically_relevant

    article = RawArticle(
        title="General market summary",
        summary="Market wrap for the day.",
        url="https://reuters.com/business/1",
        published_at=datetime.now(timezone.utc),
        source_name="Reuters Live Wire",
    )
    assert is_economically_relevant(article) is True

