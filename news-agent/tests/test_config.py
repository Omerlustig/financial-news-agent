import pytest
from pathlib import Path

from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    DirectionalStance,
    TimeHorizon,
    CausalChain,
    HardGates,
)
from src.config import load_settings, load_sources, Settings, SourceConfig


def test_raw_article_hash():
    a1 = RawArticle(
        title="Chip export curbs",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Test",
    )
    a2 = RawArticle(
        title="Chip export curbs",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Different summary",
    )
    # Deduplication hash is based on url + title
    assert a1.content_hash == a2.content_hash

    # Different url or title must yield different hash
    a3 = RawArticle(
        title="Different headline",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Test",
    )
    assert a1.content_hash != a3.content_hash


def test_market_impact_analysis_model():
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA", "$TSM"],
        affected_sectors=["Semiconductors"],
        summary="US imposes stricter curbs on GPU exports to key foreign markets.",
        catalyst_thesis="Direct revenue restriction on high-margin datacenter sales.",
    )
    assert analysis.is_market_moving is True
    assert analysis.confidence_score == 9
    assert analysis.impact_grade == ImpactGrade.CRITICAL
    assert "$NVDA" in analysis.affected_tickers

    # Defaults check for institutional specification
    assert analysis.status == EventStatus.OPPORTUNITY_CANDIDATE
    assert analysis.directional_stance == DirectionalStance.NEUTRAL
    assert analysis.time_horizon is None
    assert analysis.causal_chain is None
    assert analysis.hard_gates is None
    assert "News-derived hypothesis" in analysis.disclaimer

    # Confidence score must be 1 to 10
    with pytest.raises(Exception):
        MarketImpactAnalysis(
            is_market_moving=True,
            confidence_score=11,  # Invalid: > 10
            impact_grade=ImpactGrade.CRITICAL,
            affected_tickers=[],
            affected_sectors=[],
            summary="Invalid",
            catalyst_thesis="Invalid",
        )


def test_load_default_sources(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text("""
sources:
  - name: "Reuters Business"
    type: "rss"
    url: "https://feeds.reuters.com/reuters/businessNews"
    enabled: true
""")
    sources = load_sources(str(sources_yaml))
    assert len(sources) == 1
    assert sources[0].name == "Reuters Business"
    assert sources[0].type == "rss"
    assert sources[0].enabled is True


def test_load_actual_sources_yaml():
    # Verify the real config/sources.yaml contains Section 4 US + Israel feeds
    sources = load_sources()
    assert len(sources) >= 12
    names = [s.name.lower() for s in sources]

    # US Core
    assert any("reuters" in name for name in names)
    assert any("cnbc" in name for name in names)
    assert any("yahoo" in name for name in names)
    assert any("financial times" in name for name in names)
    assert any("bloomberg" in name for name in names)

    # US Primary & Specialist
    assert any("sec" in name for name in names)
    assert any("federal reserve" in name for name in names)
    assert any("eia" in name for name in names)

    # Israel Core
    assert any("globes" in name for name in names)
    assert any("calcalist" in name for name in names)
    assert any("themarker" in name for name in names)
    assert any("tase" in name for name in names)
    assert any("maya" in name for name in names)


def test_load_actual_settings_yaml():
    settings = load_settings()
    assert settings.agent.polling_interval_minutes > 0
    assert settings.filter.min_confidence_score >= 1
    assert isinstance(settings.filter.min_impact_grade, ImpactGrade)
    assert settings.filter.min_impact_grade == ImpactGrade.HIGH
    assert settings.llm.model
    # Unset telegram env vars should resolve to None rather than empty string
    assert settings.telegram.bot_token is None or isinstance(settings.telegram.bot_token, str)


def test_affected_tickers_normalization():
    # String with comma-separated tickers
    analysis1 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers="NVDA, AAPL, $msft",  # string input
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis1.affected_tickers == ["$NVDA", "$AAPL", "$MSFT"]

    # Single string ticker
    analysis2 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers="tsm",  # single string
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis2.affected_tickers == ["$TSM"]

    # List input
    analysis3 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["nvda", "$aapl", "AMZN"],
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis3.affected_tickers == ["$NVDA", "$AAPL", "$AMZN"]


def test_min_impact_grade_validation(tmp_path):
    from pydantic import ValidationError

    # Valid string or enum parses to ImpactGrade
    settings = Settings.model_validate({
        "filter": {"min_impact_grade": "CRITICAL"}
    })
    assert settings.filter.min_impact_grade == ImpactGrade.CRITICAL

    # MEDIUM alias maps to AVERAGE in Settings and FilterSettings
    settings_medium = Settings.model_validate({
        "filter": {"min_impact_grade": "MEDIUM"}
    })
    assert settings_medium.filter.min_impact_grade == ImpactGrade.AVERAGE

    settings_medium_lower = Settings.model_validate({
        "filter": {"min_impact_grade": "medium"}
    })
    assert settings_medium_lower.filter.min_impact_grade == ImpactGrade.AVERAGE

    # Direct enum call converts MEDIUM
    assert ImpactGrade("MEDIUM") == ImpactGrade.AVERAGE
    assert ImpactGrade("medium") == ImpactGrade.AVERAGE

    # MarketImpactAnalysis converts MEDIUM alias to AVERAGE
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=5,
        impact_grade="MEDIUM",
        affected_tickers=["$AAPL"],
        affected_sectors=["Tech"],
        summary="Medium impact event.",
        catalyst_thesis="Average market catalyst.",
    )
    assert analysis.impact_grade == ImpactGrade.AVERAGE
    assert analysis.impact_grade is ImpactGrade.AVERAGE

    # Invalid impact grade raises ValidationError
    with pytest.raises(ValidationError):
        Settings.model_validate({
            "filter": {"min_impact_grade": "INVALID_GRADE"}
        })


def test_telegram_empty_env_to_none():
    # If env var is unset in yaml expansion, it should evaluate to None
    settings = Settings.model_validate({
        "telegram": {"bot_token": "", "chat_id": ""}
    })
    assert settings.telegram.bot_token is None
    assert settings.telegram.chat_id is None


def test_event_status_enum():
    assert EventStatus.OPPORTUNITY_CANDIDATE == "OPPORTUNITY_CANDIDATE"
    assert EventStatus.WATCHLIST == "WATCHLIST"
    assert EventStatus.INFORMATIONAL == "INFORMATIONAL"
    assert EventStatus.UNVERIFIED == "UNVERIFIED"
    assert EventStatus.NO_ACTION == "NO_ACTION"
    assert EventStatus("opportunity_candidate") == EventStatus.OPPORTUNITY_CANDIDATE
    assert str(EventStatus.WATCHLIST) == "WATCHLIST"


def test_directional_stance_enum():
    assert DirectionalStance.BULLISH == "BULLISH"
    assert DirectionalStance.BEARISH == "BEARISH"
    assert DirectionalStance.VOLATILE == "VOLATILE"
    assert DirectionalStance.NEUTRAL == "NEUTRAL"
    assert DirectionalStance("bullish") == DirectionalStance.BULLISH
    assert str(DirectionalStance.BEARISH) == "BEARISH"


def test_time_horizon_enum():
    assert TimeHorizon.INTRADAY == "INTRADAY"
    assert TimeHorizon.SHORT_TERM == "SHORT_TERM"
    assert TimeHorizon.MEDIUM_TERM == "MEDIUM_TERM"
    assert TimeHorizon.LONG_TERM == "LONG_TERM"
    assert TimeHorizon("1-7 days") == TimeHorizon.SHORT_TERM
    assert TimeHorizon("2-8 weeks") == TimeHorizon.MEDIUM_TERM
    assert TimeHorizon("months+") == TimeHorizon.LONG_TERM
    assert str(TimeHorizon.SHORT_TERM) == "SHORT_TERM"


def test_causal_chain_model_and_normalization():
    chain = CausalChain(
        event_facts="FDA approves breakthrough oncology treatment.",
        economic_mechanism="Opens immediate $4B total addressable market.",
        affected_sectors="Biotechnology, Healthcare",
        beneficiaries="bmy, $mrk",
        negatively_exposed=["gild", "$pfe"],
        counterarguments="Payer reimbursement timeline remains uncertain.",
        invalidation_conditions="Phase 4 post-market study safety warning.",
    )
    assert chain.event_facts.startswith("FDA approves")
    assert chain.economic_mechanism.startswith("Opens immediate")
    assert chain.affected_sectors == ["Biotechnology", "Healthcare"]
    assert chain.beneficiaries == ["$BMY", "$MRK"]
    assert chain.negatively_exposed == ["$GILD", "$PFE"]
    assert "Payer reimbursement" in chain.counterarguments
    assert "Phase 4" in chain.invalidation_conditions


def test_hard_gates_model_and_computation():
    # Explicit passes_all_gates
    gates_explicit = HardGates(
        is_novel=True,
        is_verified=True,
        has_direct_exposure=True,
        passes_all_gates=True,
    )
    assert gates_explicit.passes_all_gates is True

    # Auto-computed passes_all_gates when all true
    gates_computed_true = HardGates(
        is_novel=True,
        is_verified=True,
        has_direct_exposure=True,
    )
    assert gates_computed_true.passes_all_gates is True

    # Auto-computed passes_all_gates when one is false
    gates_computed_false = HardGates(
        is_novel=True,
        is_verified=False,
        has_direct_exposure=True,
    )
    assert gates_computed_false.passes_all_gates is False


def test_market_impact_analysis_institutional_full():
    chain = CausalChain(
        event_facts="Bank of Israel raises benchmark rate by 50bps.",
        economic_mechanism="Expands net interest margins for tier-1 Israeli lenders.",
        affected_sectors=["Banking", "Financial Services"],
        beneficiaries=["$LUMI", "$POLI"],
        negatively_exposed=["$BLDG"],
        counterarguments="High mortgage rates may suppress loan origination volumes.",
        invalidation_conditions="Emergency rate cut or macro liquidity injection.",
    )
    gates = HardGates(
        is_novel=True,
        is_verified=True,
        has_direct_exposure=True,
        passes_all_gates=True,
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$LUMI", "$POLI"],
        affected_sectors=["Banking"],
        summary="Bank of Israel unexpectedly increased benchmark rate.",
        catalyst_thesis="Broad lending margin expansion for Israeli commercial banks.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        time_horizon=TimeHorizon.SHORT_TERM,
        causal_chain=chain,
        hard_gates=gates,
    )
    assert analysis.status == EventStatus.OPPORTUNITY_CANDIDATE
    assert analysis.directional_stance == DirectionalStance.BULLISH
    assert analysis.time_horizon == "SHORT_TERM"
    assert analysis.causal_chain is not None
    assert analysis.causal_chain.beneficiaries == ["$LUMI", "$POLI"]
    assert analysis.hard_gates is not None
    assert analysis.hard_gates.passes_all_gates is True
    assert "News-derived hypothesis" in analysis.disclaimer
