import json
from datetime import datetime, timezone
import pytest

from src.models import (
    MarketImpactAnalysis,
    ImpactGrade,
    DirectionalStance,
    TimeHorizon,
    CausalChain,
    HardGates,
    RawArticle,
)
from src.analysis.debate import DialecticDebateEngine, DebateVerdict, DebateConfig


def make_dummy_analysis(confidence: int = 8, grade: ImpactGrade = ImpactGrade.HIGH) -> MarketImpactAnalysis:
    return MarketImpactAnalysis(
        impact_grade=grade,
        confidence_score=confidence,
        is_market_moving=True,
        summary="Brief headline summary for testing.",
        affected_tickers=["$NVDA"],
        catalyst_thesis="Nvidia unveils Rubin platform with 3x compute efficiency, capturing AI server capex.",
        transmission_mechanism="Higher gross margins and unassailable moat in datacenter revenue.",
        directional_stance=DirectionalStance.BULLISH,
        time_horizon=TimeHorizon.MEDIUM_TERM,
        actionable_recommendation="Buy on pullbacks towards 50-DMA.",
        causal_chain=CausalChain(
            event_facts="Nvidia keynote in Taipei announced Rubin GPU line.",
            economic_mechanism="Accelerated upgrade cycle for hyperscalers.",
            beneficiaries=["$NVDA"],
            negatively_exposed=["$AMD"],
        ),
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=True,
        ),
    )


def test_should_debate_trigger_conditions() -> None:
    engine = DialecticDebateEngine(config=DebateConfig(min_confidence_to_debate=8))
    
    # High confidence & high impact -> Should debate
    assert engine.should_debate(make_dummy_analysis(confidence=8, grade=ImpactGrade.HIGH)) is True
    # Average impact but high confidence -> Should debate
    assert engine.should_debate(make_dummy_analysis(confidence=9, grade=ImpactGrade.AVERAGE)) is True
    # Low confidence & low impact -> Should NOT debate
    assert engine.should_debate(make_dummy_analysis(confidence=5, grade=ImpactGrade.LOW)) is False


def test_conduct_debate_synthesizes_bull_and_bear_cases() -> None:
    def mock_llm_debate(prompt: str) -> str:
        return json.dumps({
            "bull_thesis": "Strong hyperscaler capex visibility guarantees order book expansion through 2027.",
            "bear_counter_thesis": "RSI at 82 and 24% premium over 200 DMA indicates news is already priced in; high risk of near-term multiple compression.",
            "net_conviction_score": 7,
            "priced_in_risk_level": "HIGH",
            "invalidation_triggers": [
                "Hyperscaler capex guidance cuts in upcoming quarterly reports",
                "Close below 50 DMA ($126.80)",
            ],
        })

    engine = DialecticDebateEngine(llm_callable=mock_llm_debate)
    analysis = make_dummy_analysis(confidence=8, grade=ImpactGrade.HIGH)
    article = RawArticle(
        title="Nvidia unveils Rubin platform at Computex",
        url="https://example.com/nvda",
        source_name="Reuters",
        published_at=datetime.now(timezone.utc),
    )

    verdict = engine.conduct_debate(analysis, article, market_context="NVDA RSI: 82.0 | Dist 200DMA: +24%")
    assert verdict is not None
    assert verdict.net_conviction_score == 7
    assert verdict.priced_in_risk_level == "HIGH"
    assert "priced in" in verdict.bear_counter_thesis.lower()
    assert len(verdict.invalidation_triggers) == 2


def test_conduct_debate_handles_llm_failure_gracefully() -> None:
    def failing_llm(prompt: str) -> str:
        raise RuntimeError("LLM rate limit or connection drop")

    engine = DialecticDebateEngine(llm_callable=failing_llm)
    analysis = make_dummy_analysis(confidence=8, grade=ImpactGrade.HIGH)
    article = RawArticle(
        title="Nvidia unveils Rubin platform",
        url="https://example.com/nvda",
        source_name="Reuters",
        published_at=datetime.now(timezone.utc),
    )
    # Should not raise exception, but return fallback verdict
    verdict = engine.conduct_debate(analysis, article)
    assert verdict.net_conviction_score == analysis.confidence_score
    assert verdict.priced_in_risk_level == "MEDIUM"
