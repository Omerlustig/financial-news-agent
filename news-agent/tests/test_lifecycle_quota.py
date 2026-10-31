import time
import pytest

from src.models import (
    ImpactGrade,
    DirectionalStance,
    MarketImpactAnalysis,
    TimeHorizon,
    CausalChain,
    HardGates,
)
from src.analysis.lifecycle import (
    SignalEvolutionState,
    SignalLifecycleEngine,
    TickerQuotaTracker,
    TickerQuotaConfig,
)


def make_analysis(
    stance: DirectionalStance = DirectionalStance.BULLISH,
    confidence: int = 7,
    grade: ImpactGrade = ImpactGrade.HIGH,
) -> MarketImpactAnalysis:
    return MarketImpactAnalysis(
        impact_grade=grade,
        confidence_score=confidence,
        is_market_moving=True,
        summary="Summary test",
        affected_tickers=["$NVDA"],
        catalyst_thesis="Sample thesis",
        directional_stance=stance,
        time_horizon=TimeHorizon.MEDIUM_TERM,
        actionable_recommendation="Actionable",
        causal_chain=CausalChain(
            event_facts="Facts",
            economic_mechanism="Mechanism",
            beneficiaries=["$NVDA"],
            negatively_exposed=[],
        ),
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )


def test_quota_tracker_throttles_after_limit() -> None:
    tracker = TickerQuotaTracker(config=TickerQuotaConfig(max_alerts_per_ticker=2, rolling_window_hours=1.0))
    
    # First 2 alerts should be allowed
    allowed1, reason1 = tracker.check_quota("NVDA", ImpactGrade.HIGH)
    assert allowed1 is True
    assert reason1 == "QUOTA_OK"
    tracker.record_alert("NVDA", ImpactGrade.HIGH)

    allowed2, reason2 = tracker.check_quota("NVDA", ImpactGrade.HIGH)
    assert allowed2 is True
    tracker.record_alert("NVDA", ImpactGrade.HIGH)

    # 3rd alert for NVDA within window should be throttled
    allowed3, reason3 = tracker.check_quota("NVDA", ImpactGrade.HIGH)
    assert allowed3 is False
    assert reason3 == "TICKER_QUOTA_EXCEEDED"

    # Different ticker should not be throttled
    allowed_msft, _ = tracker.check_quota("MSFT", ImpactGrade.HIGH)
    assert allowed_msft is True


def test_quota_tracker_bypasses_for_critical_impact() -> None:
    tracker = TickerQuotaTracker(config=TickerQuotaConfig(max_alerts_per_ticker=1, rolling_window_hours=1.0))
    tracker.record_alert("AAPL", ImpactGrade.HIGH)

    # Normal alert blocked
    allowed_high, _ = tracker.check_quota("AAPL", ImpactGrade.HIGH)
    assert allowed_high is False

    # CRITICAL impact bypasses quota
    allowed_crit, reason_crit = tracker.check_quota("AAPL", ImpactGrade.CRITICAL)
    assert allowed_crit is True
    assert reason_crit == "CRITICAL_BYPASS"


def test_signal_lifecycle_evolution_states() -> None:
    engine = SignalLifecycleEngine()

    initial = make_analysis(stance=DirectionalStance.BULLISH, confidence=7)
    # No prior analysis -> NEW
    assert engine.evaluate_evolution(initial, prior=None) == SignalEvolutionState.NEW

    # Matching subsequent bullish thesis with higher conviction -> STRENGTHENED
    reinforced = make_analysis(stance=DirectionalStance.BULLISH, confidence=9)
    assert engine.evaluate_evolution(reinforced, prior=initial) == SignalEvolutionState.STRENGTHENED

    # Reversal to bearish -> FALSIFIED
    reversal = make_analysis(stance=DirectionalStance.BEARISH, confidence=8)
    assert engine.evaluate_evolution(reversal, prior=initial) == SignalEvolutionState.FALSIFIED
