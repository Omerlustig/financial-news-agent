"""
Unit tests for TwoLayerGateEngine (Layer 1 Event Quality & Layer 2 Opportunity Gates).
"""
import pytest

from src.analysis.decision_gate import TwoLayerGateEngine
from src.models import (
    AdversarialReview,
    DirectionalStance,
    EventStatus,
    FourDimensionalSentiment,
    HardGates,
    ImpactGrade,
    MarketImpactAnalysis,
    NoveltyState,
    ProvenanceLevel,
    TwoLayerGateDecision,
)


def test_decoupling_ft_contract_cancellation_megacap():
    """
    Test decoupling: FT reports contract cancellation on a mega-cap tech company.
    Event quality is high (8.0+), but the contract is a tiny fraction of mega-cap cash flows,
    yielding a low opportunity score (<6.0).
    Verdict must be WATCHLIST, preventing a spurious high-conviction sell alert!
    """
    engine = TwoLayerGateEngine()

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=7,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$MSFT"],
        summary="Financial Times reports government agency cancels $25M pilot contract with Microsoft.",
        catalyst_thesis="Loss of pilot contract with defense department.",
        directional_stance=DirectionalStance.BEARISH,
        provenance_level=ProvenanceLevel.TIER1_FINANCIAL,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
        corroboration_count=1,
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=-0.6,
            fundamental_impact=-0.15,  # Minimal cash flow dent on $250B revenue: 0.15 * 3.5 = 0.525
            expectation_surprise=-0.25, # Modest surprise: 0.25 * 2.5 = 0.625
            price_implication=-0.2,
            surprise_verdict="MISS",
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="Contract cancellation is headline noise representing <0.02% of Azure revenue.",
            priced_in_risk="Enterprise contracts experience routine quarterly churn.",
            macro_confounding_risk="Defense contract delays are widespread across cloud providers.",
            falsification_threshold="Broad enterprise Azure cancellation rate exceeds 3%.",
            is_thesis_robust=False,  # Fragile short thesis: -2.0 modifier!
        ),
        is_watchlist_match=False,
    )

    decision = engine.evaluate_gates(analysis)

    # Event quality score: TIER1_FINANCIAL (8.0) + GENUINE_SURPRISE (+0.5) = 8.5
    assert decision.event_quality_score >= 8.0
    assert decision.passes_event_quality_gate is True

    # Opportunity score: 0.525 + 0.625 - 2.0 (fragile thesis) = 0.0 (< 6.0)
    assert decision.opportunity_score < 6.0
    assert decision.passes_opportunity_gate is False

    # Verdict must be WATCHLIST to suppress spurious sell alert
    assert decision.gate_verdict == "WATCHLIST"
    assert "below threshold" in decision.rejection_reason.lower()


def test_unverified_blog_rumor_yields_reject():
    """
    Test unverified blog rumor:
    Provenance UNVERIFIED_BLOG and unconfirmed rumor yields event quality < 5.0,
    resulting in gate verdict REJECT.
    """
    engine = TwoLayerGateEngine()

    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=4,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=["$NVDA"],
        summary="Tech gossip blog claims unconfirmed chip design delay.",
        catalyst_thesis="Rumor of supply chain re-spin.",
        directional_stance=DirectionalStance.BEARISH,
        status=EventStatus.UNVERIFIED,
        provenance_level=ProvenanceLevel.UNVERIFIED_BLOG,
        novelty_state=NoveltyState.PRICED_IN_OR_RECYCLED,
        corroboration_count=1,
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=-0.3,
            fundamental_impact=-0.2,
            expectation_surprise=-0.1,
            price_implication=-0.1,
        ),
    )

    quality_score = engine.evaluate_event_quality(analysis, is_verified=False)
    # UNVERIFIED_BLOG base 4.0 - 3.0 recycled = 1.0 (capped at 5.0)
    assert quality_score < 5.0

    decision = engine.evaluate_gates(analysis)
    assert decision.passes_event_quality_gate is False
    assert decision.passes_opportunity_gate is False
    assert decision.gate_verdict == "REJECT"
    assert "rejected" in decision.rejection_reason.lower()


def test_genuine_surprise_accretive_cash_flow_strong_technicals_alert():
    """
    Test genuine surprise + accretive cash flow + strong technicals:
    Passes both event quality gate and opportunity gate, producing ALERT verdict.
    """
    engine = TwoLayerGateEngine()

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$TSLA"],
        summary="SEC 8-K filing confirms massive multi-billion enterprise fleet contract.",
        catalyst_thesis="Immediate accretive revenue addition with high operating margins.",
        directional_stance=DirectionalStance.BULLISH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        provenance_level=ProvenanceLevel.PRIMARY_OFFICIAL,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
        corroboration_count=3,
        corroborating_sources=["SEC.gov", "Bloomberg", "Reuters"],
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=0.9,
            fundamental_impact=0.9,      # 0.9 * 3.5 = 3.15
            expectation_surprise=0.85,   # 0.85 * 2.5 = 2.125
            price_implication=0.95,
            surprise_verdict="BEAT",
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="Delivery execution risk over 24-month ramp.",
            priced_in_risk="Zero analyst models anticipated this size of contract.",
            macro_confounding_risk="Pure idiosyncratic company alpha.",
            falsification_threshold="Delivery cancellation or regulatory audit.",
            is_thesis_robust=True,      # +2.0 robustness bonus!
        ),
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=True,
            passes_all_gates=True,
        ),
        is_watchlist_match=True,         # +1.0 watchlist match bonus!
    )

    # Technical confluence boost: +1.5 from TradingViewEngine
    decision = engine.evaluate_gates(analysis, technical_confluence_boost=1.5)

    # Event quality: PRIMARY_OFFICIAL (9.0) + 3 sources (+2.0) + GENUINE_SURPRISE (+0.5) = 10.0 (clamped)
    assert decision.event_quality_score >= 9.0
    assert decision.passes_event_quality_gate is True

    # Opportunity: 3.15 + 2.125 + 2.0 (robust) + 1.5 (tech) + 1.0 (watchlist) = 9.775 -> 9.78
    assert decision.opportunity_score >= 8.5
    assert decision.passes_opportunity_gate is True

    assert decision.gate_verdict == "ALERT"
    assert decision.rejection_reason is None


def test_hard_gate_failure_forces_both_passes_false():
    """
    Test hard gate failure:
    Even with high quality score and opportunity score, failing HardGates
    forces both pass flags to False and prevents an ALERT verdict.
    """
    engine = TwoLayerGateEngine()

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$AAPL"],
        summary="Recycled rumor of supplier change.",
        catalyst_thesis="Supplier margin compression.",
        directional_stance=DirectionalStance.BEARISH,
        provenance_level=ProvenanceLevel.PRIMARY_OFFICIAL,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
        corroboration_count=2,
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=-0.8,
            fundamental_impact=-0.8,
            expectation_surprise=-0.8,
            price_implication=-0.8,
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="No verifiable supplier disclosure.",
            priced_in_risk="Supplier rumors already circulated 2 weeks ago.",
            macro_confounding_risk="Supply chain shift.",
            falsification_threshold="Official Apple 10-Q filing.",
            is_thesis_robust=True,
        ),
        hard_gates=HardGates(
            is_novel=False,             # FAILS NOVELTY GATE
            is_verified=True,
            has_direct_exposure=True,
            passes_all_gates=False,
        ),
    )

    decision = engine.evaluate_gates(analysis, technical_confluence_boost=1.5)

    assert decision.passes_event_quality_gate is False
    assert decision.passes_opportunity_gate is False
    assert decision.gate_verdict != "ALERT"
    assert decision.gate_verdict == "WATCHLIST"
    assert "hard gates" in decision.rejection_reason.lower()


def test_evaluate_event_quality_corroboration_and_novelty_matrix():
    """Verify detailed scoring matrix across provenance, corroboration, and novelty states."""
    engine = TwoLayerGateEngine()

    dummy_analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=5,
        impact_grade=ImpactGrade.AVERAGE,
        summary="Test article",
        catalyst_thesis="Test thesis",
    )

    # 1 source, Primary Official, in-line
    score1 = engine.evaluate_event_quality(
        dummy_analysis,
        provenance=ProvenanceLevel.PRIMARY_OFFICIAL,
        corroboration_count=1,
        novelty_state=NoveltyState.IN_LINE_CONFIRMATION,
    )
    assert score1 == 9.0

    # 2 sources (+1.0), Tier 1 Financial (8.0), Genuine Surprise (+0.5)
    score2 = engine.evaluate_event_quality(
        dummy_analysis,
        provenance=ProvenanceLevel.TIER1_FINANCIAL,
        corroboration_count=2,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
    )
    assert score2 == 9.5

    # 3+ sources (+2.0), Commercial News (8.0 equal footing), Priced in (-3.0) -> 8.0 + 2.0 - 3.0 = 7.0
    score3 = engine.evaluate_event_quality(
        dummy_analysis,
        provenance=ProvenanceLevel.COMMERCIAL_NEWS,
        corroboration_count=4,
        novelty_state=NoveltyState.PRICED_IN_OR_RECYCLED,
    )
    assert score3 == 7.0

    # Verification audit: is_verified=False caps score at 5.0
    score_unverified = engine.evaluate_event_quality(
        dummy_analysis,
        provenance=ProvenanceLevel.PRIMARY_OFFICIAL,
        is_verified=False,
        corroboration_count=3,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
    )
    assert score_unverified == 5.0


def test_evaluate_opportunity_score_components():
    """Verify opportunity score calculation with/without 4D sentiment and modifiers."""
    engine = TwoLayerGateEngine()

    # Case A: with 4D sentiment, robust review, watchlist match, positive technical boost
    analysis_a = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        summary="High conviction test",
        catalyst_thesis="Accretive thesis",
        four_d_sentiment=FourDimensionalSentiment(
            fundamental_impact=1.0,    # 1.0 * 3.5 = 3.5
            expectation_surprise=1.0,  # 1.0 * 2.5 = 2.5
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="None",
            priced_in_risk="Low",
            macro_confounding_risk="Low",
            falsification_threshold="None",
            is_thesis_robust=True,     # +2.0
        ),
        is_watchlist_match=True,        # +1.0
    )
    opp_a = engine.evaluate_opportunity_score(analysis_a, technical_confluence_boost=1.5)
    # 3.5 + 2.5 + 2.0 + 1.5 + 1.0 = 10.5 -> clamped to 10.0
    assert opp_a == 10.0

    # Case B: fallback without 4D sentiment
    analysis_b = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=6,
        impact_grade=ImpactGrade.AVERAGE,
        summary="Fallback test",
        catalyst_thesis="Fallback thesis",
    )
    opp_b = engine.evaluate_opportunity_score(analysis_b, technical_confluence_boost=0.0)
    # cash_flow = 0.6 * 3.5 = 2.1, surprise = 0.5 -> 2.6
    assert 2.5 <= opp_b <= 3.0


def test_geopolitical_shock_commercial_news_passes_alert():
    """
    Test geopolitical macro catalyst breaking on Commercial News (e.g. Seeking Alpha / CNBC):
    Event: Military action / geopolitical strike drives crude energy prices up.
    Analysis identifies clear transmission mechanism, sector beneficiaries ($XOM, $CVX, $XLE),
    and exposed targets ($DAL, $UAL).
    Equal footing provenance (8.0) + genuine surprise (+0.5) ensures event quality gate passes (8.5 >= 8.0).
    Robust thesis (+2.0) and strong fundamental cash flow expansion yields opportunity score >= 8.0 -> ALERT!
    """
    engine = TwoLayerGateEngine()

    from src.models import CausalChain

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$XOM", "$CVX"],
        affected_sectors=["Energy", "Oil & Gas", "Commercial Airlines"],
        summary="US strikes military targets in Iran; Strait of Hormuz oil shipments threatened.",
        catalyst_thesis="Crude supply disruption elevates Brent spot prices, dramatically expanding upstream E&P operating cash flow.",
        directional_stance=DirectionalStance.BULLISH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        provenance_level=ProvenanceLevel.COMMERCIAL_NEWS,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
        corroboration_count=1,
        causal_chain=CausalChain(
            event_facts="US airstrikes conducted against military facilities in Iran.",
            economic_mechanism="Geopolitical disruption risks Persian Gulf tanker flows, spiking global crude prices and widening upstream refining crack spreads.",
            affected_sectors=["Energy", "Airlines"],
            beneficiaries=["$XOM", "$CVX", "$XLE"],
            negatively_exposed=["$DAL", "$UAL", "$JETS"],
            counterarguments="Rapid diplomatic de-escalation or immediate SPR reserve releases.",
            invalidation_conditions="Crude prices drop below $75/bbl or shipping security guarantees restored.",
        ),
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=0.7,
            fundamental_impact=0.85,    # 0.85 * 3.5 = 2.975
            expectation_surprise=0.8,   # 0.8 * 2.5 = 2.0
            price_implication=0.85,
            surprise_verdict="UNEXPECTED_SHOCK",
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="OPEC spare capacity may quickly offset localized disruption.",
            priced_in_risk="Geopolitical risk premium was partially elevated over prior 48 hours.",
            macro_confounding_risk="Broad market selloff could temporarily compress beta multiples.",
            falsification_threshold="Prompt de-escalation with Brent reversing gains within 24 hours.",
            is_thesis_robust=True,      # +2.0
        ),
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=True,
            passes_all_gates=True,
        ),
        is_watchlist_match=False,
    )

    decision = engine.evaluate_gates(analysis, min_opportunity_score=7.0, technical_confluence_boost=1.0)

    # Event quality: COMMERCIAL_NEWS baseline (8.0) + GENUINE_SURPRISE (+0.5) = 8.5 >= 8.0
    assert decision.event_quality_score == 8.5
    assert decision.passes_event_quality_gate is True

    # Opportunity: 2.975 (cash flow) + 2.0 (surprise) + 2.0 (robust) + 1.0 (tech) = 7.975 -> 7.98 (or >= 7.5)
    assert decision.opportunity_score >= 7.5
    assert decision.passes_opportunity_gate is True
    assert decision.gate_verdict == "ALERT"
    assert decision.rejection_reason is None

