"""
Two-Layer Decision Gate Engine for Market News Agent.
Decouples Event Information Quality Score (evidence rigor, primary provenance)
from Actionable Investment Opportunity Score (asymmetric risk/reward, expectation surprise,
technical confluence alignment).
"""
from __future__ import annotations

import logging
from typing import Optional, Union

from src.models import (
    EventStatus,
    HardGates,
    MarketImpactAnalysis,
    NoveltyState,
    ProvenanceLevel,
    TwoLayerGateDecision,
)

logger = logging.getLogger("decision_gate")


class TwoLayerGateEngine:
    """Evaluates Layer 1 Event Quality and Layer 2 Opportunity scores, applying institutional gates."""

    # Baseline Provenance Scoring
    PROVENANCE_SCORES = {
        ProvenanceLevel.PRIMARY_OFFICIAL: 9.0,
        ProvenanceLevel.TIER1_FINANCIAL: 8.0,
        ProvenanceLevel.COMMERCIAL_NEWS: 6.5,
        ProvenanceLevel.UNVERIFIED_BLOG: 4.0,
    }

    def __init__(
        self,
        default_min_event_quality: float = 8.0,
        default_min_opportunity_score: float = 8.0,
    ) -> None:
        self.default_min_event_quality = default_min_event_quality
        self.default_min_opportunity_score = default_min_opportunity_score

    def evaluate_event_quality(
        self,
        analysis: MarketImpactAnalysis,
        provenance: Optional[Union[ProvenanceLevel, str]] = None,
        is_verified: bool = True,
        corroboration_count: int = 1,
        novelty_state: Optional[Union[NoveltyState, str]] = None,
        enable_fact_verification: bool = True,
    ) -> float:
        """Evaluate Layer 1 Event Information Quality Score (0.0 to 10.0).

        Factors:
        - Source provenance tier (PRIMARY_OFFICIAL=9.0, TIER1_FINANCIAL=8.0, COMMERCIAL_NEWS=6.5, UNVERIFIED_BLOG=4.0)
        - Corroboration boost (+1.0 for 2 sources, +2.0 for 3+ sources)
        - Novelty adjustment (GENUINE_SURPRISE=+0.5, IN_LINE_CONFIRMATION=0.0, PRICED_IN_OR_RECYCLED=-3.0)
        - Verification audit (cap at 5.0 if unverified, anonymous rumor, or unverified blog when enabled)
        """
        # Resolve Provenance
        resolved_prov = provenance or getattr(analysis, "provenance_level", None)
        if resolved_prov is None:
            resolved_prov = ProvenanceLevel.COMMERCIAL_NEWS
        elif isinstance(resolved_prov, str):
            try:
                resolved_prov = ProvenanceLevel(resolved_prov.strip().upper().replace(" ", "_"))
            except ValueError:
                resolved_prov = ProvenanceLevel.COMMERCIAL_NEWS

        baseline = self.PROVENANCE_SCORES.get(resolved_prov, 6.5)

        # Resolve Corroboration
        effective_corroboration = max(
            corroboration_count,
            getattr(analysis, "corroboration_count", 1),
            len(getattr(analysis, "corroborating_sources", [])),
        )
        if effective_corroboration >= 3:
            corrob_boost = 2.0
        elif effective_corroboration == 2:
            corrob_boost = 1.0
        else:
            corrob_boost = 0.0

        # Resolve Novelty
        resolved_novelty = novelty_state or getattr(analysis, "novelty_state", None)
        if resolved_novelty is None:
            resolved_novelty = NoveltyState.IN_LINE_CONFIRMATION
        elif isinstance(resolved_novelty, str):
            try:
                resolved_novelty = NoveltyState(resolved_novelty.strip().upper().replace(" ", "_"))
            except ValueError:
                resolved_novelty = NoveltyState.IN_LINE_CONFIRMATION

        if resolved_novelty == NoveltyState.GENUINE_SURPRISE:
            novelty_adj = 0.5
        elif resolved_novelty == NoveltyState.PRICED_IN_OR_RECYCLED:
            novelty_adj = -3.0
        else:
            novelty_adj = 0.0

        raw_score = baseline + corrob_boost + novelty_adj

        # Verification audit: cap at 5.0 if not verified, unconfirmed rumor, or unverified blog
        if enable_fact_verification:
            has_unconfirmed_rumor = (
                not is_verified
                or resolved_prov == ProvenanceLevel.UNVERIFIED_BLOG
                or getattr(analysis, "status", None) == EventStatus.UNVERIFIED
                or (getattr(analysis, "hard_gates", None) is not None and not analysis.hard_gates.is_verified)
            )
            if has_unconfirmed_rumor:
                raw_score = min(raw_score, 5.0)
        else:
            # When fact verification gate is disabled, only cap outright unverified blogs
            if resolved_prov == ProvenanceLevel.UNVERIFIED_BLOG:
                raw_score = min(raw_score, 5.0)

        quality_score = max(0.0, min(10.0, raw_score))
        return round(quality_score, 2)

    def evaluate_opportunity_score(
        self,
        analysis: MarketImpactAnalysis,
        technical_confluence_boost: float = 0.0,
    ) -> float:
        """Evaluate Layer 2 Actionable Investment Opportunity Score (0.0 to 10.0).

        Factors:
        - Fundamental cash flow impact: abs(fundamental_impact) * 3.5 (or baseline from confidence/grade)
        - Expectation surprise delta: abs(expectation_surprise) * 2.5
        - Adversarial robustness: +2.0 if is_thesis_robust, -2.0 if fragile
        - Technical confluence boost: +technical_confluence_boost (-2.0 to +2.0)
        - Watchlist match boost: +1.0 if is_watchlist_match
        """
        # 1. Fundamental cash flow impact (0.0 to 3.5)
        if analysis.four_d_sentiment is not None:
            cash_flow_score = abs(analysis.four_d_sentiment.fundamental_impact) * 3.5
        else:
            conf_norm = max(1.0, min(10.0, float(analysis.confidence_score))) / 10.0
            cash_flow_score = conf_norm * 3.5

        # 2. Expectation surprise (0.0 to 2.5)
        if analysis.four_d_sentiment is not None:
            surprise_score = abs(analysis.four_d_sentiment.expectation_surprise) * 2.5
        else:
            surprise_score = 1.0 if analysis.confidence_score >= 7 else 0.5

        # 3. Adversarial robustness (+2.0 or -2.0)
        if analysis.adversarial_review is not None:
            robustness_mod = 2.0 if analysis.adversarial_review.is_thesis_robust else -2.0
        else:
            robustness_mod = 0.0

        # 4. Technical confluence boost (-2.0 to +2.0)
        tech_boost = max(-2.0, min(2.0, float(technical_confluence_boost)))

        # 5. Watchlist match boost (+1.0)
        watchlist_boost = 1.0 if getattr(analysis, "is_watchlist_match", False) else 0.0

        raw_opportunity = cash_flow_score + surprise_score + robustness_mod + tech_boost + watchlist_boost
        opportunity_score = max(0.0, min(10.0, raw_opportunity))
        return round(opportunity_score, 2)

    def evaluate_gates(
        self,
        analysis: MarketImpactAnalysis,
        min_event_quality: float = 8.0,
        min_opportunity_score: float = 8.0,
        technical_confluence_boost: float = 0.0,
        enable_fact_verification: bool = True,
    ) -> TwoLayerGateDecision:
        """Evaluate both Layer 1 and Layer 2 gates, producing final decision and institutional verdict."""
        # Check hard gates verification status if present
        is_ver = True
        if enable_fact_verification:
            if analysis.hard_gates is not None and not analysis.hard_gates.is_verified:
                is_ver = False
            if getattr(analysis, "status", None) == EventStatus.UNVERIFIED:
                is_ver = False

        event_quality_score = self.evaluate_event_quality(
            analysis=analysis,
            provenance=getattr(analysis, "provenance_level", None),
            is_verified=is_ver,
            corroboration_count=getattr(analysis, "corroboration_count", 1),
            novelty_state=getattr(analysis, "novelty_state", None),
            enable_fact_verification=enable_fact_verification,
        )

        opportunity_score = self.evaluate_opportunity_score(
            analysis=analysis,
            technical_confluence_boost=technical_confluence_boost,
        )

        passes_event_quality_gate = event_quality_score >= min_event_quality
        passes_opportunity_gate = opportunity_score >= min_opportunity_score

        # Hard gates check: if hard gates failed, force both passes to False
        hard_gates_failed = False
        if analysis.hard_gates is not None:
            if enable_fact_verification and not analysis.hard_gates.passes_all_gates:
                hard_gates_failed = True
            elif not enable_fact_verification and not (analysis.hard_gates.is_novel and analysis.hard_gates.has_direct_exposure):
                hard_gates_failed = True

        if hard_gates_failed:
            passes_event_quality_gate = False
            passes_opportunity_gate = False

        # Gate verdict determination
        if passes_event_quality_gate and passes_opportunity_gate:
            gate_verdict = "ALERT"
            rejection_reason = None
        elif event_quality_score >= 6.0 or opportunity_score >= 6.0:
            gate_verdict = "WATCHLIST"
            if hard_gates_failed:
                rejection_reason = (
                    "Hard gates check failed (verification, novelty, or direct exposure). "
                    "Held in watchlist for monitoring."
                )
            elif passes_event_quality_gate and not passes_opportunity_gate:
                rejection_reason = (
                    f"High event quality ({event_quality_score:.1f}/10) confirmed by primary/tier-1 sources, "
                    f"but actionable opportunity score ({opportunity_score:.1f}/10) below threshold ({min_opportunity_score:.1f}). "
                    f"Held in watchlist to prevent spurious or low-edge alert."
                )
            elif passes_opportunity_gate and not passes_event_quality_gate:
                rejection_reason = (
                    f"High opportunity score ({opportunity_score:.1f}/10), but event quality score "
                    f"({event_quality_score:.1f}/10) below threshold ({min_event_quality:.1f}). "
                    f"Held in watchlist awaiting primary source confirmation."
                )
            else:
                rejection_reason = (
                    f"Sub-threshold scores for immediate alert: Event Quality ({event_quality_score:.1f}/10), "
                    f"Opportunity Score ({opportunity_score:.1f}/10). Logged to watchlist."
                )
        else:
            gate_verdict = "REJECT"
            if hard_gates_failed:
                rejection_reason = (
                    "Rejected: Hard gates check failed (unverified rumor, stale duplicate, or lack of direct exposure)."
                )
            else:
                rejection_reason = (
                    f"Rejected: Insufficient event quality ({event_quality_score:.1f}/10) and low opportunity score "
                    f"({opportunity_score:.1f}/10) fail minimum standards."
                )

        return TwoLayerGateDecision(
            event_quality_score=event_quality_score,
            opportunity_score=opportunity_score,
            passes_event_quality_gate=passes_event_quality_gate,
            passes_opportunity_gate=passes_opportunity_gate,
            gate_verdict=gate_verdict,
            rejection_reason=rejection_reason,
        )
