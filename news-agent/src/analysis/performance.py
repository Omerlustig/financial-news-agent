import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from src.models import (
    CredibilityTier,
    EventLineageRecord,
    PerformanceMetricsSummary,
    SourceCredibilityRecord,
)
from src.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class PerformanceTrackerEngine:
    """Empirical forward return tracking, directional alpha calculations,
    hit rate scoring, and publisher credibility evaluation.
    """

    def __init__(self, friction_threshold_pct: float = 0.20) -> None:
        self.friction_threshold_pct = friction_threshold_pct

    def compute_directional_alpha(
        self,
        stance: Optional[str],
        initial_price: Optional[float],
        current_price: Optional[float],
    ) -> Tuple[float, float]:
        """Compute raw percentage return and directional alpha.

        Alpha convention:
        - BULLISH: Alpha = +Raw Return
        - BEARISH: Alpha = -Raw Return (falling stock is a positive win!)
        - VOLATILE / NEUTRAL: Alpha = |Raw Return|
        """
        if initial_price is None or initial_price <= 0 or current_price is None or current_price <= 0:
            return 0.0, 0.0

        raw_return = ((current_price - initial_price) / initial_price) * 100.0
        stance_norm = str(stance or "NEUTRAL").strip().upper()

        if stance_norm == "BULLISH":
            alpha = raw_return
        elif stance_norm == "BEARISH":
            alpha = -raw_return
        else:
            alpha = abs(raw_return)

        return round(raw_return, 4), round(alpha, 4)

    def classify_outcome(self, stance: Optional[str], raw_return_pct: float) -> str:
        """Classify market reaction as WIN, LOSS, or NEUTRAL accounting for transaction friction."""
        stance_norm = str(stance or "NEUTRAL").strip().upper()
        friction = self.friction_threshold_pct

        if stance_norm == "BULLISH":
            if raw_return_pct > friction:
                return "WIN"
            if raw_return_pct < -friction:
                return "LOSS"
            return "NEUTRAL"

        if stance_norm == "BEARISH":
            if raw_return_pct < -friction:
                return "WIN"
            if raw_return_pct > friction:
                return "LOSS"
            return "NEUTRAL"

        if stance_norm == "VOLATILE":
            # Material volatility shock required (>1.50%)
            if abs(raw_return_pct) > 1.50:
                return "WIN"
            return "NEUTRAL"

        # NEUTRAL stance
        if abs(raw_return_pct) > friction:
            return "WIN"
        return "NEUTRAL"

    def assign_credibility_tier(self, hit_rate_pct: float, avg_alpha_1d_pct: float) -> CredibilityTier:
        """Assign publisher credibility tier based on empirical hit rate and directional alpha."""
        if hit_rate_pct >= 65.0 and avg_alpha_1d_pct >= 0.80:
            return CredibilityTier.TIER_A_ALPHA
        if hit_rate_pct >= 55.0 and avg_alpha_1d_pct >= 0.0:
            return CredibilityTier.TIER_B_RELIABLE
        if hit_rate_pct >= 45.0 and avg_alpha_1d_pct >= 0.0:
            return CredibilityTier.TIER_C_NEUTRAL
        return CredibilityTier.TIER_D_UNRELIABLE

    def update_forward_returns(
        self,
        storage: BaseStorage,
        price_fetcher: Optional[Callable[[str], Optional[float]]] = None,
        min_age_hours_1d: int = 24,
        min_age_hours_7d: int = 168,
    ) -> Dict[str, int]:
        """Query storage for pending events and update realized 1d and 7d returns using price_fetcher."""
        if price_fetcher is None:
            logger.debug("No price_fetcher supplied; skipping forward return updates.")
            return {"updated_1d": 0, "updated_7d": 0}

        updated_1d = 0
        updated_7d = 0

        # 1. Update 1-day pending returns
        pending_1d = storage.get_pending_return_events(horizon="1d", min_age_hours=min_age_hours_1d)
        for event in pending_1d:
            if not event.primary_ticker or not event.initial_price or event.initial_price <= 0:
                continue
            try:
                curr_price = price_fetcher(event.primary_ticker)
                if curr_price is not None and curr_price > 0:
                    raw_ret, _ = self.compute_directional_alpha(
                        event.directional_stance, event.initial_price, curr_price
                    )
                    storage.update_event_return(event.event_id, "1d", raw_ret)
                    updated_1d += 1
            except Exception as exc:
                logger.debug("Failed updating 1d return for event %s: %s", event.event_id, exc)

        # 2. Update 7-day pending returns
        pending_7d = storage.get_pending_return_events(horizon="7d", min_age_hours=min_age_hours_7d)
        for event in pending_7d:
            if not event.primary_ticker or not event.initial_price or event.initial_price <= 0:
                continue
            try:
                curr_price = price_fetcher(event.primary_ticker)
                if curr_price is not None and curr_price > 0:
                    raw_ret, _ = self.compute_directional_alpha(
                        event.directional_stance, event.initial_price, curr_price
                    )
                    storage.update_event_return(event.event_id, "7d", raw_ret)
                    updated_7d += 1
            except Exception as exc:
                logger.debug("Failed updating 7d return for event %s: %s", event.event_id, exc)

        return {"updated_1d": updated_1d, "updated_7d": updated_7d}

    def calculate_performance_metrics(
        self,
        storage: BaseStorage,
        min_opportunity: float = 6.0,
        ticker: Optional[str] = None,
        source: Optional[str] = None,
    ) -> PerformanceMetricsSummary:
        """Aggregate realized forward performance metrics and publisher scorecards."""
        records = storage.get_resolved_lineage_records()

        # Clean search parameters
        norm_ticker = None
        if ticker:
            clean_t = ticker.strip().upper()
            norm_ticker = clean_t[1:] if clean_t.startswith("$") else clean_t

        norm_source = source.strip().lower() if source else None

        # Filter records
        filtered: List[EventLineageRecord] = []
        for r in records:
            if r.opportunity_score is not None and r.opportunity_score < min_opportunity:
                continue

            if norm_ticker:
                r_tick = str(r.primary_ticker or "").strip().upper()
                r_clean = r_tick[1:] if r_tick.startswith("$") else r_tick
                if r_clean != norm_ticker:
                    continue

            if norm_source:
                matches_source = any(norm_source in str(s).lower() for s in r.all_sources)
                if not matches_source:
                    continue

            filtered.append(r)

        # Aggregate portfolio-level metrics
        resolved_1d = 0
        wins_1d = 0
        losses_1d = 0
        returns_1d: List[float] = []
        alphas_1d: List[float] = []

        resolved_7d = 0
        wins_7d = 0
        losses_7d = 0
        returns_7d: List[float] = []
        alphas_7d: List[float] = []

        # Publisher aggregation tracking: source_name -> stats
        pub_stats: Dict[str, Dict[str, Any]] = {}

        for rec in filtered:
            stance = rec.directional_stance or "NEUTRAL"

            # 1-day metrics
            if rec.subsequent_return_1d is not None:
                resolved_1d += 1
                r1d = rec.subsequent_return_1d
                returns_1d.append(r1d)

                # Directional alpha calculation
                if stance == "BULLISH":
                    a1d = r1d
                elif stance == "BEARISH":
                    a1d = -r1d
                else:
                    a1d = abs(r1d)
                alphas_1d.append(a1d)

                outcome_1d = self.classify_outcome(stance, r1d)
                if outcome_1d == "WIN":
                    wins_1d += 1
                elif outcome_1d == "LOSS":
                    losses_1d += 1

            # 7-day metrics
            if rec.subsequent_return_7d is not None:
                resolved_7d += 1
                r7d = rec.subsequent_return_7d
                returns_7d.append(r7d)

                if stance == "BULLISH":
                    a7d = r7d
                elif stance == "BEARISH":
                    a7d = -r7d
                else:
                    a7d = abs(r7d)
                alphas_7d.append(a7d)

                outcome_7d = self.classify_outcome(stance, r7d)
                if outcome_7d == "WIN":
                    wins_7d += 1
                elif outcome_7d == "LOSS":
                    losses_7d += 1

            # Multi-source attribution
            sources = rec.all_sources if rec.all_sources else ["Unknown"]
            for s in sources:
                s_name = str(s).strip()
                if not s_name:
                    continue
                if s_name not in pub_stats:
                    pub_stats[s_name] = {
                        "events": 0,
                        "qualities": [],
                        "wins_1d": 0,
                        "losses_1d": 0,
                        "alphas_1d": [],
                        "alphas_7d": [],
                    }
                pub_stats[s_name]["events"] += 1
                if rec.event_quality_score is not None:
                    pub_stats[s_name]["qualities"].append(rec.event_quality_score)

                if rec.subsequent_return_1d is not None:
                    outcome = self.classify_outcome(stance, rec.subsequent_return_1d)
                    if outcome == "WIN":
                        pub_stats[s_name]["wins_1d"] += 1
                    elif outcome == "LOSS":
                        pub_stats[s_name]["losses_1d"] += 1

                    a1 = rec.subsequent_return_1d if stance == "BULLISH" else (-rec.subsequent_return_1d if stance == "BEARISH" else abs(rec.subsequent_return_1d))
                    pub_stats[s_name]["alphas_1d"].append(a1)

                if rec.subsequent_return_7d is not None:
                    a7 = rec.subsequent_return_7d if stance == "BULLISH" else (-rec.subsequent_return_7d if stance == "BEARISH" else abs(rec.subsequent_return_7d))
                    pub_stats[s_name]["alphas_7d"].append(a7)

        # Compute summary percentages
        hit_rate_1d = (wins_1d / (wins_1d + losses_1d) * 100.0) if (wins_1d + losses_1d) > 0 else 0.0
        avg_ret_1d = (sum(returns_1d) / len(returns_1d)) if returns_1d else 0.0
        avg_alpha_1d = (sum(alphas_1d) / len(alphas_1d)) if alphas_1d else 0.0

        hit_rate_7d = (wins_7d / (wins_7d + losses_7d) * 100.0) if (wins_7d + losses_7d) > 0 else 0.0
        avg_ret_7d = (sum(returns_7d) / len(returns_7d)) if returns_7d else 0.0
        avg_alpha_7d = (sum(alphas_7d) / len(alphas_7d)) if alphas_7d else 0.0

        # Build publisher scorecard
        scorecard: List[SourceCredibilityRecord] = []
        for pub_name, st in pub_stats.items():
            tot_decided = st["wins_1d"] + st["losses_1d"]
            pub_hr = (st["wins_1d"] / tot_decided * 100.0) if tot_decided > 0 else 0.0
            pub_qual = (sum(st["qualities"]) / len(st["qualities"])) if st["qualities"] else 0.0
            pub_a1d = (sum(st["alphas_1d"]) / len(st["alphas_1d"])) if st["alphas_1d"] else 0.0
            pub_a7d = (sum(st["alphas_7d"]) / len(st["alphas_7d"])) if st["alphas_7d"] else 0.0

            tier = self.assign_credibility_tier(pub_hr, pub_a1d)

            scorecard.append(
                SourceCredibilityRecord(
                    source_name=pub_name,
                    total_events=st["events"],
                    avg_quality_score=round(pub_qual, 2),
                    hit_rate_pct=round(pub_hr, 2),
                    avg_alpha_1d_pct=round(pub_a1d, 2),
                    avg_alpha_7d_pct=round(pub_a7d, 2),
                    credibility_tier=tier,
                )
            )

        # Sort scorecard: TIER A first, then highest hit rate, then highest alpha
        tier_weights = {
            CredibilityTier.TIER_A_ALPHA: 4,
            CredibilityTier.TIER_B_RELIABLE: 3,
            CredibilityTier.TIER_C_NEUTRAL: 2,
            CredibilityTier.TIER_D_UNRELIABLE: 1,
        }
        scorecard.sort(
            key=lambda x: (tier_weights.get(x.credibility_tier, 0), x.hit_rate_pct, x.avg_alpha_1d_pct),
            reverse=True,
        )

        return PerformanceMetricsSummary(
            total_catalysts=len(filtered),
            resolved_1d=resolved_1d,
            resolved_7d=resolved_7d,
            wins_1d=wins_1d,
            losses_1d=losses_1d,
            hit_rate_1d_pct=round(hit_rate_1d, 2),
            avg_return_1d_pct=round(avg_ret_1d, 2),
            avg_alpha_1d_pct=round(avg_alpha_1d, 2),
            wins_7d=wins_7d,
            losses_7d=losses_7d,
            hit_rate_7d_pct=round(hit_rate_7d, 2),
            avg_return_7d_pct=round(avg_ret_7d, 2),
            avg_alpha_7d_pct=round(avg_alpha_7d, 2),
            source_scorecard=scorecard,
        )
