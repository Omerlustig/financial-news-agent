"""Adversarial Self-Challenger and Falsification Engine.

Forces contrarian self-argumentation, audits priced-in and macro confounding risks,
and articulates explicit quantitative falsification criteria.
"""

import logging
import re
from typing import Any, Dict, List, Optional

from src.models import AdversarialReview

logger = logging.getLogger(__name__)


class AdversarialChallenger:
    """Institutional adversarial stress-testing engine.

    Evaluates the news-derived thesis against:
    1. Bearish / contrarian counter-thesis: strongest counter-argument
    2. Priced-in risk: "sell the news" or prior discounting
    3. Macro confounding risk: beta vs alpha attribution
    4. Falsification threshold: explicit, measurable condition invalidating the trade hypothesis
    5. Thesis robustness determination
    """

    def challenge_thesis(
        self,
        title: str,
        summary: str = "",
        stance: str = "NEUTRAL",
        affected_tickers: Optional[List[str]] = None,
        llm_payload: Optional[Dict[str, Any]] = None,
    ) -> AdversarialReview:
        """Executes adversarial challenge on thesis.

        If `llm_payload` is provided, parses and validates `AdversarialReview`.
        Otherwise, deterministically generates the contrarian challenge.
        """
        if llm_payload is not None:
            parsed = self._try_parse_llm_payload(llm_payload)
            if parsed is not None:
                return parsed

        return self._compute_deterministic(title, summary, stance, affected_tickers)

    def _try_parse_llm_payload(
        self, payload: Dict[str, Any]
    ) -> Optional[AdversarialReview]:
        """Attempts to parse and validate AdversarialReview from LLM payload."""
        raw_adv = payload.get("adversarial_review", payload)
        if not isinstance(raw_adv, dict):
            return None

        # Check required fields
        required_fields = [
            "bearish_counter_thesis",
            "priced_in_risk",
            "macro_confounding_risk",
            "falsification_threshold",
        ]
        if not any(field in raw_adv for field in required_fields):
            return None

        try:
            counter = str(raw_adv.get("bearish_counter_thesis", "")).strip()
            priced_in = str(raw_adv.get("priced_in_risk", "")).strip()
            macro = str(raw_adv.get("macro_confounding_risk", "")).strip()
            falsification = str(raw_adv.get("falsification_threshold", "")).strip()
            is_robust = bool(raw_adv.get("is_thesis_robust", True))

            # If any required string field is empty, fallback to deterministic fill for that field
            if not counter or not priced_in or not macro or not falsification:
                return None

            return AdversarialReview(
                bearish_counter_thesis=counter,
                priced_in_risk=priced_in,
                macro_confounding_risk=macro,
                falsification_threshold=falsification,
                is_thesis_robust=is_robust,
            )
        except Exception as e:
            logger.warning(
                "Failed to parse AdversarialReview from llm_payload: %s. Using deterministic challenger.",
                e,
            )
            return None

    def _compute_deterministic(
        self,
        title: str,
        summary: str = "",
        stance: str = "NEUTRAL",
        affected_tickers: Optional[List[str]] = None,
    ) -> AdversarialReview:
        """Deterministically generates contrarian thesis, priced-in audit, macro risk, and falsification threshold."""
        combined_text = f"{title} {summary}".lower()
        normalized_stance = str(stance).strip().upper()

        tickers_str = (
            ", ".join(affected_tickers)
            if affected_tickers and len(affected_tickers) > 0
            else "the company and exposed sector"
        )

        # Assess if unverified rumors or vague speculation weaken thesis robustness
        speculative_triggers = [
            "rumor",
            "speculation",
            "sources say",
            "unconfirmed",
            "anonymous",
            "reportedly",
        ]
        is_speculative = any(trig in combined_text for trig in speculative_triggers)
        is_robust = not is_speculative

        if "BULLISH" in normalized_stance:
            counter_thesis = (
                f"Contrarian thesis for {tickers_str}: Potential margin compression from rising input costs, "
                f"aggressive competitive pricing, execution delays in capacity scaling, customer churn, "
                f"or heightened regulatory scrutiny that offsets projected gains."
            )
            priced_in_risk = (
                f"High risk of a 'sell the news' reaction if recent multiple expansion and trading volume "
                f"already discounted the catalyst prior to formal announcement."
            )
            macro_confounding_risk = (
                "Beta contamination risk: broader index momentum, Treasury yield shifts, or sector rotation "
                "may explain price action rather than idiosyncratic company alpha."
            )
            falsification_threshold = (
                f"Company reporting next quarter EPS miss >5%, gross margin contracting below prior baseline, "
                f"or customer churn exceeding 2% for {tickers_str} invalidates the bullish thesis."
            )

        elif "BEARISH" in normalized_stance:
            counter_thesis = (
                f"Contrarian bull case for {tickers_str}: Market may overreact to short-term headwinds. "
                f"Damage could be site-specific or temporary, while cost-saving capital reallocation, "
                f"alternative supplier substitution, or short-covering squeezes offer downside cushioning."
            )
            priced_in_risk = (
                f"Downside may already be fully discounted in {tickers_str} following prior price drift "
                f"and depressed valuation multiples, limiting asymmetric short follow-through."
            )
            macro_confounding_risk = (
                "Market-wide macro risk-off sentiment or systemic index liquidations may exaggerate "
                "the perceived standalone deterioration of the company."
            )
            falsification_threshold = (
                f"Company reporting next quarter EPS > $1.20 with gross margin >= 55%, or regulatory agency "
                f"granting unconditional merger approval for {tickers_str} invalidates the bearish thesis."
            )

        elif "VOLATILE" in normalized_stance:
            counter_thesis = (
                f"Whipsaw risk for {tickers_str}: Erratic two-sided price swings may trigger stop-loss cascades "
                f"without establishing sustainable directional trend continuation."
            )
            priced_in_risk = (
                "Elevated implied volatility suggests options markets have already priced substantial uncertainty, "
                "leading to rapid volatility crush post-event."
            )
            macro_confounding_risk = (
                "Cross-asset macro volatility and benchmark liquidations overshadow single-stock idiosyncratic fundamentals."
            )
            falsification_threshold = (
                f"Realized 5-day volatility contracting below historical 30-day baseline for {tickers_str} "
                f"invalidates the elevated volatility thesis."
            )

        else:  # NEUTRAL
            counter_thesis = (
                f"Inertia and slow drag for {tickers_str}: Event lacks immediate cash-flow transmission mechanism; "
                f"secondary compliance overhead or operational friction could create an unnoticed drag on margins."
            )
            priced_in_risk = (
                "Catalyst is largely routine or consensus information, offering zero asymmetric edge over prevailing pricing."
            )
            macro_confounding_risk = (
                "Stock price behavior will almost entirely reflect benchmark beta and macroeconomic rate expectations "
                "rather than this event."
            )
            falsification_threshold = (
                f"Material unscheduled 8-K filing, unexpected dividend revision, or unexpected guidance update "
                f"for {tickers_str} invalidates the neutral thesis."
            )

        return AdversarialReview(
            bearish_counter_thesis=counter_thesis,
            priced_in_risk=priced_in_risk,
            macro_confounding_risk=macro_confounding_risk,
            falsification_threshold=falsification_threshold,
            is_thesis_robust=is_robust,
        )
