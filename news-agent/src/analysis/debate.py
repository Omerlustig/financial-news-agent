import json
import logging
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from src.models import ImpactGrade, MarketImpactAnalysis, RawArticle

logger = logging.getLogger(__name__)


@dataclass
class DebateVerdict:
    """Adversarial synthesis of the bull catalyst thesis vs bear risk counter-arguments."""
    bull_thesis: str
    bear_counter_thesis: str
    net_conviction_score: int
    priced_in_risk_level: str  # "LOW", "MEDIUM", "HIGH"
    invalidation_triggers: List[str] = field(default_factory=list)


@dataclass
class DebateConfig:
    """Configuration for adversarial dialectic debate verification."""
    enabled: bool = True
    min_confidence_to_debate: int = 8
    model: Optional[str] = None


class DialecticDebateEngine:
    """Pressure-tests high-conviction catalysts through an adversarial dialectic debate
    between a Bull Analyst (defending the upside transmission) and a Risk Skeptic Bear
    (challenging consensus, priced-in expectations, and execution risks).
    """

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        config: Optional[DebateConfig] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.config = config or DebateConfig()

    def should_debate(self, analysis: MarketImpactAnalysis) -> bool:
        """Determines if an analysis meets the threshold for deep dialectic debate."""
        if not self.config.enabled:
            return False

        # Triggers if confidence score meets the threshold or impact grade is HIGH or CRITICAL
        high_confidence = analysis.confidence_score >= self.config.min_confidence_to_debate
        high_impact = analysis.impact_grade in (ImpactGrade.HIGH, ImpactGrade.CRITICAL)
        return high_confidence or high_impact

    def _format_debate_prompt(
        self,
        analysis: MarketImpactAnalysis,
        article: RawArticle,
        market_context: Optional[str] = None,
    ) -> str:
        tickers = ", ".join(analysis.affected_tickers) if analysis.affected_tickers else "N/A"
        ctx_section = f"\n{market_context}\n" if market_context else ""
        mechanism = ""
        if analysis.causal_chain and hasattr(analysis.causal_chain, "economic_mechanism"):
            mechanism = analysis.causal_chain.economic_mechanism

        return f"""You are the Chief Investment Officer conducting an adversarial debate on a breaking market catalyst.

[HEADLINE]: {article.title}
[TICKERS]: {tickers}
[INITIAL CATALYST THESIS]: {analysis.catalyst_thesis}
[TRANSMISSION MECHANISM]: {mechanism}
[INITIAL CONFIDENCE SCORE]: {analysis.confidence_score}/10 | [IMPACT GRADE]: {analysis.impact_grade}
{ctx_section}
Conduct a two-sided debate:
1. BULL ANALYST: Defend the upside catalyst, multiple expansion, and economic transmission.
2. RISK SKEPTIC (BEAR): Ruthlessly attack the thesis. Is this already priced in? What are the valuation, regulatory, or customer concentration risks? Where is the margin of safety?
3. SYNTHESIS: Provide the net conviction score (1-10), assess priced-in risk level (LOW/MEDIUM/HIGH), and state 1-3 explicit invalidation triggers where the thesis fails.

Respond ONLY with valid JSON in this exact structure:
{{
  "bull_thesis": "...",
  "bear_counter_thesis": "...",
  "net_conviction_score": 7,
  "priced_in_risk_level": "HIGH",
  "invalidation_triggers": ["...", "..."]
}}"""

    def conduct_debate(
        self,
        analysis: MarketImpactAnalysis,
        article: RawArticle,
        market_context: Optional[str] = None,
    ) -> DebateVerdict:
        """Executes the dialectic debate and returns the synthesized verdict."""
        counterargs = ""
        if analysis.causal_chain and hasattr(analysis.causal_chain, "counterarguments"):
            counterargs = analysis.causal_chain.counterarguments

        if not self.llm_callable:
            return DebateVerdict(
                bull_thesis=analysis.catalyst_thesis,
                bear_counter_thesis=counterargs or "No immediate counter-thesis available.",
                net_conviction_score=analysis.confidence_score,
                priced_in_risk_level="MEDIUM",
                invalidation_triggers=["Adverse regulatory action", "Earnings guidance miss"],
            )

        prompt = self._format_debate_prompt(analysis, article, market_context)
        try:
            raw_response = self.llm_callable(prompt)
            data = json.loads(raw_response.strip())
            net_score = int(data.get("net_conviction_score", analysis.confidence_score))
            net_score = max(1, min(10, net_score))

            return DebateVerdict(
                bull_thesis=str(data.get("bull_thesis", analysis.catalyst_thesis)),
                bear_counter_thesis=str(data.get("bear_counter_thesis", "")),
                net_conviction_score=net_score,
                priced_in_risk_level=str(data.get("priced_in_risk_level", "MEDIUM")).upper(),
                invalidation_triggers=list(data.get("invalidation_triggers", [])),
            )
        except Exception as e:
            logger.warning("Dialectic debate LLM invocation failed, using fallback: %s", e)
            return DebateVerdict(
                bull_thesis=analysis.catalyst_thesis,
                bear_counter_thesis=counterargs or "Counter-arguments not evaluated due to service error.",
                net_conviction_score=analysis.confidence_score,
                priced_in_risk_level="MEDIUM",
                invalidation_triggers=[],
            )
