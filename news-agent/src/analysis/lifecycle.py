import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from src.models import DirectionalStance, ImpactGrade, MarketImpactAnalysis

logger = logging.getLogger(__name__)


class SignalEvolutionState(str, Enum):
    """Lifecycle evolution stages of a market-moving thesis over time."""
    NEW = "NEW"
    CORROBORATED = "CORROBORATED"
    STRENGTHENED = "STRENGTHENED"
    WEAKENED = "WEAKENED"
    FALSIFIED = "FALSIFIED"
    STALE = "STALE"


@dataclass
class TickerQuotaConfig:
    """Configuration for per-ticker fair-share alert rate-limiting."""
    enabled: bool = True
    max_alerts_per_ticker: int = 2
    rolling_window_hours: float = 6.0


class TickerQuotaTracker:
    """Guarantees fair-share alerting across tickers, throttling repetitive coverage
    of high-volume mega-caps unless the catalyst represents CRITICAL severity.
    """

    def __init__(self, config: Optional[TickerQuotaConfig] = None) -> None:
        self.config = config or TickerQuotaConfig()
        # ticker -> list of (timestamp, ImpactGrade)
        self._history: Dict[str, List[Tuple[float, ImpactGrade]]] = defaultdict(list)

    def _clean_stale(self, ticker: str, current_time: float) -> None:
        window_seconds = self.config.rolling_window_hours * 3600
        self._history[ticker] = [
            (ts, grade) for ts, grade in self._history[ticker]
            if current_time - ts <= window_seconds
        ]

    def check_quota(
        self,
        ticker: str,
        impact_grade: ImpactGrade,
        current_time: Optional[float] = None,
    ) -> Tuple[bool, str]:
        """Checks if an alert for a ticker is allowed within the rolling window."""
        if not self.config.enabled:
            return True, "QUOTA_DISABLED"

        # Critical impact catalysts always bypass fair-share quotas
        if impact_grade == ImpactGrade.CRITICAL:
            return True, "CRITICAL_BYPASS"

        symbol = ticker.strip().upper().lstrip("$")
        now = current_time or time.time()
        self._clean_stale(symbol, now)

        count = len(self._history[symbol])
        if count >= self.config.max_alerts_per_ticker:
            logger.info("Throttled alert for %s: quota exceeded (%d/%d)", symbol, count, self.config.max_alerts_per_ticker)
            return False, "TICKER_QUOTA_EXCEEDED"

        return True, "QUOTA_OK"

    def record_alert(
        self,
        ticker: str,
        impact_grade: ImpactGrade,
        current_time: Optional[float] = None,
    ) -> None:
        """Records an alert issuance timestamp for a given ticker."""
        symbol = ticker.strip().upper().lstrip("$")
        now = current_time or time.time()
        self._history[symbol].append((now, impact_grade))


class SignalLifecycleEngine:
    """Tracks how an investment thesis evolves across subsequent reporting cycles."""

    def evaluate_evolution(
        self,
        current: MarketImpactAnalysis,
        prior: Optional[MarketImpactAnalysis] = None,
    ) -> SignalEvolutionState:
        """Compares current analysis to prior reporting to infer thesis evolution."""
        if prior is None:
            return SignalEvolutionState.NEW

        # Check for stance reversal (e.g. BULLISH vs BEARISH)
        opposites = {
            DirectionalStance.BULLISH: DirectionalStance.BEARISH,
            DirectionalStance.BEARISH: DirectionalStance.BULLISH,
        }
        if current.directional_stance and prior.directional_stance:
            if opposites.get(current.directional_stance) == prior.directional_stance:
                return SignalEvolutionState.FALSIFIED

        # Check for thesis reinforcement: matching stance with equal/higher confidence
        if current.directional_stance == prior.directional_stance:
            if current.confidence_score >= prior.confidence_score:
                return SignalEvolutionState.STRENGTHENED
            else:
                return SignalEvolutionState.CORROBORATED

        return SignalEvolutionState.CORROBORATED
