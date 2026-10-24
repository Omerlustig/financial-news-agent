from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from src.models import RawArticle, MarketImpactAnalysis


class BaseNotifier(ABC):
    """Abstract interface defining delivery dispatchers for market news alerts."""

    @abstractmethod
    def send_alert(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        """Format and dispatch an alert for an analyzed market-moving article.

        Args:
            article: Ingested raw article metadata.
            analysis: Institutional market impact evaluation.

        Returns:
            bool: True if alert dispatch was acknowledged successfully, False otherwise.
        """
        pass

    @abstractmethod
    def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:

        """Send a plain or markup-formatted message directly to the notification destination.

        Args:
            text: Formatted message payload.

        Returns:
            bool: True if delivery succeeded, False otherwise.
        """
        pass

    def send_batch_alerts(self, items: Any) -> bool:
        """Format and dispatch multiple recommendations in a single notification push.

        Default implementation falls back to individual send_alert calls.
        Subclasses (e.g. TelegramNotifier) override this to combine into 1 push.
        """
        if not items:
            return False
        success = True
        for art, ana in items:
            if not self.send_alert(art, ana):
                success = False
        return success
