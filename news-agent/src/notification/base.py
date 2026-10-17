from abc import ABC, abstractmethod
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
    def send_message(self, text: str) -> bool:
        """Send a plain or markup-formatted message directly to the notification destination.

        Args:
            text: Formatted message payload.

        Returns:
            bool: True if delivery succeeded, False otherwise.
        """
        pass
