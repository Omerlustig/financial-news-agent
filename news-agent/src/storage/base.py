from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from src.models import MarketImpactAnalysis, RawArticle


class BaseStorage(ABC):
    """Abstract base class defining contract for article persistence, caching, and telemetry."""

    @abstractmethod
    def is_article_seen(self, content_hash: str) -> bool:
        """Check whether an article with the given content hash has already been recorded."""
        raise NotImplementedError

    @abstractmethod
    def filter_unseen(self, articles: List[RawArticle]) -> List[RawArticle]:
        """Filter a list of articles, returning only those not yet recorded in storage."""
        raise NotImplementedError

    @abstractmethod
    def record_article(
        self,
        article: RawArticle,
        analysis: Optional[MarketImpactAnalysis] = None,
        alerted: bool = False,
    ) -> None:
        """Persist or update an article and its optional analysis and alert status."""
        raise NotImplementedError

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """Return aggregate storage statistics (total articles, alerted count, market moving count)."""
        raise NotImplementedError

    @abstractmethod
    def get_feed_cache(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve cached ETag and Last-Modified headers for a feed URL."""
        raise NotImplementedError

    @abstractmethod
    def set_feed_cache(self, url: str, etag: Optional[str], last_modified: Optional[str]) -> None:
        """Store or update cached ETag and Last-Modified headers for a feed URL."""
        raise NotImplementedError

    @abstractmethod
    def record_llm_metric(
        self,
        content_hash: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record LLM token consumption and estimated cost."""
        raise NotImplementedError

    @abstractmethod
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return aggregate LLM usage telemetry and expenditure metrics."""
        raise NotImplementedError

    @abstractmethod
    def get_recent_catalysts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve recorded articles from the past `hours` where alerted=1 or status='OPPORTUNITY_CANDIDATE'."""
        raise NotImplementedError


    def close(self) -> None:
        """Release storage connections or system resources."""
        pass

    def __enter__(self) -> "BaseStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
