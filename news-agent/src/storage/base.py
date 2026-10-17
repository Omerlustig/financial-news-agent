from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from src.models import MarketImpactAnalysis, RawArticle


class BaseStorage(ABC):
    """Abstract base class defining contract for article persistence and deduplication."""

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

    def close(self) -> None:
        """Release storage connections or system resources."""
        pass

    def __enter__(self) -> "BaseStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
