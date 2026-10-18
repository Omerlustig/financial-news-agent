from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from src.models import EventLineageRecord, MarketImpactAnalysis, RawArticle


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

    @abstractmethod
    def record_ticker_sentiment(
        self,
        ticker: str,
        sentiment_score: float,
        sentiment_label: str,
        article_hash: str,
        published_at: Optional[datetime] = None,
    ) -> None:
        """Record a sentiment history data point for a specific ticker."""
        raise NotImplementedError

    @abstractmethod
    def get_ticker_sentiment_window(self, ticker: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve sentiment history records for a ticker within the past `hours`."""
        raise NotImplementedError

    @abstractmethod
    def get_all_tickers_trend_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return aggregate sentiment trend summaries across all tickers in the past `hours`."""
        raise NotImplementedError

    @abstractmethod
    def record_sector_sentiment(
        self,
        sector: str,
        sentiment_score: float,
        sentiment_label: str,
        article_hash: str,
        published_at: Optional[datetime] = None,
    ) -> None:
        """Record a sentiment history data point for a specific sector."""
        raise NotImplementedError

    @abstractmethod
    def get_sector_sentiment_window(self, sector: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve sentiment history records for a sector within the past `hours`."""
        raise NotImplementedError

    @abstractmethod
    def get_all_sectors_trend_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return aggregate sentiment trend summaries across all sectors in the past `hours`."""
        raise NotImplementedError

    @abstractmethod
    def record_event_lineage(self, event: EventLineageRecord) -> None:
        """Persist or update an event lineage tracking record."""
        raise NotImplementedError

    @abstractmethod
    def get_event_lineage(self, event_id: str) -> Optional[EventLineageRecord]:
        """Retrieve an event lineage record by unique event ID."""
        raise NotImplementedError

    @abstractmethod
    def find_existing_event(
        self, ticker: str, title_tokens: List[str], window_hours: int = 48
    ) -> Optional[EventLineageRecord]:
        """Search for an existing event lineage within window_hours matching ticker and >= 2 title tokens."""
        raise NotImplementedError

    @abstractmethod
    def update_event_return(self, event_id: str, horizon: str, return_pct: float) -> None:
        """Update forward realized return for a tracked event across specified horizon (e.g. 1d, 7d)."""
        raise NotImplementedError

    @abstractmethod
    def get_pending_return_events(self, horizon: str = "1d", min_age_hours: int = 24) -> List[EventLineageRecord]:
        """Retrieve tracked events eligible for forward return evaluation across specified horizon."""
        raise NotImplementedError

    @abstractmethod
    def get_resolved_lineage_records(self) -> List[EventLineageRecord]:
        """Retrieve all tracked events that have realized forward returns (1d or 7d)."""
        raise NotImplementedError

    def list_event_lineage(self, ticker: Optional[str] = None, limit: int = 20) -> List[EventLineageRecord]:
        """List recent event lineage tracking records, optionally filtered by primary ticker."""
        return []

    def close(self) -> None:
        """Release storage connections or system resources."""
        pass

    def __enter__(self) -> "BaseStorage":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
