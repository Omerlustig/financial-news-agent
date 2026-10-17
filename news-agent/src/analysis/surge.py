import logging
from datetime import datetime, timezone
from typing import List, Optional, Set

from src.analysis.corroborator import EventCluster, EventCorroborator
from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    SurgeLevel,
)

logger = logging.getLogger(__name__)


def _normalize_datetime(dt: Optional[datetime]) -> datetime:
    """Normalize datetime to timezone-aware UTC datetime."""
    if dt is None:
        return datetime.now(timezone.utc)
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


class SurgeVelocityEngine:
    """Detects rapid cross-publisher news dissemination velocity and elevates priority."""

    def __init__(
        self,
        breaking_window_minutes: float = 30.0,
        breaking_min_publishers: int = 3,
        elevated_window_minutes: float = 60.0,
        elevated_min_publishers: int = 2,
    ) -> None:
        self.breaking_window_seconds = breaking_window_minutes * 60.0
        self.breaking_min_publishers = breaking_min_publishers
        self.elevated_window_seconds = elevated_window_minutes * 60.0
        self.elevated_min_publishers = elevated_min_publishers

    def _get_distinct_publishers_in_window(
        self,
        articles: List[RawArticle],
        window_seconds: float,
    ) -> int:
        """Find the maximum number of distinct publishers reporting within any window_seconds span."""
        valid_articles = [a for a in articles if a.source_name and a.source_name.strip()]
        if not valid_articles:
            return 0

        # Sort articles chronologically by normalized UTC timestamp
        sorted_articles = sorted(
            valid_articles,
            key=lambda a: _normalize_datetime(a.published_at),
        )

        max_distinct = 0

        for i, start_art in enumerate(sorted_articles):
            start_ts = _normalize_datetime(start_art.published_at).timestamp()
            end_ts = start_ts + window_seconds

            publishers: Set[str] = set()
            for art in sorted_articles[i:]:
                art_ts = _normalize_datetime(art.published_at).timestamp()
                if art_ts <= end_ts:
                    publishers.add(art.source_name.strip().lower())
                else:
                    break

            if len(publishers) > max_distinct:
                max_distinct = len(publishers)

        return max_distinct

    def detect_surge(
        self,
        cluster: EventCluster,
        recent_articles: Optional[List[RawArticle]] = None,
    ) -> SurgeLevel:
        """Evaluate publication timestamps of corroborating articles within an event cluster.

        Rules:
        - If >= 3 distinct publishers within a 30-minute window: returns SurgeLevel.SURGE_BREAKING
        - If >= 2 distinct publishers within a 60-minute window: returns SurgeLevel.ELEVATED
        - Otherwise: returns SurgeLevel.NORMAL
        """
        if not cluster:
            return SurgeLevel.NORMAL

        articles_to_eval: List[RawArticle] = list(cluster.articles) if cluster.articles else []
        if cluster.canonical_article:
            if not any(a.content_hash == cluster.canonical_article.content_hash for a in articles_to_eval):
                articles_to_eval.append(cluster.canonical_article)

        if recent_articles:
            corroborator = EventCorroborator()
            matching_from_recent = [
                art for art in recent_articles
                if cluster.canonical_article and corroborator.are_same_event(art, cluster.canonical_article)
            ]
            if matching_from_recent:
                for art in matching_from_recent:
                    if not any(a.content_hash == art.content_hash for a in articles_to_eval):
                        articles_to_eval.append(art)
            else:
                for art in recent_articles:
                    if not any(a.content_hash == art.content_hash for a in articles_to_eval):
                        articles_to_eval.append(art)

        # 1. Check for SURGE_BREAKING (>= 3 distinct publishers within 30 minutes)
        max_30m_publishers = self._get_distinct_publishers_in_window(
            articles_to_eval,
            self.breaking_window_seconds,
        )
        if max_30m_publishers >= self.breaking_min_publishers:
            return SurgeLevel.SURGE_BREAKING

        # 2. Check for ELEVATED (>= 2 distinct publishers within 60 minutes)
        max_60m_publishers = self._get_distinct_publishers_in_window(
            articles_to_eval,
            self.elevated_window_seconds,
        )
        if max_60m_publishers >= self.elevated_min_publishers:
            return SurgeLevel.ELEVATED

        # 3. Otherwise NORMAL
        return SurgeLevel.NORMAL

    def apply_surge_metadata(
        self,
        analysis: MarketImpactAnalysis,
        surge_level: SurgeLevel,
    ) -> MarketImpactAnalysis:
        """Apply surge velocity telemetry and elevate score/grade on breaking cascades.

        If surge_level == SurgeLevel.SURGE_BREAKING:
        - Sets analysis.is_market_moving = True
        - Guarantees analysis.confidence_score = max(analysis.confidence_score, 8)
        - If analysis.impact_grade in (ImpactGrade.LOW, ImpactGrade.AVERAGE): elevates to ImpactGrade.HIGH
        """
        analysis.surge_level = surge_level

        if surge_level == SurgeLevel.SURGE_BREAKING:
            analysis.is_market_moving = True
            analysis.confidence_score = max(analysis.confidence_score, 8)
            if analysis.impact_grade in (ImpactGrade.LOW, ImpactGrade.AVERAGE):
                analysis.impact_grade = ImpactGrade.HIGH

        return analysis
