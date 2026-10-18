"""Quantitative market trend, volume buzz, and sentiment momentum engine.

Calculates short-term ticker publication frequency, baseline volume ratios,
and multi-window sentiment velocity to determine market trend verdicts.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

from src.models import (
    MarketImpactAnalysis,
    SectorTrendMetrics,
    SentimentLabel,
    TrendMetrics,
    TrendVerdict,
)
from src.storage.base import BaseStorage

logger = logging.getLogger(__name__)


class TrendEngine:
    """Quantitative momentum and news buzz acceleration analyzer."""

    def calculate_ticker_trend(
        self,
        storage: BaseStorage,
        ticker: str,
        current_sentiment: Optional[float] = None,
    ) -> TrendMetrics:
        """Calculate volume buzz ratio, sentiment velocity, acceleration, divergence, and trend verdict for a ticker.

        Args:
            storage: Storage backend used to query historical ticker sentiment records.
            ticker: Ticker symbol (with or without leading '$').
            current_sentiment: Optional sentiment score of the incoming event being evaluated.

        Returns:
            TrendMetrics containing buzz_ratio, sentiment_velocity, sentiment_acceleration,
            is_divergent, divergence_type, trend_verdict, and articles_24h_count.
        """
        if not ticker or not isinstance(ticker, str):
            return TrendMetrics(
                buzz_ratio=1.0,
                sentiment_velocity=0.0,
                sentiment_acceleration=0.0,
                is_divergent=False,
                divergence_type=None,
                trend_verdict=TrendVerdict.NEUTRAL_CONSOLIDATION,
                articles_24h_count=1,
            )

        cleaned_ticker = ticker.strip().upper()
        query_ticker = cleaned_ticker.lstrip("$")
        if not query_ticker:
            return TrendMetrics(
                buzz_ratio=1.0,
                sentiment_velocity=0.0,
                sentiment_acceleration=0.0,
                is_divergent=False,
                divergence_type=None,
                trend_verdict=TrendVerdict.NEUTRAL_CONSOLIDATION,
                articles_24h_count=1,
            )

        # 1. Query historical sentiment windows
        try:
            recent_24h = storage.get_ticker_sentiment_window(query_ticker, hours=24)
            if not recent_24h and query_ticker != cleaned_ticker:
                alt_24h = storage.get_ticker_sentiment_window(cleaned_ticker, hours=24)
                if alt_24h:
                    recent_24h = alt_24h
        except Exception as exc:
            logger.warning("Failed querying 24h ticker window for %s: %s", query_ticker, exc)
            recent_24h = []

        try:
            recent_7d = storage.get_ticker_sentiment_window(query_ticker, hours=168)
            if not recent_7d and query_ticker != cleaned_ticker:
                alt_7d = storage.get_ticker_sentiment_window(cleaned_ticker, hours=168)
                if alt_7d:
                    recent_7d = alt_7d
        except Exception as exc:
            logger.warning("Failed querying 7d ticker window for %s: %s", query_ticker, exc)
            recent_7d = []

        # 2. Count articles in last 24h & 7d baseline
        count_24h = len(recent_24h)
        if current_sentiment is not None:
            count_24h += 1

        total_7d = max(len(recent_7d), len(recent_24h))
        if current_sentiment is not None:
            total_7d += 1

        baseline_daily_avg = max(total_7d / 7.0, 1.0)
        buzz_ratio = round(max(float(count_24h), 1.0) / baseline_daily_avg, 2)

        # 3. Sentiment Velocity & Acceleration calculation
        scores_24h: List[float] = []
        for r in recent_24h:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_24h.append(float(s))
                except (ValueError, TypeError):
                    pass

        if current_sentiment is not None:
            try:
                scores_24h.append(float(current_sentiment))
            except (ValueError, TypeError):
                pass

        avg_sentiment_24h = sum(scores_24h) / len(scores_24h) if scores_24h else 0.0

        # Prior windows (Window 1: 24h - 48h; Window 2: 48h - 72h)
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_48h = now - timedelta(hours=48)
        cutoff_72h = now - timedelta(hours=72)

        recent_24h_ids = {
            r.get("article_hash") or r.get("id")
            for r in recent_24h
            if (r.get("article_hash") or r.get("id")) is not None
        }

        prior_articles: List[Dict[str, Any]] = []
        prior_2_articles: List[Dict[str, Any]] = []
        unversioned_prior: List[Dict[str, Any]] = []

        for r in recent_7d:
            item_id = r.get("article_hash") or r.get("id")
            if item_id is not None and item_id in recent_24h_ids:
                continue
            if r in recent_24h:
                continue

            # Check timestamp if present
            ts_val = r.get("published_at") or r.get("created_at")
            if ts_val is not None:
                try:
                    dt = datetime.fromisoformat(ts_val) if isinstance(ts_val, str) else ts_val
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if cutoff_48h <= dt < cutoff_24h:
                        prior_articles.append(r)
                    elif cutoff_72h <= dt < cutoff_48h:
                        prior_2_articles.append(r)
                    continue
                except Exception:
                    pass

            # If timestamps are missing/unparseable (e.g. test fixtures), collect as unversioned
            unversioned_prior.append(r)

        if not prior_articles and unversioned_prior:
            prior_articles.append(unversioned_prior[0])
            if len(unversioned_prior) > 1:
                prior_2_articles.extend(unversioned_prior[1:])

        scores_prior: List[float] = []
        for r in prior_articles:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_prior.append(float(s))
                except (ValueError, TypeError):
                    pass

        scores_prior_2: List[float] = []
        for r in prior_2_articles:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_prior_2.append(float(s))
                except (ValueError, TypeError):
                    pass

        if scores_prior and scores_24h:
            avg_sentiment_prior = sum(scores_prior) / len(scores_prior)
            sentiment_velocity = round(avg_sentiment_24h - avg_sentiment_prior, 3)
        else:
            sentiment_velocity = 0.0

        if scores_prior and scores_prior_2:
            avg_sentiment_prior_2 = sum(scores_prior_2) / len(scores_prior_2)
            velocity_prior = avg_sentiment_prior - avg_sentiment_prior_2
            sentiment_acceleration = round(sentiment_velocity - velocity_prior, 3)
        else:
            sentiment_acceleration = 0.0

        # Divergence detector
        if buzz_ratio >= 1.8 and avg_sentiment_24h <= -0.3 and sentiment_velocity <= -0.1:
            is_divergent = True
            divergence_type = "BEARISH_PANIC_LIQUIDATION"
        elif buzz_ratio <= 1.3 and avg_sentiment_24h >= 0.3 and sentiment_velocity >= 0.15:
            is_divergent = True
            divergence_type = "BULLISH_ACCUMULATION_DIVERGENCE"
        else:
            is_divergent = False
            divergence_type = None

        # 4. Determine TrendVerdict
        if avg_sentiment_24h >= 0.45 and (buzz_ratio >= 1.5 or sentiment_velocity > 0.2):
            verdict = TrendVerdict.STRONG_BULLISH
        elif avg_sentiment_24h <= -0.45 and buzz_ratio >= 1.5:
            verdict = TrendVerdict.PANIC_CAPITULATION
        elif sentiment_acceleration >= 0.25 and avg_sentiment_24h > 0:
            verdict = TrendVerdict.BULLISH_REVERSAL
        elif sentiment_acceleration <= -0.25 and avg_sentiment_24h < 0:
            verdict = TrendVerdict.BEARISH_BREAKDOWN
        elif avg_sentiment_24h >= 0.15 and sentiment_velocity >= -0.05:
            verdict = TrendVerdict.MODERATE_BULLISH
        elif avg_sentiment_24h <= -0.15:
            verdict = TrendVerdict.MODERATE_BEARISH
        else:
            verdict = TrendVerdict.NEUTRAL_CONSOLIDATION

        return TrendMetrics(
            buzz_ratio=buzz_ratio,
            sentiment_velocity=sentiment_velocity,
            sentiment_acceleration=sentiment_acceleration,
            is_divergent=is_divergent,
            divergence_type=divergence_type,
            trend_verdict=verdict,
            articles_24h_count=max(count_24h, 1),
        )

    def calculate_sector_trend(
        self,
        storage: BaseStorage,
        sector: str,
        current_sentiment: Optional[float] = None,
    ) -> SectorTrendMetrics:
        """Calculate volume buzz ratio, sentiment velocity, acceleration, and trend verdict for a sector.

        Args:
            storage: Storage backend used to query historical sector sentiment records.
            sector: Market sector name (e.g. 'Semiconductors').
            current_sentiment: Optional sentiment score of the incoming event being evaluated.

        Returns:
            SectorTrendMetrics containing sector momentum, velocity, acceleration, and verdict.
        """
        if not sector or not isinstance(sector, str) or storage is None:
            return SectorTrendMetrics(
                sector=str(sector or ""),
                avg_sentiment=0.0,
                sentiment_velocity=0.0,
                sentiment_acceleration=0.0,
                buzz_ratio=1.0,
                article_count=1,
                sentiment_label=SentimentLabel.NEUTRAL,
                trend_verdict=TrendVerdict.NEUTRAL_CONSOLIDATION,
            )

        cleaned_sector = sector.strip()
        if not cleaned_sector:
            return SectorTrendMetrics(
                sector="",
                avg_sentiment=0.0,
                sentiment_velocity=0.0,
                sentiment_acceleration=0.0,
                buzz_ratio=1.0,
                article_count=1,
                sentiment_label=SentimentLabel.NEUTRAL,
                trend_verdict=TrendVerdict.NEUTRAL_CONSOLIDATION,
            )

        # 1. Query historical sector sentiment windows
        try:
            recent_24h = storage.get_sector_sentiment_window(cleaned_sector, hours=24)
        except Exception as exc:
            logger.warning("Failed querying 24h sector window for %s: %s", cleaned_sector, exc)
            recent_24h = []

        try:
            recent_7d = storage.get_sector_sentiment_window(cleaned_sector, hours=168)
        except Exception as exc:
            logger.warning("Failed querying 7d sector window for %s: %s", cleaned_sector, exc)
            recent_7d = []

        # 2. Count articles in last 24h & 7d baseline
        count_24h = len(recent_24h)
        if current_sentiment is not None:
            count_24h += 1

        total_7d = max(len(recent_7d), len(recent_24h))
        if current_sentiment is not None:
            total_7d += 1

        baseline_daily_avg = max(total_7d / 7.0, 1.0)
        buzz_ratio = round(max(float(count_24h), 1.0) / baseline_daily_avg, 2)

        # 3. Sentiment & Velocity calculation
        scores_24h: List[float] = []
        for r in recent_24h:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_24h.append(float(s))
                except (ValueError, TypeError):
                    pass

        if current_sentiment is not None:
            try:
                scores_24h.append(float(current_sentiment))
            except (ValueError, TypeError):
                pass

        avg_sentiment_24h = sum(scores_24h) / len(scores_24h) if scores_24h else 0.0

        # Prior windows (Window 1: 24h - 48h; Window 2: 48h - 72h)
        now = datetime.now(timezone.utc)
        cutoff_24h = now - timedelta(hours=24)
        cutoff_48h = now - timedelta(hours=48)
        cutoff_72h = now - timedelta(hours=72)

        recent_24h_ids = {
            r.get("article_hash") or r.get("id")
            for r in recent_24h
            if (r.get("article_hash") or r.get("id")) is not None
        }

        prior_articles: List[Dict[str, Any]] = []
        prior_2_articles: List[Dict[str, Any]] = []
        unversioned_prior: List[Dict[str, Any]] = []

        for r in recent_7d:
            item_id = r.get("article_hash") or r.get("id")
            if item_id is not None and item_id in recent_24h_ids:
                continue
            if r in recent_24h:
                continue

            ts_val = r.get("published_at") or r.get("created_at")
            if ts_val is not None:
                try:
                    dt = datetime.fromisoformat(ts_val) if isinstance(ts_val, str) else ts_val
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    if cutoff_48h <= dt < cutoff_24h:
                        prior_articles.append(r)
                    elif cutoff_72h <= dt < cutoff_48h:
                        prior_2_articles.append(r)
                    continue
                except Exception:
                    pass

            unversioned_prior.append(r)

        if not prior_articles and unversioned_prior:
            prior_articles.append(unversioned_prior[0])
            if len(unversioned_prior) > 1:
                prior_2_articles.extend(unversioned_prior[1:])

        scores_prior: List[float] = []
        for r in prior_articles:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_prior.append(float(s))
                except (ValueError, TypeError):
                    pass

        scores_prior_2: List[float] = []
        for r in prior_2_articles:
            s = r.get("sentiment_score")
            if s is not None:
                try:
                    scores_prior_2.append(float(s))
                except (ValueError, TypeError):
                    pass

        if scores_prior and scores_24h:
            avg_sentiment_prior = sum(scores_prior) / len(scores_prior)
            sentiment_velocity = round(avg_sentiment_24h - avg_sentiment_prior, 3)
        else:
            sentiment_velocity = 0.0

        if scores_prior and scores_prior_2:
            avg_sentiment_prior_2 = sum(scores_prior_2) / len(scores_prior_2)
            velocity_prior = avg_sentiment_prior - avg_sentiment_prior_2
            sentiment_acceleration = round(sentiment_velocity - velocity_prior, 3)
        else:
            sentiment_acceleration = 0.0

        # Sentiment label
        if avg_sentiment_24h >= 0.5:
            label = SentimentLabel.VERY_BULLISH
        elif avg_sentiment_24h >= 0.15:
            label = SentimentLabel.BULLISH
        elif avg_sentiment_24h <= -0.5:
            label = SentimentLabel.VERY_BEARISH
        elif avg_sentiment_24h <= -0.15:
            label = SentimentLabel.BEARISH
        else:
            label = SentimentLabel.NEUTRAL

        # Trend verdict
        if avg_sentiment_24h >= 0.45 and (buzz_ratio >= 1.5 or sentiment_velocity > 0.2):
            verdict = TrendVerdict.STRONG_BULLISH
        elif avg_sentiment_24h <= -0.45 and buzz_ratio >= 1.5:
            verdict = TrendVerdict.PANIC_CAPITULATION
        elif sentiment_acceleration >= 0.25 and avg_sentiment_24h > 0:
            verdict = TrendVerdict.BULLISH_REVERSAL
        elif sentiment_acceleration <= -0.25 and avg_sentiment_24h < 0:
            verdict = TrendVerdict.BEARISH_BREAKDOWN
        elif avg_sentiment_24h >= 0.15 and sentiment_velocity >= -0.05:
            verdict = TrendVerdict.MODERATE_BULLISH
        elif avg_sentiment_24h <= -0.15:
            verdict = TrendVerdict.MODERATE_BEARISH
        else:
            verdict = TrendVerdict.NEUTRAL_CONSOLIDATION

        return SectorTrendMetrics(
            sector=cleaned_sector,
            avg_sentiment=round(avg_sentiment_24h, 3),
            sentiment_velocity=sentiment_velocity,
            sentiment_acceleration=sentiment_acceleration,
            buzz_ratio=buzz_ratio,
            article_count=max(count_24h, 1),
            sentiment_label=label,
            trend_verdict=verdict,
        )

    def enrich_analysis_with_sector_trends(
        self,
        storage: BaseStorage,
        analysis: MarketImpactAnalysis,
    ) -> MarketImpactAnalysis:
        """Enrich a MarketImpactAnalysis instance with sector trend metrics for its affected sectors.

        Args:
            storage: Active storage backend.
            analysis: The MarketImpactAnalysis output.

        Returns:
            The mutated analysis with its `sector_trends` populated.
        """
        if storage is None or not analysis.affected_sectors:
            return analysis

        current_sentiment = (
            float(analysis.sentiment.score)
            if analysis.sentiment is not None and analysis.sentiment.score is not None
            else None
        )

        sector_trends: List[SectorTrendMetrics] = []
        for sector in analysis.affected_sectors:
            sector_name = sector.strip() if isinstance(sector, str) else str(sector)
            if not sector_name:
                continue
            metrics = self.calculate_sector_trend(
                storage=storage,
                sector=sector_name,
                current_sentiment=current_sentiment,
            )
            sector_trends.append(metrics)

        analysis.sector_trends = sector_trends
        return analysis

    def enrich_analysis_with_trend(
        self,
        storage: BaseStorage,
        analysis: MarketImpactAnalysis,
    ) -> MarketImpactAnalysis:
        """Enrich a MarketImpactAnalysis instance with historical trend metrics for its primary ticker and sectors.

        Args:
            storage: Active storage backend.
            analysis: The MarketImpactAnalysis output from the LLM/corroborator.

        Returns:
            The mutated analysis with its `trend` and `sector_trends` fields populated.
        """
        if storage is None:
            return analysis

        if analysis.affected_tickers:
            primary_ticker = analysis.affected_tickers[0]
            current_sentiment = (
                float(analysis.sentiment.score)
                if analysis.sentiment is not None and analysis.sentiment.score is not None
                else None
            )

            trend_metrics = self.calculate_ticker_trend(
                storage=storage,
                ticker=primary_ticker,
                current_sentiment=current_sentiment,
            )
            analysis.trend = trend_metrics

        # Also enrich sector trends for all affected sectors
        self.enrich_analysis_with_sector_trends(storage=storage, analysis=analysis)

        return analysis
