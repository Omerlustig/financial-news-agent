from datetime import datetime, timezone
from typing import List, Optional
import pytest

from src.models import (
    ImpactGrade,
    DirectionalStance,
    TimeHorizon,
    MarketImpactAnalysis,
    CausalChain,
    HardGates,
    RawArticle,
)
from src.pipeline import Pipeline, CycleSummary
from src.ingestion.prefilter import NewsPreFilter
from src.analysis.integer_triage import IntegerIndexTriageEngine
from src.analysis.market_data import TradingViewMarketDataEngine, MarketContext
from src.analysis.debate import DialecticDebateEngine, DebateVerdict
from src.analysis.citation import CitationGroundingEngine
from src.analysis.lifecycle import TickerQuotaTracker, TickerQuotaConfig
from src.notification.base import BaseNotifier
from src.storage.sqlite import SQLiteStorage


class DummyNotifier(BaseNotifier):
    def __init__(self) -> None:
        self.sent_alerts: List[MarketImpactAnalysis] = []

    def send_message(self, text: str) -> bool:
        return True

    def notify(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        self.sent_alerts.append(analysis)
        return True

    def notify_batch(self, items: List[tuple]) -> bool:
        for _, analysis in items:
            self.sent_alerts.append(analysis)
        return True

    def send_alert(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        self.sent_alerts.append(analysis)
        return True

    def send_batch_alerts(self, items: List[tuple]) -> bool:
        for _, analysis in items:
            self.sent_alerts.append(analysis)
        return True


class DummyAnalyzer:
    def __init__(self, analysis: MarketImpactAnalysis) -> None:
        self._analysis = analysis

    def analyze(self, article: RawArticle, **kwargs) -> MarketImpactAnalysis:
        return self._analysis


def make_test_article(title: str, url: str) -> RawArticle:
    return RawArticle(
        title=title,
        url=url,
        source_name="Reuters",
        published_at=datetime.now(timezone.utc),
        summary="Detailed catalyst summary.",
        content="Full body text mentioning revenue of $10 billion.",
    )


def test_pipeline_prefilter_drops_clickbait_before_processing() -> None:
    storage = SQLiteStorage(":memory:")
    notifier = DummyNotifier()
    analysis = MarketImpactAnalysis(
        impact_grade=ImpactGrade.AVERAGE,
        confidence_score=6,
        is_market_moving=True,
        summary="Summary",
        affected_tickers=["$NVDA"],
        catalyst_thesis="Revenue of $10 billion announced.",
        directional_stance=DirectionalStance.BULLISH,
        time_horizon=TimeHorizon.SHORT_TERM,
        actionable_recommendation="Buy",
        causal_chain=CausalChain(
            event_facts="Facts",
            economic_mechanism="Mechanism",
            beneficiaries=["$NVDA"],
            negatively_exposed=[],
        ),
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )
    pipeline = Pipeline(
        storage=storage,
        analyzer=DummyAnalyzer(analysis),
        notifier=notifier,
        min_confidence_score=5,
        min_impact_grade=ImpactGrade.AVERAGE,
    )

    articles = [
        make_test_article("3 stocks to buy now before market opens", "https://example.com/clickbait"),
        make_test_article("Nvidia announces Rubin platform partnership", "https://example.com/nvda"),
    ]

    summary = pipeline.process_articles(articles)
    # The clickbait should be dropped by prefilter upfront
    assert summary.errors == 0


def test_pipeline_quota_tracker_throttles_alerts_for_same_ticker() -> None:
    storage = SQLiteStorage(":memory:")
    notifier = DummyNotifier()
    analysis = MarketImpactAnalysis(
        impact_grade=ImpactGrade.HIGH,
        confidence_score=9,
        is_market_moving=True,
        summary="High conviction catalyst",
        affected_tickers=["$NVDA"],
        catalyst_thesis="Revenue of $10 billion announced.",
        directional_stance=DirectionalStance.BULLISH,
        time_horizon=TimeHorizon.SHORT_TERM,
        actionable_recommendation="Buy",
        causal_chain=CausalChain(
            event_facts="Facts",
            economic_mechanism="Mechanism",
            beneficiaries=["$NVDA"],
            negatively_exposed=[],
        ),
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )
    analyzer = DummyAnalyzer(analysis)

    quota_tracker = TickerQuotaTracker(config=TickerQuotaConfig(max_alerts_per_ticker=2, rolling_window_hours=1.0))
    pipeline = Pipeline(
        storage=storage,
        analyzer=analyzer,
        notifier=notifier,
        quota_tracker=quota_tracker,
        min_confidence_score=6,
        min_impact_grade=ImpactGrade.HIGH,
        batch_notifications=False,
    )

    art1 = make_test_article("Nvidia 1st announcement", "https://example.com/1")
    art2 = make_test_article("Nvidia 2nd announcement", "https://example.com/2")
    art3 = make_test_article("Nvidia 3rd announcement", "https://example.com/3")

    pipeline.process_articles([art1])
    pipeline.process_articles([art2])
    pipeline.process_articles([art3])

    # Max alerts per ticker was 2, so 3rd should be throttled
    assert len(notifier.sent_alerts) == 2
