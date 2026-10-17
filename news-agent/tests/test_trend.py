"""Unit tests for TrendEngine, buzz ratio, sentiment velocity, and TrendVerdict."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from src.analysis.trend import TrendEngine
from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    SentimentScore,
    TrendMetrics,
    TrendVerdict,
)
from src.storage.sqlite import SQLiteStorage


@pytest.fixture
def trend_engine() -> TrendEngine:
    return TrendEngine()


def test_buzz_ratio_empty_database(trend_engine: TrendEngine):
    """Verify buzz ratio calculation when no historical records exist."""
    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.return_value = []

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$NVDA", current_sentiment=None)
    assert metrics.buzz_ratio == 1.0
    assert metrics.articles_24h_count == 1
    assert metrics.trend_verdict == TrendVerdict.NEUTRAL_CONSOLIDATION


def test_buzz_ratio_empty_database_with_current_sentiment(trend_engine: TrendEngine):
    """Verify buzz ratio with 1 incoming article on an empty database."""
    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.return_value = []

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$AAPL", current_sentiment=0.5)
    assert metrics.buzz_ratio == 1.0
    assert metrics.articles_24h_count == 1


def test_buzz_ratio_normal_steady_volume(trend_engine: TrendEngine):
    """Verify normal volume (1 article/day over 7 days) yields buzz_ratio == 1.0."""
    now = datetime.now(timezone.utc)
    recent_24h = [{"sentiment_score": 0.2, "created_at": (now - timedelta(hours=2)).isoformat()}]
    recent_7d = [
        {"sentiment_score": 0.2, "created_at": (now - timedelta(days=i)).isoformat()}
        for i in range(7)
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "MSFT")
    # count_24h = 1, baseline_daily_avg = max(7/7, 1.0) = 1.0 => buzz_ratio = 1.0
    assert metrics.buzz_ratio == 1.0
    assert metrics.articles_24h_count == 1


def test_buzz_ratio_spike_volume(trend_engine: TrendEngine):
    """Verify volume surge creates high buzz ratio."""
    now = datetime.now(timezone.utc)
    # 14 articles published today
    recent_24h = [
        {"sentiment_score": 0.3, "created_at": (now - timedelta(hours=1)).isoformat(), "id": i}
        for i in range(14)
    ]
    recent_7d = list(recent_24h)

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$TSLA")
    # count_24h = 14, baseline_daily_avg = max(14/7, 1.0) = 2.0 => buzz_ratio = 7.0
    assert metrics.buzz_ratio == 7.0
    assert metrics.articles_24h_count == 14


def test_sentiment_velocity_improving_sentiment(trend_engine: TrendEngine):
    """Test velocity calculation when sentiment accelerates upward."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.6, "created_at": (now - timedelta(hours=3)).isoformat(), "id": 1},
        {"sentiment_score": 0.8, "created_at": (now - timedelta(hours=6)).isoformat(), "id": 2},
    ]
    prior_24h_48h = [
        {"sentiment_score": 0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 3},
        {"sentiment_score": 0.2, "created_at": (now - timedelta(hours=36)).isoformat(), "id": 4},
    ]
    recent_7d = recent_24h + prior_24h_48h

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$NVDA")
    # avg_24h = 0.7, avg_prior = 0.2 => velocity = 0.5
    assert metrics.sentiment_velocity == pytest.approx(0.5, abs=0.001)


def test_sentiment_velocity_deteriorating_sentiment(trend_engine: TrendEngine):
    """Test velocity calculation when sentiment rapidly deteriorates."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": -0.4, "created_at": (now - timedelta(hours=4)).isoformat(), "id": 1},
    ]
    prior_24h_48h = [
        {"sentiment_score": 0.4, "created_at": (now - timedelta(hours=28)).isoformat(), "id": 2},
    ]
    recent_7d = recent_24h + prior_24h_48h

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$META")
    # avg_24h = -0.4, avg_prior = 0.4 => velocity = -0.8
    assert metrics.sentiment_velocity == pytest.approx(-0.8, abs=0.001)


def test_sentiment_velocity_insufficient_history(trend_engine: TrendEngine):
    """Test sentiment velocity defaults to 0.0 when no prior window history exists."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.5, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = list(recent_24h)

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$GOOGL")
    assert metrics.sentiment_velocity == 0.0


def test_trend_verdict_strong_bullish_via_buzz(trend_engine: TrendEngine):
    """STRONG_BULLISH: 24h avg sentiment >= 0.45 AND buzz_ratio >= 1.5."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.6, "created_at": (now - timedelta(hours=1)).isoformat(), "id": i}
        for i in range(4)
    ]
    recent_7d = list(recent_24h)  # count_24h=4, 7d=4, daily_avg=max(4/7, 1)=1.0, buzz=4.0 >= 1.5

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$AMD")
    assert metrics.trend_verdict == TrendVerdict.STRONG_BULLISH


def test_trend_verdict_strong_bullish_via_velocity(trend_engine: TrendEngine):
    """STRONG_BULLISH: 24h avg sentiment >= 0.45 AND sentiment_velocity > 0.2."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.5, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    # 7 articles over 7 days so buzz_ratio == 1.0 (< 1.5)
    recent_7d = [
        {"sentiment_score": 0.5, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
        {"sentiment_score": 0.1, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
    ] + [
        {"sentiment_score": 0.1, "created_at": (now - timedelta(days=i)).isoformat(), "id": 10 + i}
        for i in range(3, 8)
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$NVDA")
    # avg_24h = 0.5, avg_prior = 0.1, velocity = 0.4 > 0.2
    assert metrics.sentiment_velocity == pytest.approx(0.4, abs=0.001)
    assert metrics.trend_verdict == TrendVerdict.STRONG_BULLISH


def test_trend_verdict_moderate_bullish(trend_engine: TrendEngine):
    """MODERATE_BULLISH: 24h avg sentiment >= 0.15 AND sentiment_velocity >= -0.05."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.25, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = [
        {"sentiment_score": 0.25, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
        {"sentiment_score": 0.25, "created_at": (now - timedelta(hours=28)).isoformat(), "id": 2},
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$INTC")
    assert metrics.trend_verdict == TrendVerdict.MODERATE_BULLISH


def test_trend_verdict_panic_capitulation(trend_engine: TrendEngine):
    """PANIC_CAPITULATION: 24h avg sentiment <= -0.45 AND buzz_ratio >= 1.5."""
    now = datetime.now(timezone.utc)
    # Heavy negative news volume surge
    recent_24h = [
        {"sentiment_score": -0.6, "created_at": (now - timedelta(hours=i)).isoformat(), "id": i}
        for i in range(5)
    ]
    recent_7d = list(recent_24h)

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$CRWD")
    assert metrics.buzz_ratio >= 1.5
    assert metrics.trend_verdict == TrendVerdict.PANIC_CAPITULATION


def test_trend_verdict_moderate_bearish(trend_engine: TrendEngine):
    """MODERATE_BEARISH: 24h avg sentiment <= -0.15 (without extreme panic buzz)."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": -0.3, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    # Normal volume (7 articles over 7 days)
    recent_7d = [
        {"sentiment_score": -0.3, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ] + [
        {"sentiment_score": 0.0, "created_at": (now - timedelta(days=i)).isoformat(), "id": 10 + i}
        for i in range(1, 7)
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$BA")
    assert metrics.trend_verdict == TrendVerdict.MODERATE_BEARISH


def test_trend_verdict_neutral_consolidation(trend_engine: TrendEngine):
    """NEUTRAL_CONSOLIDATION: Flat or mixed sentiment."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.05, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = list(recent_24h)

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$SPY")
    assert metrics.trend_verdict == TrendVerdict.NEUTRAL_CONSOLIDATION


def test_trend_verdict_divergent_bullish_drops_to_neutral(trend_engine: TrendEngine):
    """Positive sentiment but sharp negative velocity (<-0.05) drops to NEUTRAL."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.20, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    prior_24h_48h = [
        {"sentiment_score": 0.40, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
    ]
    recent_7d = recent_24h + prior_24h_48h

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$MSFT")
    # avg_24h = 0.20, velocity = -0.20 (< -0.05) -> not MODERATE_BULLISH -> NEUTRAL
    assert metrics.sentiment_velocity == pytest.approx(-0.20, abs=0.001)
    assert metrics.trend_verdict == TrendVerdict.NEUTRAL_CONSOLIDATION


def test_enrich_analysis_with_trend(trend_engine: TrendEngine):
    """Verify enrich_analysis_with_trend attaches TrendMetrics to MarketImpactAnalysis."""
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA", "$AMD"],
        affected_sectors=["Semiconductors"],
        summary="NVIDIA reveals breakthrough next-gen AI chip.",
        catalyst_thesis="Substantial revenue acceleration expected from datacenter demand.",
        sentiment=SentimentScore(score=0.7, label="VERY_BULLISH"),
    )

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.return_value = []

    enriched = trend_engine.enrich_analysis_with_trend(mock_storage, analysis)
    assert enriched.trend is not None
    assert isinstance(enriched.trend, TrendMetrics)
    assert enriched.trend.buzz_ratio >= 1.0
    mock_storage.get_ticker_sentiment_window.assert_any_call("NVDA", hours=24)


def test_enrich_analysis_with_empty_tickers(trend_engine: TrendEngine):
    """Verify analysis without affected tickers is unmodified."""
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=["General"],
        summary="Macroeconomic overview without direct ticker exposure.",
        catalyst_thesis="General market interest rate trends.",
    )

    mock_storage = MagicMock()
    enriched = trend_engine.enrich_analysis_with_trend(mock_storage, analysis)
    assert enriched.trend is None
    mock_storage.get_ticker_sentiment_window.assert_not_called()


def test_enrich_analysis_with_none_storage(trend_engine: TrendEngine):
    """Verify graceful handling when storage is None."""
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=7,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$AAPL"],
        summary="Apple files 8-K.",
        catalyst_thesis="Material agreement signed.",
    )
    enriched = trend_engine.enrich_analysis_with_trend(None, analysis)  # type: ignore[arg-type]
    assert enriched.trend is None


def test_ticker_trend_with_sqlite_integration(tmp_path, trend_engine: TrendEngine):
    """Integration test validating TrendEngine with real SQLiteStorage."""
    db_file = str(tmp_path / "trend_integration.db")
    storage = SQLiteStorage(db_file)
    now = datetime.now(timezone.utc)

    import sqlite3
    # 1. Record an article 30 hours ago (prior window) with matching created_at
    prior_time = now - timedelta(hours=30)
    conn = sqlite3.connect(db_file)
    conn.execute(
        """
        INSERT INTO ticker_sentiment_history (ticker, sentiment_score, sentiment_label, article_hash, published_at, created_at)
        VALUES ('$NVDA', 0.2, 'BULLISH', 'hash_nvda_prior', ?, ?)
        """,
        (prior_time.isoformat(), prior_time.isoformat()),
    )
    conn.commit()
    conn.close()

    # 2. Record 2 articles in the last 2 hours
    storage.record_ticker_sentiment(
        ticker="$NVDA",
        sentiment_score=0.8,
        sentiment_label="VERY_BULLISH",
        article_hash="hash_nvda_recent_1",
        published_at=now - timedelta(hours=2),
    )
    storage.record_ticker_sentiment(
        ticker="$NVDA",
        sentiment_score=0.6,
        sentiment_label="BULLISH",
        article_hash="hash_nvda_recent_2",
        published_at=now - timedelta(hours=1),
    )

    # 3. Calculate trend for NVDA
    metrics = trend_engine.calculate_ticker_trend(storage, "$NVDA")
    assert metrics.articles_24h_count == 2
    # 24h avg = 0.7, prior = 0.2 => velocity = 0.5
    assert metrics.sentiment_velocity == pytest.approx(0.5, abs=0.001)
    # 24h count = 2, 7d count = 3, baseline = max(3/7, 1.0) = 1.0 => buzz_ratio = 2.0
    assert metrics.buzz_ratio == 2.0
    assert metrics.trend_verdict == TrendVerdict.STRONG_BULLISH

    storage.close()


def test_pipeline_process_articles_enriches_trend():
    """Verify that Pipeline.process_articles executes TrendEngine.enrich_analysis_with_trend."""
    from src.models import RawArticle
    from src.pipeline import Pipeline

    mock_storage = MagicMock()
    mock_storage.filter_unseen.side_effect = lambda arts: arts
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_trend_engine = MagicMock()

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        summary="NVIDIA revenue beats estimates.",
        catalyst_thesis="Datacenter AI revenue acceleration.",
        sentiment=SentimentScore(score=0.7, label="VERY_BULLISH"),
    )
    mock_analyzer.analyze.return_value = analysis

    trend_result = TrendMetrics(
        buzz_ratio=2.5,
        sentiment_velocity=0.3,
        trend_verdict=TrendVerdict.STRONG_BULLISH,
        articles_24h_count=5,
    )
    mock_trend_engine.enrich_analysis_with_trend.side_effect = lambda st, an: (
        setattr(an, "trend", trend_result) or an
    )

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        trend_engine=mock_trend_engine,
        alert_on_all=True,
    )

    article = RawArticle(
        title="NVIDIA Q3 Earnings Beat",
        url="https://example.com/nvda",
        source_name="TechDaily",
    )

    summary = pipeline.process_articles([article])
    assert summary.analyzed == 1
    mock_trend_engine.enrich_analysis_with_trend.assert_called_once_with(mock_storage, analysis)
    assert analysis.trend == trend_result
    # Storage should be passed the enriched analysis
    mock_storage.record_article.assert_called()
    call_args = mock_storage.record_article.call_args[0]
    assert call_args[1].trend == trend_result

