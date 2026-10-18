"""Unit tests for TrendEngine, buzz ratio, sentiment velocity, and TrendVerdict."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
import pytest

from src.analysis.trend import TrendEngine
from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    SectorTrendMetrics,
    SentimentLabel,
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
    # Normal volume (7 articles over 7 days) with steady bearish sentiment
    recent_7d = [
        {"sentiment_score": -0.3, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ] + [
        {"sentiment_score": -0.3, "created_at": (now - timedelta(days=i)).isoformat(), "id": 10 + i}
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


def test_sentiment_acceleration_inflection_reversal(trend_engine: TrendEngine):
    """Test sentiment acceleration on inflection from negative to positive velocity.

    Window 2 (48h-72h): sentiment 0.5
    Window 1 (24h-48h): sentiment 0.2 -> V_prior = 0.2 - 0.5 = -0.3
    Window 0 (0-24h): sentiment 0.4 -> V_current = 0.4 - 0.2 = +0.2
    Acceleration = 0.2 - (-0.3) = +0.5
    Verdict: BULLISH_REVERSAL (A >= 0.25 and avg_sentiment > 0)
    """
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.4, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = [
        {"sentiment_score": 0.4, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
        {"sentiment_score": 0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
        {"sentiment_score": 0.5, "created_at": (now - timedelta(hours=54)).isoformat(), "id": 3},
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$NVDA")
    assert metrics.sentiment_velocity == pytest.approx(0.2, abs=0.001)
    assert metrics.sentiment_acceleration == pytest.approx(0.5, abs=0.001)
    assert metrics.trend_verdict == TrendVerdict.BULLISH_REVERSAL


def test_sentiment_acceleration_bearish_breakdown(trend_engine: TrendEngine):
    """Test sentiment acceleration on breakdown from positive to negative velocity.

    Window 2 (48h-72h): sentiment -0.1
    Window 1 (24h-48h): sentiment 0.2 -> V_prior = 0.2 - (-0.1) = +0.3
    Window 0 (0-24h): sentiment -0.2 -> V_current = -0.2 - 0.2 = -0.4
    Acceleration = -0.4 - 0.3 = -0.7
    Verdict: BEARISH_BREAKDOWN (A <= -0.25 and avg_sentiment < 0)
    """
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": -0.2, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = [
        {"sentiment_score": -0.2, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
        {"sentiment_score": 0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
        {"sentiment_score": -0.1, "created_at": (now - timedelta(hours=54)).isoformat(), "id": 3},
    ]

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$INTC")
    assert metrics.sentiment_velocity == pytest.approx(-0.4, abs=0.001)
    assert metrics.sentiment_acceleration == pytest.approx(-0.7, abs=0.001)
    assert metrics.trend_verdict == TrendVerdict.BEARISH_BREAKDOWN


def test_divergence_bearish_panic_liquidation(trend_engine: TrendEngine):
    """Test volume buzz spike with negative sentiment and negative velocity triggers BEARISH_PANIC_LIQUIDATION."""
    now = datetime.now(timezone.utc)
    # 6 articles in last 24h, all negative
    recent_24h = [
        {"sentiment_score": -0.5, "created_at": (now - timedelta(hours=i)).isoformat(), "id": i}
        for i in range(6)
    ]
    # Prior window has higher sentiment (-0.2) so velocity is negative (-0.3)
    prior = [
        {"sentiment_score": -0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 10},
    ]
    recent_7d = recent_24h + prior

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$CRWD")
    assert metrics.buzz_ratio >= 1.8
    assert metrics.sentiment_velocity <= -0.1
    assert metrics.is_divergent is True
    assert metrics.divergence_type == "BEARISH_PANIC_LIQUIDATION"


def test_divergence_bullish_accumulation(trend_engine: TrendEngine):
    """Test low buzz with strong positive sentiment and positive velocity triggers BULLISH_ACCUMULATION_DIVERGENCE."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.45, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    # 7 articles over 7 days -> buzz_ratio == 1.0 (<= 1.3)
    prior = [
        {"sentiment_score": 0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
    ] + [
        {"sentiment_score": 0.2, "created_at": (now - timedelta(days=i)).isoformat(), "id": 10 + i}
        for i in range(2, 7)
    ]
    recent_7d = recent_24h + prior

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$SMH")
    assert metrics.buzz_ratio <= 1.3
    assert metrics.sentiment_velocity >= 0.15
    assert metrics.is_divergent is True
    assert metrics.divergence_type == "BULLISH_ACCUMULATION_DIVERGENCE"


def test_divergence_normal_non_divergent(trend_engine: TrendEngine):
    """Verify normal conditions result in is_divergent=False and divergence_type=None."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sentiment_score": 0.1, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = list(recent_24h)

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.side_effect = lambda ticker, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    metrics = trend_engine.calculate_ticker_trend(mock_storage, "$SPY")
    assert metrics.is_divergent is False
    assert metrics.divergence_type is None


def test_calculate_sector_trend_mock(trend_engine: TrendEngine):
    """Verify calculate_sector_trend with mock storage across 3 historical windows."""
    now = datetime.now(timezone.utc)
    recent_24h = [
        {"sector": "Semiconductors", "sentiment_score": 0.6, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
    ]
    recent_7d = [
        {"sector": "Semiconductors", "sentiment_score": 0.6, "created_at": (now - timedelta(hours=2)).isoformat(), "id": 1},
        {"sector": "Semiconductors", "sentiment_score": 0.2, "created_at": (now - timedelta(hours=30)).isoformat(), "id": 2},
        {"sector": "Semiconductors", "sentiment_score": 0.5, "created_at": (now - timedelta(hours=54)).isoformat(), "id": 3},
    ]

    mock_storage = MagicMock()
    mock_storage.get_sector_sentiment_window.side_effect = lambda sector, hours: (
        recent_24h if hours == 24 else recent_7d
    )

    sector_metrics = trend_engine.calculate_sector_trend(mock_storage, "Semiconductors")
    assert isinstance(sector_metrics, SectorTrendMetrics)
    assert sector_metrics.sector == "Semiconductors"
    assert sector_metrics.avg_sentiment == pytest.approx(0.6, abs=0.001)
    assert sector_metrics.sentiment_velocity == pytest.approx(0.4, abs=0.001)
    # V_prior = 0.2 - 0.5 = -0.3, V_current = 0.4 => A = 0.4 - (-0.3) = 0.7
    assert sector_metrics.sentiment_acceleration == pytest.approx(0.7, abs=0.001)
    assert sector_metrics.sentiment_label == SentimentLabel.VERY_BULLISH
    assert sector_metrics.trend_verdict == TrendVerdict.STRONG_BULLISH
    mock_storage.get_sector_sentiment_window.assert_any_call("Semiconductors", hours=24)
    mock_storage.get_sector_sentiment_window.assert_any_call("Semiconductors", hours=168)


def test_calculate_sector_trend_edge_cases(trend_engine: TrendEngine):
    """Verify calculate_sector_trend handles empty input or None storage gracefully."""
    empty_res = trend_engine.calculate_sector_trend(None, "")  # type: ignore[arg-type]
    assert empty_res.sector == ""
    assert empty_res.avg_sentiment == 0.0
    assert empty_res.trend_verdict == TrendVerdict.NEUTRAL_CONSOLIDATION

    mock_storage = MagicMock()
    mock_storage.get_sector_sentiment_window.return_value = []
    res_empty_db = trend_engine.calculate_sector_trend(mock_storage, "Energy")
    assert res_empty_db.sector == "Energy"
    assert res_empty_db.buzz_ratio == 1.0
    assert res_empty_db.sentiment_velocity == 0.0


def test_calculate_sector_trend_sqlite_integration(tmp_path, trend_engine: TrendEngine):
    """Integration test validating calculate_sector_trend against real SQLiteStorage."""
    db_file = str(tmp_path / "sector_trend_integration.db")
    storage = SQLiteStorage(db_file)
    now = datetime.now(timezone.utc)

    import sqlite3
    conn = sqlite3.connect(db_file)
    # Prior 2 window: 54h ago
    time_p2 = now - timedelta(hours=54)
    # Prior 1 window: 30h ago
    time_p1 = now - timedelta(hours=30)
    # Recent window: 2h ago
    time_rec = now - timedelta(hours=2)

    conn.execute(
        """
        INSERT INTO sector_sentiment_history (sector, sentiment_score, sentiment_label, article_hash, published_at, created_at)
        VALUES ('Semiconductors', 0.5, 'BULLISH', 'h_p2', ?, ?)
        """,
        (time_p2.isoformat(), time_p2.isoformat()),
    )
    conn.execute(
        """
        INSERT INTO sector_sentiment_history (sector, sentiment_score, sentiment_label, article_hash, published_at, created_at)
        VALUES ('Semiconductors', 0.2, 'BULLISH', 'h_p1', ?, ?)
        """,
        (time_p1.isoformat(), time_p1.isoformat()),
    )
    conn.execute(
        """
        INSERT INTO sector_sentiment_history (sector, sentiment_score, sentiment_label, article_hash, published_at, created_at)
        VALUES ('Semiconductors', 0.4, 'BULLISH', 'h_rec', ?, ?)
        """,
        (time_rec.isoformat(), time_rec.isoformat()),
    )
    conn.commit()
    conn.close()

    metrics = trend_engine.calculate_sector_trend(storage, "Semiconductors")
    assert metrics.sector == "Semiconductors"
    assert metrics.article_count == 1
    assert metrics.avg_sentiment == pytest.approx(0.4, abs=0.001)
    assert metrics.sentiment_velocity == pytest.approx(0.2, abs=0.001)
    assert metrics.sentiment_acceleration == pytest.approx(0.5, abs=0.001)
    assert metrics.sentiment_label == SentimentLabel.BULLISH
    assert metrics.trend_verdict == TrendVerdict.BULLISH_REVERSAL

    storage.close()


def test_enrich_analysis_with_sector_trends(trend_engine: TrendEngine):
    """Verify enrich_analysis_with_sector_trends enriches multiple sectors."""
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors", "Cloud Computing"],
        summary="AI acceleration announcement.",
        catalyst_thesis="Semiconductor boom expands datacenter cloud spending.",
        sentiment=SentimentScore(score=0.6, label="VERY_BULLISH"),
    )

    mock_storage = MagicMock()
    mock_storage.get_sector_sentiment_window.return_value = []

    enriched = trend_engine.enrich_analysis_with_sector_trends(mock_storage, analysis)
    assert len(enriched.sector_trends) == 2
    assert enriched.sector_trends[0].sector == "Semiconductors"
    assert enriched.sector_trends[1].sector == "Cloud Computing"
    assert enriched.sector_trends[0].avg_sentiment == pytest.approx(0.6, abs=0.001)


def test_enrich_analysis_with_trend_enriches_both_ticker_and_sectors(trend_engine: TrendEngine):
    """Verify enrich_analysis_with_trend populates both ticker trend and sector_trends."""
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="NVIDIA announces next-generation chip.",
        catalyst_thesis="Substantial revenue upside.",
        sentiment=SentimentScore(score=0.7, label="VERY_BULLISH"),
    )

    mock_storage = MagicMock()
    mock_storage.get_ticker_sentiment_window.return_value = []
    mock_storage.get_sector_sentiment_window.return_value = []

    enriched = trend_engine.enrich_analysis_with_trend(mock_storage, analysis)
    assert enriched.trend is not None
    assert isinstance(enriched.trend, TrendMetrics)
    assert len(enriched.sector_trends) == 1
    assert isinstance(enriched.sector_trends[0], SectorTrendMetrics)
    assert enriched.sector_trends[0].sector == "Semiconductors"

