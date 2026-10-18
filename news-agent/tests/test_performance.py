import datetime
import pytest
from unittest.mock import MagicMock

from src.models import (
    CredibilityTier,
    EventLineageRecord,
    PerformanceMetricsSummary,
    SourceCredibilityRecord,
)
from src.analysis.performance import PerformanceTrackerEngine
from src.storage.sqlite import SQLiteStorage


def test_compute_directional_alpha_bullish():
    engine = PerformanceTrackerEngine()

    # Bullish price increase -> Positive Raw Return, Positive Alpha
    raw_ret, alpha = engine.compute_directional_alpha("BULLISH", initial_price=100.0, current_price=105.0)
    assert raw_ret == pytest.approx(5.0)
    assert alpha == pytest.approx(5.0)
    assert engine.classify_outcome("BULLISH", raw_ret) == "WIN"

    # Bullish price decrease -> Negative Raw Return, Negative Alpha
    raw_ret_down, alpha_down = engine.compute_directional_alpha("BULLISH", initial_price=100.0, current_price=95.0)
    assert raw_ret_down == pytest.approx(-5.0)
    assert alpha_down == pytest.approx(-5.0)
    assert engine.classify_outcome("BULLISH", raw_ret_down) == "LOSS"


def test_compute_directional_alpha_bearish():
    engine = PerformanceTrackerEngine()

    # Bearish price drop -> Negative Raw Return, Positive Alpha (+WIN!)
    raw_ret_drop, alpha_gain = engine.compute_directional_alpha("BEARISH", initial_price=200.0, current_price=190.0)
    assert raw_ret_drop == pytest.approx(-5.0)
    assert alpha_gain == pytest.approx(5.0)
    assert engine.classify_outcome("BEARISH", raw_ret_drop) == "WIN"

    # Bearish price rally -> Positive Raw Return, Negative Alpha (LOSS)
    raw_ret_rally, alpha_loss = engine.compute_directional_alpha("BEARISH", initial_price=200.0, current_price=210.0)
    assert raw_ret_rally == pytest.approx(5.0)
    assert alpha_loss == pytest.approx(-5.0)
    assert engine.classify_outcome("BEARISH", raw_ret_rally) == "LOSS"


def test_friction_threshold_buffer():
    engine = PerformanceTrackerEngine()

    # Within friction threshold (+0.10% on bullish) -> NEUTRAL (neither win nor loss)
    raw_ret, _ = engine.compute_directional_alpha("BULLISH", initial_price=100.0, current_price=100.10)
    assert raw_ret == pytest.approx(0.10)
    assert engine.classify_outcome("BULLISH", raw_ret) == "NEUTRAL"

    # Beyond friction threshold (+0.25% on bullish) -> WIN
    raw_ret_win, _ = engine.compute_directional_alpha("BULLISH", initial_price=100.0, current_price=100.25)
    assert engine.classify_outcome("BULLISH", raw_ret_win) == "WIN"


def test_zero_or_negative_price_safety():
    engine = PerformanceTrackerEngine()

    raw_ret, alpha = engine.compute_directional_alpha("BULLISH", initial_price=0.0, current_price=100.0)
    assert raw_ret == 0.0
    assert alpha == 0.0
    assert engine.classify_outcome("BULLISH", raw_ret) == "NEUTRAL"


def test_publisher_credibility_tier_grading():
    engine = PerformanceTrackerEngine()

    # Tier A: Hit Rate >= 65% and Alpha >= 0.80%
    assert engine.assign_credibility_tier(hit_rate_pct=70.0, avg_alpha_1d_pct=1.2) == CredibilityTier.TIER_A_ALPHA

    # Tier B: Hit Rate 58%, any non-negative alpha
    assert engine.assign_credibility_tier(hit_rate_pct=58.0, avg_alpha_1d_pct=0.3) == CredibilityTier.TIER_B_RELIABLE

    # Tier C: Hit Rate 50%
    assert engine.assign_credibility_tier(hit_rate_pct=50.0, avg_alpha_1d_pct=0.1) == CredibilityTier.TIER_C_NEUTRAL

    # Tier D: Negative alpha or hit rate < 45%
    assert engine.assign_credibility_tier(hit_rate_pct=40.0, avg_alpha_1d_pct=0.5) == CredibilityTier.TIER_D_UNRELIABLE
    assert engine.assign_credibility_tier(hit_rate_pct=60.0, avg_alpha_1d_pct=-0.4) == CredibilityTier.TIER_D_UNRELIABLE


def test_update_forward_returns_via_price_fetcher(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "test_perf_update.db"))
    engine = PerformanceTrackerEngine()

    now = datetime.datetime.now(datetime.timezone.utc)

    # 1. Eligible event for 1d resolution (age 28h)
    ev1 = EventLineageRecord(
        event_id="evt_nvda_run",
        title="NVIDIA secures cloud supply contract",
        primary_ticker="$NVDA",
        first_seen=now - datetime.timedelta(hours=28),
        last_updated=now - datetime.timedelta(hours=28),
        source_count=1,
        all_sources=["Reuters"],
        opportunity_score=8.5,
        directional_stance="BULLISH",
        initial_price=100.0,
        subsequent_return_1d=None,
    )
    storage.record_event_lineage(ev1)

    # Mock price fetcher returning 106.0 for $NVDA (+6.0%)
    mock_fetcher = MagicMock()
    mock_fetcher.side_effect = lambda t: 106.0 if "NVDA" in t else None

    counts = engine.update_forward_returns(storage, price_fetcher=mock_fetcher, min_age_hours_1d=24)
    assert counts["updated_1d"] == 1

    updated_ev = storage.get_event_lineage("evt_nvda_run")
    assert updated_ev is not None
    assert updated_ev.subsequent_return_1d == pytest.approx(6.0)

    storage.close()


def test_calculate_performance_metrics_and_publisher_scorecard(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "test_perf_metrics.db"))
    engine = PerformanceTrackerEngine()

    now = datetime.datetime.now(datetime.timezone.utc)

    # Event 1: FT report on MSFT, Bearish, dropped from 400 to 380 (-5% raw, +5% alpha) -> WIN
    ev1 = EventLineageRecord(
        event_id="evt_msft_bear",
        title="Microsoft cancels datacenter expansion",
        primary_ticker="$MSFT",
        first_seen=now - datetime.timedelta(days=2),
        last_updated=now - datetime.timedelta(days=1),
        source_count=2,
        all_sources=["Financial Times", "Bloomberg"],
        event_quality_score=9.0,
        opportunity_score=8.0,
        directional_stance="BEARISH",
        initial_price=400.0,
        subsequent_return_1d=-5.0,
        subsequent_return_7d=-7.0,
    )
    storage.record_event_lineage(ev1)

    # Event 2: Reuters report on NVDA, Bullish, rose from 100 to 103 (+3% raw, +3% alpha) -> WIN
    ev2 = EventLineageRecord(
        event_id="evt_nvda_bull",
        title="Nvidia accelerates GPU roadmaps",
        primary_ticker="$NVDA",
        first_seen=now - datetime.timedelta(days=3),
        last_updated=now - datetime.timedelta(days=2),
        source_count=1,
        all_sources=["Reuters"],
        event_quality_score=8.5,
        opportunity_score=7.5,
        directional_stance="BULLISH",
        initial_price=100.0,
        subsequent_return_1d=3.0,
        subsequent_return_7d=6.0,
    )
    storage.record_event_lineage(ev2)

    # Event 3: Rumor Blog on TSLA, Bullish, dropped from 200 to 190 (-5% raw, -5% alpha) -> LOSS
    ev3 = EventLineageRecord(
        event_id="evt_tsla_loss",
        title="Rumor: Tesla robotaxi launch early",
        primary_ticker="$TSLA",
        first_seen=now - datetime.timedelta(days=2),
        last_updated=now - datetime.timedelta(days=1),
        source_count=1,
        all_sources=["Unverified Blog"],
        event_quality_score=4.0,
        opportunity_score=6.5,
        directional_stance="BULLISH",
        initial_price=200.0,
        subsequent_return_1d=-5.0,
        subsequent_return_7d=-8.0,
    )
    storage.record_event_lineage(ev3)

    summary = engine.calculate_performance_metrics(storage, min_opportunity=6.0)

    assert summary.total_catalysts == 3
    assert summary.resolved_1d == 3
    assert summary.wins_1d == 2
    assert summary.losses_1d == 1
    assert summary.hit_rate_1d_pct == pytest.approx(66.67, rel=1e-2)
    # Alpha 1d: (+5.0 + 3.0 - 5.0) / 3 = 1.0%
    assert summary.avg_alpha_1d_pct == pytest.approx(1.0, rel=1e-2)

    # Check publisher scorecards
    sources_map = {s.source_name: s for s in summary.source_scorecard}
    assert "Financial Times" in sources_map
    ft_record = sources_map["Financial Times"]
    assert ft_record.hit_rate_pct == 100.0
    assert ft_record.avg_alpha_1d_pct == pytest.approx(5.0)
    assert ft_record.credibility_tier == CredibilityTier.TIER_A_ALPHA

    assert "Unverified Blog" in sources_map
    blog_record = sources_map["Unverified Blog"]
    assert blog_record.hit_rate_pct == 0.0
    assert blog_record.avg_alpha_1d_pct == pytest.approx(-5.0)
    assert blog_record.credibility_tier == CredibilityTier.TIER_D_UNRELIABLE

    # Test filtering by ticker
    nvda_summary = engine.calculate_performance_metrics(storage, ticker="$NVDA")
    assert nvda_summary.total_catalysts == 1
    assert nvda_summary.wins_1d == 1
    assert nvda_summary.hit_rate_1d_pct == 100.0

    storage.close()
