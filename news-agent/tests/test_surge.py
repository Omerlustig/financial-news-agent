from datetime import datetime, timedelta, timezone
import pytest

from src.analysis.corroborator import EventCluster
from src.analysis.surge import SurgeVelocityEngine
from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    SurgeLevel,
)


def _make_article(
    title: str,
    source: str,
    published_at: datetime,
    url_id: str,
) -> RawArticle:
    return RawArticle(
        title=title,
        url=f"https://example.com/news/{url_id}",
        source_name=source,
        published_at=published_at,
        summary="Summary test",
    )


def test_detect_surge_breaking_three_publishers_within_30m():
    """Verify >= 3 distinct publishers within 30 minutes triggers SURGE_BREAKING."""
    engine = SurgeVelocityEngine()
    base_time = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)

    art1 = _make_article("Fed emergency rate cut", "Reuters", base_time, "1")
    art2 = _make_article("Federal Reserve cuts rates unexpectedly", "Bloomberg", base_time + timedelta(minutes=10), "2")
    art3 = _make_article("Central bank emergency 50bps cut announced", "CNBC", base_time + timedelta(minutes=25), "3")

    cluster = EventCluster(
        canonical_article=art1,
        articles=[art1, art2, art3],
        sources={"Reuters", "Bloomberg", "CNBC"},
    )

    level = engine.detect_surge(cluster)
    assert level == SurgeLevel.SURGE_BREAKING


def test_detect_surge_elevated_two_publishers_within_60m():
    """Verify >= 2 distinct publishers within 60 minutes triggers ELEVATED."""
    engine = SurgeVelocityEngine()
    base_time = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)

    art1 = _make_article("Major tech merger talks underway", "Reuters", base_time, "1")
    art2 = _make_article("Tech consolidation talks reported", "Bloomberg", base_time + timedelta(minutes=45), "2")

    cluster = EventCluster(
        canonical_article=art1,
        articles=[art1, art2],
        sources={"Reuters", "Bloomberg"},
    )

    level = engine.detect_surge(cluster)
    assert level == SurgeLevel.ELEVATED


def test_detect_surge_normal_spaced_apart():
    """Verify publishers spaced beyond 60 minutes evaluate to NORMAL."""
    engine = SurgeVelocityEngine()
    base_time = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)

    art1 = _make_article("Quarterly earnings preview", "Reuters", base_time, "1")
    art2 = _make_article("Earnings outlook update", "Bloomberg", base_time + timedelta(minutes=90), "2")

    cluster = EventCluster(
        canonical_article=art1,
        articles=[art1, art2],
        sources={"Reuters", "Bloomberg"},
    )

    level = engine.detect_surge(cluster)
    assert level == SurgeLevel.NORMAL


def test_detect_surge_duplicate_publishers_do_not_inflate_surge():
    """Verify multiple articles from the same publisher are counted as 1 distinct publisher."""
    engine = SurgeVelocityEngine()
    base_time = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)

    art1 = _make_article("Earnings preview part 1", "Reuters", base_time, "1")
    art2 = _make_article("Earnings preview part 2", "Reuters", base_time + timedelta(minutes=5), "2")
    art3 = _make_article("Earnings preview part 3", "Reuters", base_time + timedelta(minutes=15), "3")

    cluster = EventCluster(
        canonical_article=art1,
        articles=[art1, art2, art3],
        sources={"Reuters"},
    )

    level = engine.detect_surge(cluster)
    assert level == SurgeLevel.NORMAL


def test_detect_surge_with_recent_articles_parameter():
    """Verify detect_surge incorporates supplementary recent_articles."""
    engine = SurgeVelocityEngine()
    base_time = datetime(2026, 10, 17, 14, 0, tzinfo=timezone.utc)

    art1 = _make_article("Nvidia Blackwell GPU expansion announced", "Reuters", base_time, "1")
    art2 = _make_article("Nvidia Blackwell GPU hyperscaler orders surge", "Bloomberg", base_time + timedelta(minutes=10), "2")
    art3 = _make_article("Nvidia Blackwell production capacity accelerates", "WSJ", base_time + timedelta(minutes=20), "3")

    cluster = EventCluster(
        canonical_article=art1,
        articles=[art1],
        sources={"Reuters"},
    )

    level = engine.detect_surge(cluster, recent_articles=[art2, art3])
    assert level == SurgeLevel.SURGE_BREAKING


def test_detect_surge_empty_cluster():
    """Verify empty or None cluster safely returns NORMAL."""
    engine = SurgeVelocityEngine()
    dummy_art = _make_article("Test", "Source", datetime.now(timezone.utc), "0")
    empty_cluster = EventCluster(canonical_article=dummy_art, articles=[], sources=set())
    assert engine.detect_surge(empty_cluster) == SurgeLevel.NORMAL


def test_apply_surge_metadata_breaking_elevates_score_and_grade():
    """Verify apply_surge_metadata elevates confidence score and LOW/AVERAGE grade on SURGE_BREAKING."""
    engine = SurgeVelocityEngine()
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=4,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$AAPL"],
        affected_sectors=["Technology"],
        summary="Apple announces new product.",
        catalyst_thesis="Incremental consumer adoption.",
    )

    updated = engine.apply_surge_metadata(analysis, SurgeLevel.SURGE_BREAKING)
    assert updated.surge_level == SurgeLevel.SURGE_BREAKING
    assert updated.is_market_moving is True
    assert updated.confidence_score == 8
    assert updated.impact_grade == ImpactGrade.HIGH


def test_apply_surge_metadata_breaking_preserves_higher_score_and_critical_grade():
    """Verify apply_surge_metadata preserves higher score (e.g. 10) and CRITICAL grade."""
    engine = SurgeVelocityEngine()
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=10,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Massive datacenter chip shortage announced.",
        catalyst_thesis="Severe price squeeze and margin expansion.",
    )

    updated = engine.apply_surge_metadata(analysis, SurgeLevel.SURGE_BREAKING)
    assert updated.surge_level == SurgeLevel.SURGE_BREAKING
    assert updated.confidence_score == 10
    assert updated.impact_grade == ImpactGrade.CRITICAL


def test_apply_surge_metadata_elevated_and_normal_retains_base_metrics():
    """Verify ELEVATED and NORMAL update surge_level without modifying score or grade."""
    engine = SurgeVelocityEngine()
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=5,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$MSFT"],
        affected_sectors=["Software"],
        summary="Routine software update.",
        catalyst_thesis="Minor operational update.",
    )

    updated_elevated = engine.apply_surge_metadata(analysis, SurgeLevel.ELEVATED)
    assert updated_elevated.surge_level == SurgeLevel.ELEVATED
    assert updated_elevated.confidence_score == 5
    assert updated_elevated.impact_grade == ImpactGrade.AVERAGE
    assert updated_elevated.is_market_moving is False

    updated_normal = engine.apply_surge_metadata(analysis, SurgeLevel.NORMAL)
    assert updated_normal.surge_level == SurgeLevel.NORMAL
    assert updated_normal.confidence_score == 5
