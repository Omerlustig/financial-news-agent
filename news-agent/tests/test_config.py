import pytest
from pathlib import Path

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.config import load_settings, load_sources, Settings, SourceConfig


def test_raw_article_hash():
    a1 = RawArticle(
        title="Chip export curbs",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Test",
    )
    a2 = RawArticle(
        title="Chip export curbs",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Different summary",
    )
    # Deduplication hash is based on url + title
    assert a1.content_hash == a2.content_hash

    # Different url or title must yield different hash
    a3 = RawArticle(
        title="Different headline",
        url="https://news.com/1",
        source_name="Reuters",
        published_at=None,
        summary="Test",
    )
    assert a1.content_hash != a3.content_hash


def test_market_impact_analysis_model():
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA", "$TSM"],
        affected_sectors=["Semiconductors"],
        summary="US imposes stricter curbs on GPU exports to key foreign markets.",
        catalyst_thesis="Direct revenue restriction on high-margin datacenter sales.",
    )
    assert analysis.is_market_moving is True
    assert analysis.confidence_score == 9
    assert analysis.impact_grade == ImpactGrade.CRITICAL
    assert "$NVDA" in analysis.affected_tickers

    # Confidence score must be 1 to 10
    with pytest.raises(Exception):
        MarketImpactAnalysis(
            is_market_moving=True,
            confidence_score=11,  # Invalid: > 10
            impact_grade=ImpactGrade.CRITICAL,
            affected_tickers=[],
            affected_sectors=[],
            summary="Invalid",
            catalyst_thesis="Invalid",
        )


def test_load_default_sources(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text("""
sources:
  - name: "Reuters Business"
    type: "rss"
    url: "https://feeds.reuters.com/reuters/businessNews"
    enabled: true
""")
    sources = load_sources(str(sources_yaml))
    assert len(sources) == 1
    assert sources[0].name == "Reuters Business"
    assert sources[0].type == "rss"
    assert sources[0].enabled is True


def test_load_actual_sources_yaml():
    # Verify the real config/sources.yaml contains top financial feeds
    sources = load_sources()
    assert len(sources) >= 5
    names = [s.name.lower() for s in sources]
    assert any("reuters" in name for name in names)
    assert any("cnbc" in name for name in names)
    assert any("yahoo" in name for name in names)
    assert any("marketwatch" in name for name in names)
    assert any("sec" in name for name in names)


def test_load_actual_settings_yaml():
    settings = load_settings()
    assert settings.agent.polling_interval_minutes > 0
    assert settings.filter.min_confidence_score >= 1
    assert settings.filter.min_impact_grade in [e.value for e in ImpactGrade]
    assert settings.llm.model
