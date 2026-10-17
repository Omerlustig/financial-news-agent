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
    assert isinstance(settings.filter.min_impact_grade, ImpactGrade)
    assert settings.filter.min_impact_grade == ImpactGrade.HIGH
    assert settings.llm.model
    # Unset telegram env vars should resolve to None rather than empty string
    assert settings.telegram.bot_token is None or isinstance(settings.telegram.bot_token, str)


def test_affected_tickers_normalization():
    # String with comma-separated tickers
    analysis1 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers="NVDA, AAPL, $msft",  # string input
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis1.affected_tickers == ["$NVDA", "$AAPL", "$MSFT"]

    # Single string ticker
    analysis2 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers="tsm",  # single string
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis2.affected_tickers == ["$TSM"]

    # List input
    analysis3 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["nvda", "$aapl", "AMZN"],
        affected_sectors=["Tech"],
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis3.affected_tickers == ["$NVDA", "$AAPL", "$AMZN"]


def test_min_impact_grade_validation(tmp_path):
    from pydantic import ValidationError

    # Valid string or enum parses to ImpactGrade
    settings = Settings.model_validate({
        "filter": {"min_impact_grade": "CRITICAL"}
    })
    assert settings.filter.min_impact_grade == ImpactGrade.CRITICAL

    # MEDIUM alias maps to AVERAGE in Settings and FilterSettings
    settings_medium = Settings.model_validate({
        "filter": {"min_impact_grade": "MEDIUM"}
    })
    assert settings_medium.filter.min_impact_grade == ImpactGrade.AVERAGE

    settings_medium_lower = Settings.model_validate({
        "filter": {"min_impact_grade": "medium"}
    })
    assert settings_medium_lower.filter.min_impact_grade == ImpactGrade.AVERAGE

    # Direct enum call converts MEDIUM
    assert ImpactGrade("MEDIUM") == ImpactGrade.AVERAGE
    assert ImpactGrade("medium") == ImpactGrade.AVERAGE

    # MarketImpactAnalysis converts MEDIUM alias to AVERAGE
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=5,
        impact_grade="MEDIUM",
        affected_tickers=["$AAPL"],
        affected_sectors=["Tech"],
        summary="Medium impact event.",
        catalyst_thesis="Average market catalyst.",
    )
    assert analysis.impact_grade == ImpactGrade.AVERAGE
    assert analysis.impact_grade is ImpactGrade.AVERAGE

    # Invalid impact grade raises ValidationError
    with pytest.raises(ValidationError):
        Settings.model_validate({
            "filter": {"min_impact_grade": "INVALID_GRADE"}
        })


def test_telegram_empty_env_to_none():
    # If env var is unset in yaml expansion, it should evaluate to None
    settings = Settings.model_validate({
        "telegram": {"bot_token": "", "chat_id": ""}
    })
    assert settings.telegram.bot_token is None
    assert settings.telegram.chat_id is None
