from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.storage.sqlite import SQLiteStorage
from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade, EventStatus, DirectionalStance


@pytest.fixture
def temp_cli_db(tmp_path: Path):
    db_file = tmp_path / "test_cli.db"
    storage = SQLiteStorage(str(db_file))
    yield storage, str(db_file)
    storage.close()


def test_heatmap_command_empty(temp_cli_db):
    """Verify heatmap command informs user gracefully when no sentiment records exist."""
    _, db_path = temp_cli_db
    runner = CliRunner()
    result = runner.invoke(app, ["heatmap", "--hours", "24", "--db", db_path])

    assert result.exit_code == 0
    assert "No sector or ticker sentiment records found" in result.stdout


def test_heatmap_command_populated(temp_cli_db):
    """Verify heatmap command renders Sector News and Ticker News sentiment tables."""
    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    # Record sector sentiments
    storage.record_sector_sentiment("Semiconductors", 0.80, "VERY_BULLISH", "h_semi", published_at=now)
    storage.record_sector_sentiment("Aviation", -0.45, "BEARISH", "h_aero", published_at=now)
    storage.record_sector_sentiment("Financials", 0.0, "NEUTRAL", "h_fin", published_at=now)

    # Record ticker sentiments
    storage.record_ticker_sentiment("$NVDA", 0.85, "VERY_BULLISH", "h_nvda", published_at=now)
    storage.record_ticker_sentiment("$DAL", -0.50, "BEARISH", "h_dal", published_at=now)

    runner = CliRunner()
    result = runner.invoke(app, ["heatmap", "--hours", "24", "--db", db_path])

    assert result.exit_code == 0
    assert "Sector News Sentiment" in result.stdout
    assert "Semiconductors" in result.stdout
    assert "Aviation" in result.stdout
    assert "Ticker News Sentiment" in result.stdout
    assert "$NVDA" in result.stdout
    assert "$DAL" in result.stdout


def test_scan_command_alias(temp_cli_db, tmp_path: Path):
    """Verify scan command executes a complete cycle, summary table, and actionable bottom line."""
    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(f"""
agent:
  database_path: "{db_path}"
filter:
  min_confidence_score: 6
  min_impact_grade: "HIGH"
  alert_on_all: true
llm:
  model: "gemini/gemini-1.5-flash"
telegram:
  bot_token: null
  chat_id: null
watchlist:
  enabled: true
  tickers: ["$NVDA"]
  sectors: ["Semiconductors"]
  score_boost: 1
""")

    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text("""
sources:
  - name: "TestSource"
    type: "rss"
    url: "https://example.com/rss"
    enabled: true
""")

    mock_article = RawArticle(
        title="Nvidia Unveils Next-Gen AI Superchip",
        url="https://tech.example.com/nvda-superchip",
        source_name="TechDaily",
        published_at=now,
        summary="Nvidia launches new generation computing architecture.",
    )

    mock_analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Nvidia launches new generation computing architecture.",
        catalyst_thesis="Immediate hardware upgrade cycle across hyperscalers.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
    )

    runner = CliRunner()
    with patch("src.pipeline.fetch_all_sources", return_value=[mock_article]), \
         patch("src.analysis.analyzer.MarketImpactAnalyzer.analyze", return_value=mock_analysis):

        result = runner.invoke(
            app,
            [
                "scan",
                "-c", str(settings_file),
                "-s", str(sources_file),
                "--alert-on-all",
            ],
        )

        assert result.exit_code == 0
        assert "Executing single scan cycle..." in result.stdout
        assert "Scan Cycle Summary" in result.stdout
        assert "ACTIONABLE INVESTMENT BOTTOM LINE" in result.stdout
        assert "$NVDA" in result.stdout


def test_run_once_displays_sector_news_when_available(temp_cli_db, tmp_path: Path):
    """Verify run-once displays the Sector News Sentiment Summary table when sectors have news."""
    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    # Pre-populate sector sentiment in SQLite
    storage.record_sector_sentiment("Semiconductors", 0.70, "BULLISH", "h_semi_pre", published_at=now)

    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(f"""
agent:
  database_path: "{db_path}"
filter:
  min_confidence_score: 6
  alert_on_all: true
llm:
  model: "gemini/gemini-1.5-flash"
telegram:
  bot_token: null
  chat_id: null
""")

    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text("sources: []\n")

    runner = CliRunner()
    with patch("src.pipeline.fetch_all_sources", return_value=[]):
        result = runner.invoke(
            app,
            [
                "run-once",
                "-c", str(settings_file),
                "-s", str(sources_file),
            ],
        )

        assert result.exit_code == 0
        assert "Scan Cycle Summary" in result.stdout
        assert "Sector News Sentiment Summary" in result.stdout
        assert "Semiconductors" in result.stdout
