from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.storage.sqlite import SQLiteStorage
from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    DirectionalStance,
    CausalChain,
    HardGates,
)


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


def test_history_command_empty(temp_cli_db):
    """Verify history command informs user when no event lineage records exist."""
    _, db_path = temp_cli_db
    runner = CliRunner()
    result = runner.invoke(app, ["history", "--db", db_path])

    assert result.exit_code == 0
    assert "No event lineage records found" in result.stdout


def test_history_command_populated(temp_cli_db):
    """Verify history command renders table of event lineage records."""
    from src.models import EventLineageRecord

    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    record = EventLineageRecord(
        event_id="evt_test_12345",
        title="Nvidia Blackwell racks begin mass delivery",
        primary_ticker="$NVDA",
        first_seen=now,
        last_updated=now,
        source_count=3,
        all_sources=["Reuters", "Bloomberg", "FT"],
        event_quality_score=9.0,
        opportunity_score=8.5,
        status="ALERT",
        initial_price=125.50,
    )
    storage.record_event_lineage(record)

    runner = CliRunner()
    result = runner.invoke(app, ["history", "--ticker", "$NVDA", "--db", db_path])

    assert result.exit_code == 0
    assert "Structured Event Lineage History" in result.stdout
    assert "$NVDA" in result.stdout
    assert "Nvidia" in result.stdout
    assert "ALERT" in result.stdout
    assert "125" in result.stdout


def test_performance_command_empty(temp_cli_db):
    """Verify performance command informs user when no realized catalyst return records exist."""
    _, db_path = temp_cli_db
    runner = CliRunner()
    result = runner.invoke(app, ["performance", "--db", db_path])

    assert result.exit_code == 0
    assert "No realized catalyst return records found" in result.stdout


def test_performance_command_populated(temp_cli_db):
    """Verify performance command displays Global Metrics, Publisher Scorecard, and Realized Return Ledger."""
    from src.models import EventLineageRecord

    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    # 1. Bullish win on Reuters / Bloomberg
    ev1 = EventLineageRecord(
        event_id="evt_win_nvda",
        title="NVIDIA secures cloud supply contract",
        primary_ticker="$NVDA",
        first_seen=now,
        last_updated=now,
        source_count=2,
        all_sources=["Reuters", "Bloomberg"],
        event_quality_score=9.0,
        opportunity_score=8.5,
        status="ALERT",
        directional_stance="BULLISH",
        initial_price=120.0,
        subsequent_return_1d=4.5,
        subsequent_return_7d=8.0,
    )
    storage.record_event_lineage(ev1)

    # 2. Bearish win on Financial Times
    ev2 = EventLineageRecord(
        event_id="evt_win_msft",
        title="Microsoft cancels datacenter expansion",
        primary_ticker="$MSFT",
        first_seen=now,
        last_updated=now,
        source_count=1,
        all_sources=["Financial Times"],
        event_quality_score=9.0,
        opportunity_score=8.0,
        status="ALERT",
        directional_stance="BEARISH",
        initial_price=420.0,
        subsequent_return_1d=-3.2,
        subsequent_return_7d=-5.5,
    )
    storage.record_event_lineage(ev2)

    runner = CliRunner(env={"COLUMNS": "200"})
    result = runner.invoke(app, ["performance", "--db", db_path])

    assert result.exit_code == 0
    assert "GLOBAL ALPHA & HIT RATE METRICS" in result.stdout
    assert "PUBLISHER CREDIBILITY SCORECARD" in result.stdout
    assert "CATALYST REALIZED RETURN LEDGER" in result.stdout
    assert "Reuters" in result.stdout
    assert "Financial Times" in result.stdout
    assert "$NVDA" in result.stdout
    assert "$MSFT" in result.stdout
    assert "TIER_A_ALPHA" in result.stdout or "TIER A" in result.stdout


def test_performance_command_update_flag(temp_cli_db):
    """Verify performance command with --update triggers price updates."""
    from src.models import EventLineageRecord

    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    # Pending event eligible for update
    ev_pending = EventLineageRecord(
        event_id="evt_pending_quote",
        title="AMD expands enterprise accelerator lineup",
        primary_ticker="$AMD",
        first_seen=now,
        last_updated=now,
        source_count=1,
        all_sources=["Reuters"],
        event_quality_score=8.0,
        opportunity_score=7.5,
        status="ALERT",
        directional_stance="BULLISH",
        initial_price=150.0,
        subsequent_return_1d=None,
    )
    storage.record_event_lineage(ev_pending)

    runner = CliRunner()
    with patch("src.analysis.performance.PerformanceTrackerEngine.update_forward_returns", return_value={"updated_1d": 1, "updated_7d": 0}):
        result = runner.invoke(app, ["performance", "--update", "--db", db_path])
        assert result.exit_code == 0
        assert "Updated 1 forward return quote" in result.stdout


def test_push_pending_dry_run_and_dispatch(temp_cli_db):
    """Verify push-pending identifies qualifying unalerted articles, dry runs, and dispatches."""
    storage, db_path = temp_cli_db
    now = datetime.now(timezone.utc)

    # Seed an article that qualifies: score 8, HIGH, with tickers
    raw = RawArticle(
        title="Test Company Signs Transformational $5B Contract",
        url="https://example.com/test-article",
        source_name="Reuters Business",
        published_at=now,
        content="Test Company has signed a definitive $5B deal.",
    )
    analysis = MarketImpactAnalysis(
        summary="Test Company has signed a definitive $5B deal.",
        catalyst_thesis="Material accretive contract expansion.",
        impact_grade=ImpactGrade.HIGH,
        confidence_score=8,
        affected_tickers=["$TEST"],
        affected_sectors=["Technology"],
        is_market_moving=True,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        causal_chain=CausalChain(
            event_facts="Test Company signed contract.",
            economic_mechanism="Accelerated revenue recognition.",
            affected_sectors=["Technology"],
            beneficiaries=["$TEST"],
            negatively_exposed=[],
            counterarguments="Execution risks exist.",
            invalidation_conditions="Deal canceled by counterparty.",
        ),
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True, passes_all_gates=True),
    )
    storage.record_article(raw, analysis, alerted=False)

    runner = CliRunner()

    # 1. Test dry-run mode
    res_dry = runner.invoke(app, ["push-pending", "--dry-run", "--db", db_path])
    assert res_dry.exit_code == 0
    assert "Found 1 qualifying article(s)" in res_dry.stdout
    assert "Dry run mode active" in res_dry.stdout

    # Verify article is still unalerted in storage
    stats_before = storage.get_stats()
    assert stats_before["alerted_count"] == 0

    # 2. Test live dispatch mode with mocked notifier
    with patch("src.notification.telegram.TelegramNotifier.send_alert", return_value=True):
        res_live = runner.invoke(app, ["push-pending", "--delay", "0.0", "--db", db_path])
        assert res_live.exit_code == 0
        assert "SENT & RECORDED" in res_live.stdout
        assert "Dispatch complete: 1 sent successfully" in res_live.stdout

    # Verify article is now marked alerted in storage
    stats_after = storage.get_stats()
    assert stats_after["alerted_count"] == 1


