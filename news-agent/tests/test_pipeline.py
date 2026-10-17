from unittest.mock import MagicMock, patch
import pytest

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.pipeline import Pipeline, CycleSummary
from src.analysis.analyzer import AnalysisError
from src.config import Settings, FilterSettings


def test_pipeline_filters_below_threshold():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    # Configure article and low-impact analysis
    art = RawArticle(
        title="Routine CEO Interview",
        url="https://test.com/1",
        source_name="Test",
        published_at=None,
        summary="Routine",
    )
    mock_storage.filter_unseen.return_value = [art]
    mock_analyzer.analyze.return_value = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=4,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=[],
        affected_sectors=[],
        summary="Routine interview",
        catalyst_thesis="No catalyst",
    )

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art])

    assert summary["seen"] == 1
    assert summary["alerted"] == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(
        art, mock_analyzer.analyze.return_value, alerted=False
    )


def test_pipeline_alerts_above_threshold():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Fed Unexpected 50bps Rate Cut",
        url="https://test.com/fed-cut",
        source_name="Test",
        published_at=None,
        summary="Emergency cut",
    )
    mock_storage.filter_unseen.return_value = [art]
    high_impact_analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$SPY", "$QQQ"],
        affected_sectors=["Financials"],
        summary="Fed cuts rates by 50bps in surprise meeting.",
        catalyst_thesis="Immediate liquidity injection re-rating market multiples.",
    )
    mock_analyzer.analyze.return_value = high_impact_analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
        min_impact_grade=ImpactGrade.HIGH,
    )
    summary = pipeline.process_articles([art])

    assert summary.fetched == 1
    assert summary.new_unseen == 1
    assert summary.analyzed == 1
    assert summary.alerted == 1
    mock_notifier.send_alert.assert_called_once_with(art, high_impact_analysis)
    mock_storage.record_article.assert_called_once_with(
        art, high_impact_analysis, alerted=True
    )


def test_pipeline_alert_on_all_bypass():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Minor Product Update",
        url="https://test.com/minor",
        source_name="Test",
        published_at=None,
        summary="Minor",
    )
    mock_storage.filter_unseen.return_value = [art]
    low_analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=2,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=[],
        summary="Minor minor update",
        catalyst_thesis="Negligible impact",
    )
    mock_analyzer.analyze.return_value = low_analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        alert_on_all=True,
    )
    summary = pipeline.process_articles([art])

    assert summary["alerted"] == 1
    mock_notifier.send_alert.assert_called_once_with(art, low_analysis)
    mock_storage.record_article.assert_called_once_with(art, low_analysis, alerted=True)


def test_pipeline_skips_already_seen_articles():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="Seen Article",
        url="https://test.com/seen",
        source_name="Test",
        published_at=None,
        summary="Seen",
    )
    mock_storage.filter_unseen.return_value = []

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
    )
    summary = pipeline.process_articles([art])

    assert summary.fetched == 1
    assert summary.new_unseen == 0
    assert summary.analyzed == 0
    assert summary.alerted == 0
    mock_analyzer.analyze.assert_not_called()
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_not_called()


def test_pipeline_error_isolation_analyzer_failure():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art1 = RawArticle(
        title="Article Failing Analysis",
        url="https://test.com/fail",
        source_name="Test",
        published_at=None,
        summary="Fail",
    )
    art2 = RawArticle(
        title="Valid Moving Article",
        url="https://test.com/pass",
        source_name="Test",
        published_at=None,
        summary="Pass",
    )

    mock_storage.filter_unseen.return_value = [art1, art2]
    valid_analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Export license granted.",
        catalyst_thesis="Key market access restored.",
    )

    mock_analyzer.analyze.side_effect = [
        AnalysisError("LLM API parsing failure"),
        valid_analysis,
    ]

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art1, art2])

    assert summary.fetched == 2
    assert summary.new_unseen == 2
    assert summary.analyzed == 1
    assert summary.alerted == 1
    assert summary.errors == 1
    # art2 must be alerted and recorded
    mock_notifier.send_alert.assert_called_once_with(art2, valid_analysis)
    mock_storage.record_article.assert_called_once_with(
        art2, valid_analysis, alerted=True
    )


def test_pipeline_error_isolation_notifier_failure():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.side_effect = Exception("Telegram connection reset")

    art = RawArticle(
        title="Moving News",
        url="https://test.com/news",
        source_name="Test",
        published_at=None,
        summary="Moving",
    )
    mock_storage.filter_unseen.return_value = [art]
    valid_analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$AAPL"],
        affected_sectors=["Tech"],
        summary="Surprise dividend hike.",
        catalyst_thesis="Shareholder capital return acceleration.",
    )
    mock_analyzer.analyze.return_value = valid_analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 0
    # Still records article with alerted=False when notifier throws
    mock_storage.record_article.assert_called_once_with(
        art, valid_analysis, alerted=False
    )


def test_pipeline_grade_filtering():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="Average Impact News",
        url="https://test.com/avg",
        source_name="Test",
        published_at=None,
        summary="Avg",
    )
    mock_storage.filter_unseen.return_value = [art]
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,  # Meets score 6
        impact_grade=ImpactGrade.AVERAGE,  # Below min_impact_grade HIGH
        affected_tickers=["$MSFT"],
        affected_sectors=["Software"],
        summary="Cloud service update.",
        catalyst_thesis="Minor revenue driver.",
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
        min_impact_grade=ImpactGrade.HIGH,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(art, analysis, alerted=False)


def test_pipeline_run_cycle_full():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Headline",
        url="https://test.com/headline",
        source_name="Test",
        published_at=None,
        summary="Summary",
    )
    mock_storage.filter_unseen.return_value = [art]
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=7,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$TSLA"],
        affected_sectors=["Automotive"],
        summary="Factory expansion approved.",
        catalyst_thesis="Unit output growth.",
    )
    mock_analyzer.analyze.return_value = analysis

    mock_fetcher = MagicMock(return_value=[art])

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        fetcher=mock_fetcher,
        min_score=6,
    )
    summary = pipeline.run_cycle()

    mock_fetcher.assert_called_once()
    assert summary.fetched == 1
    assert summary.alerted == 1
    assert summary["seen"] == 1


def test_cycle_summary_dataclass_contract():
    cs = CycleSummary(fetched=10, new_unseen=5, analyzed=5, alerted=2, errors=1)
    assert cs.fetched == 10
    assert cs.new_unseen == 5
    assert cs.seen == 5
    assert cs.analyzed == 5
    assert cs.alerted == 2
    assert cs.errors == 1

    # Subscripting support
    assert cs["fetched"] == 10
    assert cs["seen"] == 5
    assert cs["alerted"] == 2

    # Dictionary conversion
    d = cs.to_dict()
    assert d["seen"] == 5
    assert d["alerted"] == 2


def test_pipeline_from_settings(tmp_path):
    db_file = tmp_path / "test.db"
    settings = Settings.model_validate({
        "agent": {"database_path": str(db_file), "polling_interval_minutes": 5},
        "filter": {"min_confidence_score": 7, "min_impact_grade": "CRITICAL"},
        "telegram": {"bot_token": "token123", "chat_id": "chat456"},
    })
    pipeline = Pipeline.from_settings(settings=settings, sources=[])
    assert pipeline.min_confidence_score == 7
    assert pipeline.min_impact_grade == ImpactGrade.CRITICAL
    assert pipeline.storage is not None
    assert pipeline.notifier.bot_token == "token123"
    pipeline.close()


def test_pipeline_context_manager():
    mock_storage = MagicMock()
    with Pipeline(storage=mock_storage) as p:
        assert p.storage == mock_storage
    mock_storage.close.assert_called_once()


def test_cli_stats(tmp_path):
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    db_file = tmp_path / "test_stats.db"
    result = runner.invoke(app, ["stats", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "Total Articles Seen" in result.output


def test_cli_test_connections():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["test-connections"])
    assert result.exit_code == 0
    assert "Unified System Connectivity Diagnostics" in result.output


def test_cli_test_telegram_missing():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""}):
        result = runner.invoke(app, ["test-telegram"])
        assert result.exit_code == 1
        assert "Telegram credentials missing" in result.output


def test_cli_test_telegram_success():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    with patch("src.cli.load_settings") as mock_settings, \
         patch("src.cli.TelegramNotifier.send_message", return_value=True):
        mock_settings.return_value.telegram.bot_token = "valid_token"
        mock_settings.return_value.telegram.chat_id = "valid_chat"
        result = runner.invoke(app, ["test-telegram"])
        assert result.exit_code == 0
        assert "delivered successfully" in result.output


def test_cli_test_source_not_found():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["test-source", "--name", "NonExistentSource12345"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_cli_test_source_success():
    from typer.testing import CliRunner
    from src.cli import app
    from src.config import SourceConfig

    runner = CliRunner()
    mock_source = SourceConfig(name="MockSource", type="rss", url="https://mock.com/rss", enabled=True)
    mock_art = RawArticle(title="Mock Headline", url="https://mock.com/1", source_name="MockSource")

    with patch("src.cli.load_sources", return_value=[mock_source]), \
         patch("src.cli.RSSFetcher.fetch", return_value=[mock_art]):
        result = runner.invoke(app, ["test-source", "--name", "MockSource"])
        assert result.exit_code == 0
        assert "Successfully fetched 1 articles" in result.output
        assert "Mock Headline" in result.output


def test_cli_run_once():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    with patch("src.cli.Pipeline.run_cycle") as mock_run:
        mock_run.return_value = CycleSummary(fetched=5, new_unseen=3, analyzed=3, alerted=1, errors=0)
        result = runner.invoke(app, ["run-once"])
        assert result.exit_code == 0
        assert "Scan Cycle Summary" in result.output
        assert "Fetched Articles" in result.output


def test_cli_run_daemon_graceful_shutdown():
    import threading
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    real_event = threading.Event()

    def mock_cycle():
        # Trigger stop after first cycle execution
        real_event.set()
        return CycleSummary(fetched=1, new_unseen=1, analyzed=1, alerted=0, errors=0)

    with patch("src.cli.Pipeline.run_cycle", side_effect=mock_cycle), \
         patch("src.cli.threading.Event", return_value=real_event):
        result = runner.invoke(app, ["run", "--interval", "1"])
        assert result.exit_code == 0
        assert "Starting Market News Agent daemon" in result.output
        assert "Daemon shutdown complete" in result.output

