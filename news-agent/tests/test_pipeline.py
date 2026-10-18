from unittest.mock import MagicMock, patch
import pytest

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade, EventStatus, HardGates
from src.pipeline import Pipeline, CycleSummary
from src.analysis.analyzer import AnalysisError
from src.config import Settings, FilterSettings, WatchlistSettings
from src.notification.console import ConsoleNotifier



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


def test_pipeline_from_settings_unconfigured_telegram_falls_back_to_console(tmp_path):
    """Verify Pipeline defaults to ConsoleNotifier when Telegram credentials are unconfigured."""
    db_file = tmp_path / "test_local.db"
    settings = Settings.model_validate({
        "agent": {"database_path": str(db_file), "polling_interval_minutes": 5},
        "filter": {"min_confidence_score": 6},
        "telegram": {"bot_token": "", "chat_id": ""},
    })
    pipeline = Pipeline.from_settings(settings=settings, sources=[])
    assert isinstance(pipeline.notifier, ConsoleNotifier)
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
        result = runner.invoke(app, ["run", "--daemon", "--interval", "1"])
        assert result.exit_code == 0
        assert "Starting Market News Agent daemon" in result.output
        assert "Daemon shutdown complete" in result.output


def test_cli_run_no_daemon():
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    with patch("src.cli.Pipeline.run_cycle") as mock_run:
        mock_run.return_value = CycleSummary(fetched=2, new_unseen=1, analyzed=1, alerted=0, errors=0)
        result = runner.invoke(app, ["run", "--no-daemon"])
        assert result.exit_code == 0
        assert "Executing single news agent scan cycle" in result.output
        assert "Cycle completed: 2 fetched" in result.output


def test_pipeline_run_cycle_ingestion_exception_containment():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    mock_fetcher = MagicMock(side_effect=Exception("Critical RSS network timeout"))

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        fetcher=mock_fetcher,
    )
    summary = pipeline.run_cycle()

    assert summary.fetched == 0
    assert summary.errors == 1
    mock_analyzer.analyze.assert_not_called()
    mock_notifier.send_alert.assert_not_called()


def test_pipeline_run_cycle_enforces_batch_size_per_source():
    from src.config import Settings, AgentSettings
    mock_storage = MagicMock()
    mock_storage.filter_unseen.return_value = []
    settings = Settings(agent=AgentSettings(batch_size_per_source=7))

    with patch("src.pipeline.fetch_all_sources") as mock_fetch:
        mock_fetch.return_value = []
        pipeline = Pipeline(
            storage=mock_storage,
            settings=settings,
        )
        pipeline.run_cycle()

        mock_fetch.assert_called_once()
        _, kwargs = mock_fetch.call_args
        assert kwargs.get("max_items_per_source") == 7


def test_pipeline_hard_gates_watchlist_suppressed():
    """Verify WATCHLIST status articles are stored with alerted=False and never alert Telegram."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="Unconfirmed M&A Rumors",
        url="https://test.com/rumor",
        source_name="BlogNews",
    )
    mock_storage.filter_unseen.return_value = [art]
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        status=EventStatus.WATCHLIST,
        affected_tickers=["$M&A"],
        affected_sectors=["Technology"],
        summary="Rumors circulating regarding acquisition.",
        catalyst_thesis="Potential valuation re-rating if true.",
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(art, analysis, alerted=False)


def test_pipeline_hard_gates_unverified_suppressed():
    """Verify UNVERIFIED status articles are stored with alerted=False and never alert Telegram."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="Unverified Rumor on Social Media",
        url="https://test.com/unverified",
        source_name="Twitter",
    )
    mock_storage.filter_unseen.return_value = [art]
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        status=EventStatus.UNVERIFIED,
        affected_tickers=["$SPEC"],
        affected_sectors=["Energy"],
        summary="Speculative leak without primary documentation.",
        catalyst_thesis="Hypothetical supply drop.",
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(art, analysis, alerted=False)


def test_pipeline_hard_gates_informational_and_no_action_suppressed():
    """Verify INFORMATIONAL and NO_ACTION statuses are suppressed from Telegram alerting."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art1 = RawArticle(title="Routine Filing", url="https://test.com/info", source_name="SEC")
    art2 = RawArticle(title="Irrelevant Opinion", url="https://test.com/noaction", source_name="Opinion")
    mock_storage.filter_unseen.return_value = [art1, art2]

    analysis1 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        status=EventStatus.INFORMATIONAL,
        summary="Routine 10-Q filing.",
        catalyst_thesis="Baseline filings info.",
    )
    analysis2 = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=7,
        impact_grade=ImpactGrade.AVERAGE,
        status=EventStatus.NO_ACTION,
        summary="Opinion piece without actionable facts.",
        catalyst_thesis="No impact.",
    )
    mock_analyzer.analyze.side_effect = [analysis1, analysis2]

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=6,
    )
    summary = pipeline.process_articles([art1, art2])

    assert summary.analyzed == 2
    assert summary.alerted == 0
    mock_notifier.send_alert.assert_not_called()
    assert mock_storage.record_article.call_count == 2
    mock_storage.record_article.assert_any_call(art1, analysis1, alerted=False)
    mock_storage.record_article.assert_any_call(art2, analysis2, alerted=False)


def test_pipeline_hard_gates_failure_blocks_even_with_alert_on_all():
    """Verify that failed hard-gates block alerts even if alert_on_all CLI flag is enabled."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(title="Indirect Exposure", url="https://test.com/indirect", source_name="Reuters")
    mock_storage.filter_unseen.return_value = [art]

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=10,
        impact_grade=ImpactGrade.CRITICAL,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        summary="Tenuous connection.",
        catalyst_thesis="Third order speculation.",
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=False,  # Failed gate!
        ),
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        alert_on_all=True,  # Even with CLI override
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(art, analysis, alerted=False)


def test_pipeline_opportunity_candidate_with_passing_hard_gates_alerts():
    """Verify OPPORTUNITY_CANDIDATE with passing hard gates and qualifying score triggers alert."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Department of Defense Awards $5B Drone Contract",
        url="https://defense.gov/drone-contract",
        source_name="DoD",
    )
    mock_storage.filter_unseen.return_value = [art]

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        affected_tickers=["$KTOS", "$AVAV"],
        affected_sectors=["Defense"],
        summary="DoD finalized $5B production contract for autonomous loitering munitions.",
        catalyst_thesis="Immediate revenue recognition and substantial order backlog expansion.",
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=True,
            passes_all_gates=True,
        ),
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        min_score=7,
        min_impact_grade=ImpactGrade.HIGH,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    assert summary.alerted == 1
    mock_notifier.send_alert.assert_called_once_with(art, analysis)
    mock_storage.record_article.assert_called_once_with(art, analysis, alerted=True)


def test_pipeline_records_llm_metric_when_token_usage_present():
    """Verify that pipeline records LLM metrics in storage whenever analysis.token_usage is present."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="NVIDIA Reports Revenue",
        url="https://example.com/nvda",
        source_name="CNBC",
    )
    mock_storage.filter_unseen.return_value = [art]

    from src.models import TokenUsage
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=5,
        impact_grade=ImpactGrade.AVERAGE,
        summary="Routine revenue report.",
        catalyst_thesis="In line with consensus.",
        status=EventStatus.INFORMATIONAL,
        token_usage=TokenUsage(
            prompt_tokens=320,
            completion_tokens=90,
            total_tokens=410,
            estimated_cost_usd=0.00015,
            model_name="gemini/gemini-1.5-flash",
        ),
    )
    mock_analyzer.analyze.return_value = analysis

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
    )
    summary = pipeline.process_articles([art])

    assert summary.analyzed == 1
    mock_storage.record_llm_metric.assert_called_once_with(
        content_hash=art.content_hash,
        model_name="gemini/gemini-1.5-flash",
        prompt_tokens=320,
        completion_tokens=90,
        total_tokens=410,
        cost_usd=0.00015,
    )


def test_cli_metrics_command(tmp_path):
    """Verify CLI metrics command executes cleanly and outputs full Rich telemetry table."""
    from typer.testing import CliRunner
    from src.cli import app
    from src.storage.sqlite import SQLiteStorage

    db_file = tmp_path / "cli_metrics.db"
    storage = SQLiteStorage(str(db_file))
    storage.record_llm_metric(
        content_hash="h1",
        model_name="gemini/gemini-1.5-flash",
        prompt_tokens=400,
        completion_tokens=100,
        total_tokens=500,
        cost_usd=0.0001,
    )
    storage.record_llm_metric(
        content_hash="h2",
        model_name="gpt-4o-mini",
        prompt_tokens=600,
        completion_tokens=200,
        total_tokens=800,
        cost_usd=0.00025,
    )
    storage.close()

    runner = CliRunner()
    result = runner.invoke(app, ["metrics", "--db", str(db_file)])

    assert result.exit_code == 0
    output = result.output
    assert "LLM Observability & Cost Metrics" in output
    assert "Total LLM Inferences" in output
    assert "Prompt Tokens" in output
    assert "Completion Tokens" in output
    assert "Total Tokens" in output
    assert "Total Estimated Cost ($ USD)" in output
    assert "Spend this Month vs Monthly Budget" in output
    assert "Calls by Model" in output
    assert "gemini/gemini-1.5-flash" in output
    assert "gpt-4o-mini" in output


def test_pipeline_recency_filtering(tmp_path):
    """Verify that articles older than max_article_age_minutes are filtered out and marked as seen."""
    from datetime import datetime, timezone, timedelta
    from src.storage.sqlite import SQLiteStorage
    from src.models import CausalChain

    db_file = tmp_path / "recency_test.db"
    storage = SQLiteStorage(str(db_file))

    now = datetime.now(timezone.utc)
    fresh_article = RawArticle(
        title="Breaking Tech Surge",
        url="https://example.com/fresh",
        source_name="Reuters",
        published_at=now - timedelta(minutes=5),
        summary="Fresh breaking article",
    )
    stale_article = RawArticle(
        title="Old Routine Report",
        url="https://example.com/old",
        source_name="CNBC",
        published_at=now - timedelta(hours=3),
        summary="3 hour old article",
    )

    mock_analyzer = MagicMock()
    mock_analyzer.analyze.return_value = MarketImpactAnalysis(
        summary="Nvidia GPU production ramps up.",
        catalyst_thesis="Accelerating hyperscaler capex drives hardware revenue.",
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        causal_chain=CausalChain(
            event_facts="Facts",
            economic_mechanism="Mechanism",
            affected_sectors=["Semiconductors"],
            beneficiaries=["$NVDA"],
            negatively_exposed=["$INTC"],
        ),
    )

    pipeline = Pipeline(
        storage=storage,
        analyzer=mock_analyzer,
        notifier=ConsoleNotifier(),
        max_article_age_minutes=15,
        alert_on_all=True,
    )

    summary = pipeline.process_articles([fresh_article, stale_article])

    # Only fresh article should be analyzed
    assert summary.fetched == 2
    assert summary.analyzed == 1
    assert len(summary.top_catalysts) == 1
    assert summary.top_catalysts[0][0].title == "Breaking Tech Surge"

    # Both articles must be recorded in storage so neither is re-analyzed
    assert storage.is_article_seen(fresh_article.content_hash) is True
    assert storage.is_article_seen(stale_article.content_hash) is True

    # Immediate second run: zero unseen, zero analyzed
    summary2 = pipeline.process_articles([fresh_article, stale_article])
    assert summary2.new_unseen == 0
    assert summary2.analyzed == 0
    storage.close()


def test_render_actionable_investment_bottom_line():
    """Verify that render_actionable_investment_bottom_line formats stock/sector bottom line."""
    from rich.console import Console
    from io import StringIO
    from src.cli import render_actionable_investment_bottom_line
    from src.models import CausalChain, DirectionalStance

    buf = StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)

    art = RawArticle(
        title="Nvidia Revenue Surge",
        url="https://example.com/nvda",
        source_name="Reuters",
        summary="AI demand spikes",
    )
    analysis = MarketImpactAnalysis(
        summary="Hyperscaler capex expanded 40%",
        catalyst_thesis="Accelerated datacenter capex expands shipments",
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        affected_sectors=["Semiconductors", "AI Infrastructure"],
        causal_chain=CausalChain(
            event_facts="Hyperscaler demand expanded 40%",
            economic_mechanism="Accelerated datacenter capex directly expands GPU shipments",
            affected_sectors=["Semiconductors"],
            beneficiaries=["$NVDA", "$TSM"],
            negatively_exposed=["$INTC"],
            invalidation_conditions="Capex reductions by top 4 hyperscalers",
        ),
    )

    summary = CycleSummary(
        fetched=1,
        new_unseen=1,
        analyzed=1,
        alerted=1,
        errors=0,
        top_catalysts=[(art, analysis)],
    )

    render_actionable_investment_bottom_line(test_console, summary, storage=None)
    output = buf.getvalue()

    assert "ACTIONABLE INVESTMENT BOTTOM LINE" in output
    assert "STOCKS & SECTORS TO INVEST IN (BENEFICIARIES)" in output
    assert "$NVDA" in output
    assert "$TSM" in output
    assert "$INTC" in output
    assert "Semiconductors" in output
    assert "Directional Stance: BULLISH" in output
    assert "Causal Transmission Thesis" in output
    assert "Thesis Invalidation Condition" in output


def test_render_actionable_investment_bottom_line_sector_only():
    """Verify that render_actionable_investment_bottom_line formats sector-level plays when no single stock is named."""
    from rich.console import Console
    from io import StringIO
    from src.cli import render_actionable_investment_bottom_line
    from src.models import CausalChain, DirectionalStance

    buf = StringIO()
    test_console = Console(file=buf, force_terminal=False, color_system=None)

    art = RawArticle(
        title="Global Jet Fuel Price Plunge Boosts Aviation Margins",
        url="https://example.com/jet-fuel",
        source_name="Bloomberg",
        summary="Crude slump reduces jet fuel operating cost by 25%.",
    )
    analysis = MarketImpactAnalysis(
        summary="Aviation fuel operating costs dropping sharply across industry.",
        catalyst_thesis="Immediate gross margin expansion across commercial carriers.",
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        affected_tickers=[],  # No specific single stock, broad sector play!
        affected_sectors=["Commercial Airlines", "Aviation & Travel"],
        causal_chain=CausalChain(
            event_facts="Jet fuel prices dropped 25% week-over-week.",
            economic_mechanism="Direct fuel cost relief flows straight to carrier operating cash flow.",
            affected_sectors=["Commercial Airlines"],
            beneficiaries=[],  # No single stock
            negatively_exposed=[],
        ),
    )

    summary = CycleSummary(
        fetched=1,
        new_unseen=1,
        analyzed=1,
        alerted=1,
        errors=0,
        top_catalysts=[(art, analysis)],
    )

    render_actionable_investment_bottom_line(test_console, summary, storage=None)
    output = buf.getvalue()

    assert "ACTIONABLE INVESTMENT BOTTOM LINE" in output
    assert "STOCKS & SECTORS TO INVEST IN (BENEFICIARIES)" in output
    assert "Commercial Airlines" in output
    assert "(Sector-wide play / Industry ETFs)" in output



def test_cli_limit_per_source_and_max_age_options(tmp_path):
    """Verify that CLI run-once accepts --limit-per-source and --max-age-minutes."""
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    db_file = tmp_path / "cli_test.db"
    settings_file = tmp_path / "settings.yaml"
    settings_file.write_text(f"""
agent:
  polling_interval_minutes: 15
  batch_size_per_source: 10
  max_article_age_minutes: 30
  database_path: "{db_file}"
filter:
  min_confidence_score: 8
  min_impact_grade: "CRITICAL"
  alert_on_all: false
llm:
  model: "gemini/gemini-3.8-flash"
telegram:
  bot_token: null
  chat_id: null
""")

    sources_file = tmp_path / "sources.yaml"
    sources_file.write_text("""
sources:
  - name: "TestSource"
    type: "rss"
    url: "https://example.com/feed"
    enabled: true
""")

    with patch("src.pipeline.fetch_all_sources", return_value=[]) as mock_fetch:
        result = runner.invoke(
            app,
            [
                "run-once",
                "-c", str(settings_file),
                "-s", str(sources_file),
                "--limit-per-source", "3",
                "--max-age-minutes", "10",
            ],
        )
        assert result.exit_code == 0
        assert mock_fetch.called
        kwargs = mock_fetch.call_args.kwargs
        assert kwargs["max_items_per_source"] == 3
        assert kwargs["max_article_age_minutes"] == 10


def test_pipeline_watchlist_ticker_match_boosts_score_and_elevates_grade():
    """Verify that an article matching watchlist tickers receives priority scoring boost and grade elevation."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Nvidia Unveils Next-Gen AI Architecture",
        url="https://test.com/nvda-ai",
        source_name="TechWire",
    )
    mock_storage.filter_unseen.return_value = [art]

    # Analysis initially with score 6 (AVERAGE grade, not market moving)
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=6,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Nvidia announcements.",
        catalyst_thesis="New products.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )
    mock_analyzer.analyze.return_value = analysis

    watchlist = WatchlistSettings(
        enabled=True,
        tickers=["$NVDA", "$AAPL"],
        sectors=[],
        score_boost=1,
    )

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        watchlist=watchlist,
        min_score=7,
        min_impact_grade=ImpactGrade.HIGH,
    )

    summary = pipeline.process_articles([art])

    # Watchlist match checks
    assert analysis.is_watchlist_match is True
    assert "$NVDA" in analysis.matched_watchlist_items
    assert analysis.confidence_score == 7
    assert analysis.impact_grade == ImpactGrade.HIGH
    assert analysis.is_market_moving is True

    # Alert should have fired because score was elevated from 6 to 7 (HIGH)
    assert summary.alerted == 1
    mock_notifier.send_alert.assert_called_once()


def test_pipeline_watchlist_sector_match():
    """Verify that an article matching watchlist sectors receives score boost."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    mock_notifier.send_alert.return_value = True

    art = RawArticle(
        title="Aviation Sector Facing Fuel Headwinds",
        url="https://test.com/aviation-fuel",
        source_name="AeroDaily",
    )
    mock_storage.filter_unseen.return_value = [art]

    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=5,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$DAL"],
        affected_sectors=["Aviation"],
        summary="Jet fuel spikes impact airline margins.",
        catalyst_thesis="Fuel expenditure cuts operating margin.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )
    mock_analyzer.analyze.return_value = analysis

    watchlist = WatchlistSettings(
        enabled=True,
        tickers=[],
        sectors=["Aviation", "Defense"],
        score_boost=2,
    )

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        watchlist=watchlist,
        min_score=7,
    )

    summary = pipeline.process_articles([art])

    assert analysis.is_watchlist_match is True
    assert "Aviation" in analysis.matched_watchlist_items
    assert analysis.confidence_score == 7
    assert analysis.impact_grade == ImpactGrade.HIGH
    assert analysis.is_market_moving is True
    assert summary.alerted == 1


def test_pipeline_watchlist_disabled_no_boost():
    """Verify that when watchlist is disabled, no matching or score boost is applied."""
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()

    art = RawArticle(
        title="Nvidia Routine Announcement",
        url="https://test.com/nvda-routine",
        source_name="TechDaily",
    )
    mock_storage.filter_unseen.return_value = [art]

    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=6,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Routine.",
        catalyst_thesis="None.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        hard_gates=HardGates(is_novel=True, is_verified=True, has_direct_exposure=True),
    )
    mock_analyzer.analyze.return_value = analysis

    watchlist = WatchlistSettings(
        enabled=False,
        tickers=["$NVDA"],
        score_boost=2,
    )

    pipeline = Pipeline(
        storage=mock_storage,
        analyzer=mock_analyzer,
        notifier=mock_notifier,
        watchlist=watchlist,
        min_score=7,
    )

    summary = pipeline.process_articles([art])

    assert analysis.is_watchlist_match is False
    assert analysis.confidence_score == 6
    assert analysis.impact_grade == ImpactGrade.AVERAGE
    assert summary.alerted == 0








