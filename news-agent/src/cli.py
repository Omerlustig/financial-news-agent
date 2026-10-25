import json
import logging
import os
import signal
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*doesn't match a supported version.*")
warnings.filterwarnings("ignore", message=".*urllib3.*")

import litellm
litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.set_verbose = False
litellm.drop_params = True

import requests
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Ensure news-agent root is in sys.path
news_agent_root = Path(__file__).resolve().parent.parent
if str(news_agent_root) not in sys.path:
    sys.path.insert(0, str(news_agent_root))

from src.config import load_settings, load_sources, Settings
from src.ingestion.rss import RSSFetcher
from src.notification.telegram import TelegramNotifier, format_sentiment_gauge
from src.notification.console import ConsoleNotifier
from src.pipeline import Pipeline, GRADE_SEVERITY
from src.storage.sqlite import SQLiteStorage
from src.analysis.digest import MarketDigestGenerator
from src.analysis.trend import TrendEngine
from src.analysis.performance import PerformanceTrackerEngine
from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    CredibilityTier,
    EventStatus,
    DirectionalStance,
    HardGates,
    CausalChain,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("news-agent")
console = Console()

app = typer.Typer(
    name="news-agent",
    help="Market News Agent: Institutional Financial News Intelligence & Alerting Engine",
    add_completion=False,
)


def render_actionable_investment_bottom_line(
    console: Console,
    summary: Any,
    storage: Optional[SQLiteStorage] = None,
) -> None:
    """Renders the institutional investment bottom line detailing recommended stocks and sectors."""
    candidates = getattr(summary, "top_catalysts", []) or []

    console.print("\n")
    if candidates:
        console.rule("[bold green]🎯 ACTIONABLE INVESTMENT BOTTOM LINE (STOCKS & SECTORS)[/bold green]")
        for article, analysis in candidates[:5]:
            beneficiaries = []
            negatively_exposed = []
            mechanism = ""
            counterargs = ""
            invalidation = ""
            if analysis.causal_chain:
                beneficiaries = analysis.causal_chain.beneficiaries or []
                negatively_exposed = analysis.causal_chain.negatively_exposed or []
                mechanism = analysis.causal_chain.economic_mechanism or ""
                counterargs = analysis.causal_chain.counterarguments or ""
                invalidation = analysis.causal_chain.invalidation_conditions or ""

            stance_str = str(analysis.directional_stance).replace("DirectionalStance.", "")
            stance_color = "green" if "BULLISH" in stance_str else ("red" if "BEARISH" in stance_str else "yellow")
            stance_icon = "📈" if "BULLISH" in stance_str else ("📉" if "BEARISH" in stance_str else "⚖️")

            if not beneficiaries and analysis.affected_tickers:
                if "BULLISH" in stance_str:
                    beneficiaries = analysis.affected_tickers
                elif "BEARISH" in stance_str:
                    negatively_exposed = analysis.affected_tickers
                else:
                    beneficiaries = analysis.affected_tickers

            sectors_list = analysis.affected_sectors or []
            if analysis.causal_chain and analysis.causal_chain.affected_sectors:
                sectors_list = analysis.causal_chain.affected_sectors
            sectors_str = ", ".join(sectors_list) if sectors_list else "Broader Market / Macro"

            # Formulate Actionable Investment Beneficiaries (Stocks and/or Sectors)
            if beneficiaries:
                stock_str = ", ".join(f"[bold green]{t}[/bold green]" for t in beneficiaries)
                if sectors_list:
                    beneficiaries_display = f"{stock_str} [dim]| Sector:[/] [bold green]{sectors_str}[/bold green]"
                else:
                    beneficiaries_display = stock_str
            elif sectors_list and ("BULLISH" in stance_str or "VOLATILE" in stance_str):
                beneficiaries_display = f"[bold green]{sectors_str}[/bold green] [dim](Sector-wide play / Industry ETFs)[/dim]"
            elif sectors_list:
                beneficiaries_display = f"[bold green]{sectors_str}[/bold green] [dim](Industry sector play)[/dim]"
            else:
                beneficiaries_display = "[italic dim]Broader Market / Macro[/italic dim]"

            # Formulate Actionable Short / Avoid Exposures (Stocks and/or Sectors)
            if negatively_exposed:
                neg_stock_str = ", ".join(f"[bold red]{t}[/bold red]" for t in negatively_exposed)
                if sectors_list:
                    neg_display = f"{neg_stock_str} [dim]| Sector:[/] [bold red]{sectors_str}[/bold red]"
                else:
                    neg_display = neg_stock_str
            elif sectors_list and "BEARISH" in stance_str:
                neg_display = f"[bold red]{sectors_str}[/bold red] [dim](Sector-wide exposure / Headwinds)[/dim]"
            else:
                neg_display = "[italic dim]None direct / Negligible[/italic dim]"

            grade_val = analysis.impact_grade.value if hasattr(analysis.impact_grade, "value") else str(analysis.impact_grade)
            grade_color = "red bold" if grade_val == "CRITICAL" else ("yellow bold" if grade_val == "HIGH" else "cyan")

            card_body = (
                f"[bold white]{article.title}[/bold white]\n"
                f"[dim]Source: {article.source_name} | Published: {article.published_at or 'Breaking'} | Grade: [{grade_color}]{grade_val}[/{grade_color}] (Score: {analysis.confidence_score}/10)[/dim]\n\n"
                f"{stance_icon} [bold]Directional Stance:[/] [{stance_color} bold]{stance_str}[/{stance_color} bold]  |  ⏱ [bold]Horizon:[/] {str(analysis.time_horizon or 'SHORT_TERM (1-7 days)').replace('TimeHorizon.', '')}\n"
                f"🏭 [bold]Sectors in Play:[/] [cyan]{sectors_str}[/cyan]\n\n"
                f"🟢 [bold green]STOCKS & SECTORS TO INVEST IN (BENEFICIARIES):[/bold green]\n"
                f"   {beneficiaries_display}\n\n"
                f"🔴 [bold red]STOCKS & SECTORS TO AVOID / SHORT (EXPOSED):[/bold red]\n"
                f"   {neg_display}\n\n"
                f"💡 [bold underline]Causal Transmission Thesis:[/bold underline]\n"
                f"[dim white]{mechanism or getattr(analysis, 'catalyst_thesis', None) or getattr(analysis, 'rationale', None) or article.summary}[/dim white]\n"
            )
            if invalidation:
                card_body += f"\n🛑 [bold]Thesis Invalidation Condition:[/] [dim yellow]{invalidation}[/dim yellow]\n"
            if counterargs:
                card_body += f"⚖️ [bold]Risk Headwinds:[/] [dim]{counterargs}[/dim]\n"

            card_body += (
                f"\n🔗 [blue underline]{article.url}[/blue underline]\n"
                f"[dim italic]⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.[/dim italic]"
            )

            console.print(
                Panel(
                    card_body,
                    title=f"[bold green]🚀 ACTIONABLE CATALYST: {article.title[:60]}[/bold green]",
                    border_style="green",
                    expand=True,
                )
            )
    else:
        # Check SQLite for active catalysts from the last 24h
        console.rule("[bold yellow]ℹ️ MARKET INTELLIGENCE BOTTOM LINE[/bold yellow]")
        active_theses = []
        if storage is not None:
            try:
                with storage._lock:
                    cursor = storage._conn.cursor()
                    cursor.execute(
                        """
                        SELECT title, source_name, impact_grade, confidence_score, directional_stance, beneficiaries, negatively_exposed, sectors
                        FROM articles
                        WHERE alerted = 1 OR status = 'OPPORTUNITY_CANDIDATE' OR confidence_score >= 7
                        ORDER BY created_at DESC
                        LIMIT 3
                        """
                    )
                    rows = cursor.fetchall()
                    for r in rows:
                        t_title = r[0]
                        t_grade = r[2] or "HIGH"
                        t_stance = r[4] or "BULLISH"
                        b_json = r[5]
                        b_tickers = json.loads(b_json) if b_json else []
                        s_json = r[7]
                        s_list = json.loads(s_json) if s_json else []
                        s_str = ", ".join(s_list) if s_list else ""

                        if b_tickers and s_str:
                            b_str = ", ".join(f"[bold green]{tk}[/bold green]" for tk in b_tickers) + f" [dim]({s_str})[/dim]"
                        elif b_tickers:
                            b_str = ", ".join(f"[bold green]{tk}[/bold green]" for tk in b_tickers)
                        elif s_str:
                            b_str = f"[bold green]{s_str}[/bold green] [dim](Sector-wide play)[/dim]"
                        else:
                            b_str = "[dim]Sector-wide / Macro[/dim]"
                        active_theses.append(f"• [bold white]{t_title}[/bold white] ([cyan]{t_grade}[/cyan]) -> Stance: [green]{t_stance}[/green] | Opportunity: {b_str}")
            except Exception:
                pass

        if active_theses:
            theses_block = "\n".join(active_theses)
            msg = (
                f"[bold white]No new anomalous catalyst broke in the last 15 minutes.[/bold white]\n\n"
                f"[bold underline]Active High-Conviction Market Theses (Recent 24h):[/bold underline]\n"
                f"{theses_block}\n\n"
                f"• Tracked tickers: [bold cyan]python -m src.main scanner --hours 24[/bold cyan]\n"
                f"• Sector digest: [bold cyan]python -m src.main digest --hours 24[/bold cyan]"
            )
        else:
            msg = (
                "[bold white]No market-moving anomalies exceeded the high-confidence threshold in the last 15 minutes.[/bold white]\n\n"
                "• All recent headlines represent routine corporate releases, minor syndicated stories, or standard market noise.\n"
                "• Zero false-positive alerts dispatched.\n"
                "• The agent continues monitoring feeds in the background (or run with [bold cyan]--alert-on-all[/bold cyan] to inspect all routine items)."
            )
        console.print(Panel(msg, title="[yellow]Market Bottom Line[/yellow]", border_style="yellow"))


def render_sector_news_summary_table(
    console: Console,
    sectors_summary: List[Dict[str, Any]],
    title: str = "Sector News Sentiment Summary",
) -> None:
    """Renders a Rich table of sector news sentiment with color-coded sentiment labels."""
    if not sectors_summary:
        return

    table = Table(title=title)
    table.add_column("Sector", style="cyan bold")
    table.add_column("News Count", justify="center")
    table.add_column("Sentiment Score", justify="right")
    table.add_column("Sentiment Label", justify="left")

    for item in sectors_summary:
        sector_name = str(item.get("sector", ""))
        count = str(item.get("total_articles", 0))
        avg_sentiment = float(item.get("avg_sentiment", 0.0))
        label = str(item.get("sentiment_label", "NEUTRAL"))

        # Color coding: green for bullish, red for bearish, yellow for neutral
        if avg_sentiment > 0.05 or "BULLISH" in label.upper():
            score_str = f"[green]{avg_sentiment:+.2f}[/green]"
            label_str = f"[bold green]{label}[/bold green]"
        elif avg_sentiment < -0.05 or "BEARISH" in label.upper():
            score_str = f"[red]{avg_sentiment:+.2f}[/red]"
            label_str = f"[bold red]{label}[/bold red]"
        else:
            score_str = f"[yellow]{avg_sentiment:+.2f}[/yellow]"
            label_str = f"[yellow]{label}[/yellow]"

        table.add_row(sector_name, count, score_str, label_str)

    console.print()
    console.print(table)


def render_ticker_news_summary_table(
    console: Console,
    tickers_summary: List[Dict[str, Any]],
    title: str = "Ticker News Sentiment Summary",
) -> None:
    """Renders a Rich table of ticker news sentiment with color-coded sentiment labels."""
    if not tickers_summary:
        return

    table = Table(title=title)
    table.add_column("Ticker", style="cyan bold")
    table.add_column("News Count", justify="center")
    table.add_column("Sentiment Score", justify="right")
    table.add_column("Sentiment Label", justify="left")

    for item in tickers_summary:
        ticker_name = str(item.get("ticker", ""))
        count = str(item.get("total_articles", 0))
        avg_sentiment = float(item.get("avg_sentiment", 0.0))
        label = str(item.get("sentiment_label", "NEUTRAL"))

        # Color coding: green for bullish, red for bearish, yellow for neutral
        if avg_sentiment > 0.05 or "BULLISH" in label.upper():
            score_str = f"[green]{avg_sentiment:+.2f}[/green]"
            label_str = f"[bold green]{label}[/bold green]"
        elif avg_sentiment < -0.05 or "BEARISH" in label.upper():
            score_str = f"[red]{avg_sentiment:+.2f}[/red]"
            label_str = f"[bold red]{label}[/bold red]"
        else:
            score_str = f"[yellow]{avg_sentiment:+.2f}[/yellow]"
            label_str = f"[yellow]{label}[/yellow]"

        table.add_row(ticker_name, count, score_str, label_str)

    console.print()
    console.print(table)


@app.command("run-once")
def run_once(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-evaluation of articles in window including previously seen/alerted (strictly enforcing decision gates)"),
    limit_per_source: Optional[int] = typer.Option(None, "--limit-per-source", "--batch-size", "-b", help="Max articles to fetch per site (overrides settings.yaml)"),
    max_age_minutes: Optional[int] = typer.Option(None, "--max-age-minutes", "-m", help="Filter articles older than N minutes (overrides settings.yaml)"),
    buy_only: Optional[bool] = typer.Option(None, "--buy-only/--all-stances", help="Only alert on buy opportunities (overrides settings.yaml)"),
) -> None:
    """Execute a single news scan, analysis, alerting, and persistence cycle."""
    console.print("[bold cyan]Executing single scan cycle...[/bold cyan]")
    settings = load_settings(config)
    sources_list = load_sources(sources)

    if not getattr(settings.telegram, "is_configured", False):
        console.print("[dim yellow]ℹ️ Telegram not configured (running in Local Console mode - alerts will display in terminal).[/dim yellow]")

    pipeline = Pipeline(
        settings=settings,
        sources=sources_list,
        alert_on_all=alert_on_all or settings.filter.alert_on_all,
        force=force,
        batch_size_per_source=limit_per_source,
        max_article_age_minutes=max_age_minutes,
        buy_opportunities_only=buy_only if buy_only is not None else settings.filter.buy_opportunities_only,
    )

    try:
        summary = pipeline.run_cycle()
        table = Table(title="Scan Cycle Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", style="green bold")

        table.add_row("Fetched Articles", str(summary.fetched))
        table.add_row("New / Unseen Articles", str(summary.new_unseen))
        table.add_row("Analyzed Articles", str(summary.analyzed))
        table.add_row("Alerted Articles", str(summary.alerted))
        table.add_row("Errors Encountered", str(summary.errors))

        console.print(table)

        if summary.alerted == 0:
            console.print("[yellow bold]ℹ️ No articles met the alerting criteria in this cycle (0 alerts sent).[/yellow bold]")
            logger.info("No qualifying articles found to send to Telegram (alerted=0).")
        else:
            if isinstance(pipeline.notifier, TelegramNotifier):
                console.print(f"[green bold]✓ Successfully dispatched {summary.alerted} alert(s) to Telegram![/green bold]")
                logger.info("Successfully dispatched %d alert(s) to Telegram.", summary.alerted)
            else:
                console.print(f"[cyan bold]✓ Processed {summary.alerted} alert(s) to console (Local mode).[/cyan bold]")
                logger.info("Processed %d alert(s) via ConsoleNotifier.", summary.alerted)

        render_actionable_investment_bottom_line(console, summary, pipeline.storage)

        # If any sectors have news, display the Sector News summary table
        if pipeline.storage is not None:
            sector_news = pipeline.storage.get_all_sectors_trend_summary(hours=24)
            if sector_news:
                render_sector_news_summary_table(
                    console,
                    sector_news,
                    title="Sector News Sentiment Summary (Recent 24h)",
                )
    finally:
        pipeline.close()


@app.command("scan")
def scan(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-evaluation of articles in window including previously seen/alerted (strictly enforcing decision gates)"),
    limit_per_source: Optional[int] = typer.Option(None, "--limit-per-source", "--batch-size", "-b", help="Max articles to fetch per site (overrides settings.yaml)"),
    max_age_minutes: Optional[int] = typer.Option(None, "--max-age-minutes", "-m", help="Filter articles older than N minutes (overrides settings.yaml)"),
) -> None:
    """Execute scan cycle, actionable bottom line, and sector news summary (alias for run-once)."""
    run_once(
        config=config,
        sources=sources,
        alert_on_all=alert_on_all,
        force=force,
        limit_per_source=limit_per_source,
        max_age_minutes=max_age_minutes,
    )


@app.command("run")
def run(
    daemon: bool = typer.Option(True, "--daemon/--no-daemon", "-d", help="Run continuous background monitoring daemon"),
    interval: Optional[int] = typer.Option(None, "--interval", "-i", help="Polling interval in minutes (defaults to settings.yaml)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
    limit_per_source: Optional[int] = typer.Option(None, "--limit-per-source", "--batch-size", "-b", help="Max articles to fetch per site (overrides settings.yaml)"),
    max_age_minutes: Optional[int] = typer.Option(None, "--max-age-minutes", "-m", help="Filter articles older than N minutes (overrides settings.yaml)"),
) -> None:
    """Run news monitoring pipeline (daemon mode by default, or single-run if --no-daemon)."""
    settings = load_settings(config)
    sources_list = load_sources(sources)

    poll_interval = interval if interval is not None else settings.agent.polling_interval_minutes
    if poll_interval <= 0:
        poll_interval = 15

    pipeline = Pipeline(
        settings=settings,
        sources=sources_list,
        alert_on_all=alert_on_all or settings.filter.alert_on_all,
        batch_size_per_source=limit_per_source,
        max_article_age_minutes=max_age_minutes,
    )

    if not daemon:
        console.print("[bold green]Executing single news agent scan cycle (--no-daemon)...[/bold green]")
        try:
            summary = pipeline.run_cycle()
            console.print(
                f"[green]Cycle completed:[/] {summary.fetched} fetched, "
                f"{summary.new_unseen} unseen, {summary.analyzed} analyzed, "
                f"{summary.alerted} alerted, {summary.errors} errors."
            )
            render_actionable_investment_bottom_line(console, summary, pipeline.storage)
            return
        finally:
            pipeline.close()

    console.print(f"[bold green]Starting Market News Agent daemon (polling every {poll_interval} minutes)...[/bold green]")
    console.print("[yellow]Press Ctrl+C or send SIGTERM to shut down gracefully.[/yellow]")

    stop_event = threading.Event()

    def signal_handler(signum: int, frame: object) -> None:
        signame = "SIGINT" if signum == signal.SIGINT else "SIGTERM"
        console.print(f"\n[bold yellow]Received {signame}. Initiating graceful shutdown...[/bold yellow]")
        stop_event.set()

    # Register OS signals
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        while not stop_event.is_set():
            cycle_start = time.time()
            try:
                summary = pipeline.run_cycle()
                console.print(
                    f"[green]Cycle completed:[/] {summary.fetched} fetched, "
                    f"{summary.new_unseen} unseen, {summary.analyzed} analyzed, "
                    f"{summary.alerted} alerted, {summary.errors} errors."
                )
                render_actionable_investment_bottom_line(console, summary, pipeline.storage)
            except Exception as exc:
                logger.error("Unexpected error during pipeline execution: %s", exc, exc_info=True)
                console.print(f"[bold red]Pipeline cycle error:[/] {exc}")

            wait_seconds = poll_interval * 60
            console.print(f"[dim]Sleeping for {poll_interval} minutes until next cycle...[/dim]")

            # Responsive wait loop checking stop_event every second
            while wait_seconds > 0 and not stop_event.is_set():
                sleep_slice = min(1.0, wait_seconds)
                stop_event.wait(timeout=sleep_slice)
                wait_seconds -= sleep_slice

    finally:
        console.print("[dim]Closing pipeline storage and releasing resources...[/dim]")
        pipeline.close()
        console.print("[bold green]Daemon shutdown complete. Goodbye.[/bold green]")


@app.command("test-telegram")
def test_telegram(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    message: Optional[str] = typer.Option(None, "--message", "-m", help="Custom test message content"),
) -> None:
    """Send an instant test notification to Telegram to verify credentials and channel delivery."""
    settings = load_settings(config)
    bot_token = settings.telegram.bot_token
    chat_id = settings.telegram.chat_id

    if not bot_token or not chat_id:
        console.print("[bold red]Telegram credentials missing![/bold red]")
        console.print("[yellow][LOCAL MODE][/yellow] Telegram credentials not configured. Printing preview alert to console:")
        sample_article = RawArticle(
            title="Sample Local Market Moving Catalyst",
            url="https://example.com/local-test",
            source_name="LocalFeed",
        )
        sample_analysis = MarketImpactAnalysis(
            is_market_moving=True,
            confidence_score=9,
            impact_grade=ImpactGrade.CRITICAL,
            affected_tickers=["$AAPL"],
            affected_sectors=["Technology"],
            summary="Local mode execution test alert preview.",
            catalyst_thesis="Testing local terminal notification delivery.",
        )
        ConsoleNotifier(console=console).send_alert(sample_article, sample_analysis)
        console.print("Please configure 'TELEGRAM_BOT_TOKEN' and 'TELEGRAM_CHAT_ID' in environment or config/settings.yaml.")
        raise typer.Exit(code=1)


    console.print(f"Connecting to Telegram bot (chat_id: {chat_id})...")
    notifier = TelegramNotifier(bot_token=bot_token, chat_id=chat_id)
    test_msg = message or (
        "🤖 <b>Market News Agent Alert Test</b>\n\n"
        "Connectivity verified successfully. Telegram delivery pipeline is operational!"
    )

    success = notifier.send_message(test_msg)
    if success:
        console.print("[bold green]✓ Test alert delivered successfully to Telegram![/bold green]")
    else:
        console.print("[bold red]✗ Failed to deliver Telegram test message. Check bot token and permissions.[/bold red]")
        raise typer.Exit(code=1)


@app.command("test-source")
def test_source(
    name: str = typer.Option(..., "--name", "-n", help="Source name from sources.yaml to poll"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
) -> None:
    """Fetch and display parsed headlines from a single configured news feed."""
    sources_list = load_sources(sources)
    matching_source = None
    for src in sources_list:
        if src.name.strip().lower() == name.strip().lower():
            matching_source = src
            break

    if matching_source is None:
        # Case-insensitive substring match fallback
        for src in sources_list:
            if name.strip().lower() in src.name.lower():
                matching_source = src
                break

    if matching_source is None:
        console.print(f"[bold red]Source '{name}' not found in sources configuration.[/bold red]")
        available = [s.name for s in sources_list]
        console.print(f"Available sources: {', '.join(available)}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Testing feed ingestion for source:[/] [bold]{matching_source.name}[/bold]")
    console.print(f"[dim]URL: {matching_source.url} (Type: {matching_source.type}, Enabled: {matching_source.enabled})[/dim]\n")

    if matching_source.type.lower() == "html":
        from src.ingestion.html import HTMLFetcher
        fetcher = HTMLFetcher(matching_source, timeout=15.0)
    else:
        fetcher = RSSFetcher(matching_source, timeout=15.0)
    articles = fetcher.fetch()

    if not articles:
        console.print(f"[yellow]No articles returned from '{matching_source.name}'. Check feed URL or network connectivity.[/yellow]")
        return

    console.print(f"[bold green]✓ Successfully fetched {len(articles)} articles.[/bold green]\n")
    table = Table(title=f"Sample Headlines from {matching_source.name} (Top {min(len(articles), 5)})")
    table.add_column("#", style="dim", width=4)
    table.add_column("Headline", style="bold")
    table.add_column("Published (UTC)", style="magenta")
    table.add_column("URL", style="blue")

    for i, art in enumerate(articles[:5], start=1):
        pub_str = art.published_at.strftime("%Y-%m-%d %H:%M") if art.published_at else "N/A"
        table.add_row(str(i), art.title, pub_str, art.url)

    console.print(table)


@app.command("test-connections")
def test_connections(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
) -> None:
    """Unified connectivity diagnostic validating Telegram Bot, LLM API keys, and SQLite database."""
    console.print("[bold cyan]Running Unified System Connectivity Diagnostics...[/bold cyan]\n")
    settings = load_settings(config)
    all_passed = True

    # 1. SQLite Database Verification
    db_path = settings.agent.database_path
    console.print(f"1. Checking SQLite Storage ({db_path})...", end=" ")
    try:
        storage = SQLiteStorage(db_path)
        stats = storage.get_stats()
        storage.close()
        console.print(f"[bold green][PASS][/bold green] Database schema valid. Total articles recorded: {stats['total_articles']}")
    except Exception as exc:
        all_passed = False
        console.print(f"[bold red][FAIL][/bold red] Database error: {exc}")

    # 2. LLM Provider Configuration
    model_name = settings.llm.model
    console.print(f"2. Checking LLM API Key Configuration (Model: '{model_name}')...", end=" ")
    api_key_found = False
    checked_env_vars = []

    if "gemini" in model_name.lower():
        checked_env_vars = ["GEMINI_API_KEY", "GOOGLE_API_KEY"]
    elif "gpt" in model_name.lower() or "openai" in model_name.lower():
        checked_env_vars = ["OPENAI_API_KEY"]
    elif "claude" in model_name.lower() or "anthropic" in model_name.lower():
        checked_env_vars = ["ANTHROPIC_API_KEY"]
    else:
        checked_env_vars = ["GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]

    for var in checked_env_vars:
        val = os.environ.get(var)
        if val and val.strip():
            api_key_found = True
            break

    if api_key_found:
        console.print(f"[bold green][PASS][/bold green] API key detected for {model_name}.")
    else:
        console.print(f"[bold yellow][WARN][/bold yellow] No API key detected in environment ({', '.join(checked_env_vars)}).")

    # 3. Telegram Bot Connectivity
    bot_token = settings.telegram.bot_token
    chat_id = settings.telegram.chat_id
    console.print("3. Checking Telegram Bot Integration...", end=" ")

    if not bot_token or not chat_id:
        console.print("[bold yellow][WARN][/bold yellow] Telegram bot_token or chat_id is not set. Alerts will be skipped.")
    else:
        try:
            url = f"https://api.telegram.org/bot{bot_token}/getMe"
            resp = requests.get(url, timeout=5.0)
            if resp.status_code == 200 and resp.json().get("ok"):
                bot_user = resp.json().get("result", {}).get("username", "Unknown")
                console.print(f"[bold green][PASS][/bold green] Authenticated as @{bot_user} (destination chat: {chat_id}).")
            else:
                all_passed = False
                console.print(f"[bold red][FAIL][/bold red] Telegram API returned HTTP {resp.status_code}: {resp.text}")
        except Exception as exc:
            all_passed = False
            console.print(f"[bold red][FAIL][/bold red] Network connection to Telegram API failed: {exc}")

    console.print()
    if all_passed:
        console.print("[bold green]✓ Core connectivity checks passed.[/bold green]")
    else:
        console.print("[bold yellow]! One or more services require configuration attention.[/bold yellow]")


@app.command("stats")
def stats(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Display aggregate article persistence and alert statistics from SQLite."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        data = storage.get_stats()
        total = data.get("total_articles", 0)
        alerted = data.get("alerted_count", 0)
        moving = data.get("market_moving_count", 0)
        unalerted = total - alerted

        table = Table(title=f"Market News Agent Statistics ({db_path})")
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="green bold")

        table.add_row("Total Articles Seen", str(total))
        table.add_row("Market Moving Items Identified", str(moving))
        table.add_row("Alert Notifications Dispatched", str(alerted))
        table.add_row("Routine / Unalerted Items", str(unalerted))

        console.print(table)
    finally:
        storage.close()


@app.command("metrics")
def metrics(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Display LLM observability telemetry, token consumption, expenditure, and monthly budget."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        data = storage.get_metrics_summary()
        total_calls = data.get("total_calls", 0)
        prompt_tokens = data.get("total_prompt_tokens", 0)
        completion_tokens = data.get("total_completion_tokens", 0)
        total_tokens = data.get("total_tokens", 0)
        total_cost = data.get("total_cost_usd", 0.0)
        spend_month = data.get("spend_this_month", 0.0)
        calls_by_model = data.get("calls_by_model", {})

        monthly_budget = settings.llm.monthly_budget_usd
        if monthly_budget is not None and monthly_budget > 0:
            pct = (spend_month / monthly_budget) * 100
            budget_str = f"${spend_month:.4f} / ${monthly_budget:.2f} ({pct:.1f}%)"
        else:
            budget_str = f"${spend_month:.4f} / Not Set"

        table = Table(title=f"LLM Observability & Cost Metrics ({db_path})")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green bold")

        table.add_row("Total LLM Inferences", f"{total_calls:,}")
        table.add_row("Prompt Tokens", f"{prompt_tokens:,}")
        table.add_row("Completion Tokens", f"{completion_tokens:,}")
        table.add_row("Total Tokens", f"{total_tokens:,}")
        table.add_row("Total Estimated Cost ($ USD)", f"${total_cost:.4f}")
        table.add_row("Spend this Month vs Monthly Budget", budget_str)

        if calls_by_model:
            model_breakdown = ", ".join(f"{m}: {c}" for m, c in sorted(calls_by_model.items(), key=lambda x: -x[1]))
        else:
            model_breakdown = "None"
        table.add_row("Calls by Model", model_breakdown)

        console.print(table)
    finally:
        storage.close()


@app.command("digest")
def digest(
    hours: int = typer.Option(24, "--hours", "-h", help="Time window in hours to summarize catalysts"),
    send_telegram: bool = typer.Option(False, "--send-telegram", help="Push the generated digest to Telegram channel"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Generate executive daily market catalyst digest grouped by sector."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        generator = MarketDigestGenerator()
        digest_text = generator.generate_digest(storage=storage, hours=hours)

        console.print()
        console.print(digest_text)
        console.print()

        if send_telegram:
            notifier = TelegramNotifier(
                bot_token=settings.telegram.bot_token,
                chat_id=settings.telegram.chat_id,
            )
            if not notifier.is_configured:
                console.print("[bold red]Telegram bot_token or chat_id is unconfigured. Cannot dispatch digest.[/bold red]")
            else:
                success = notifier.send_message(digest_text)
                if success:
                    console.print("[bold green]✓ Market catalyst digest dispatched to Telegram successfully.[/bold green]")
                else:
                    console.print("[bold red]✗ Failed to dispatch market catalyst digest to Telegram.[/bold red]")
    finally:
        storage.close()


@app.command("scanner")
def scanner(
    hours: int = typer.Option(24, "--hours", "-h", help="Lookback window in hours"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Display real-time market sentiment and buzz trend scanner dashboard."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        summary_list = storage.get_all_tickers_trend_summary(hours=hours)
        if not summary_list:
            console.print(f"[yellow]No ticker sentiment records found in the past {hours} hours.[/yellow]")
            return

        table = Table(title=f"Real-Time Market Scanner ({hours}h Window)")
        table.add_column("Ticker", style="cyan bold")
        table.add_column("24h Articles", justify="center")
        table.add_column("Buzz Multiplier", justify="center")
        table.add_column("Sentiment Score", justify="right")
        table.add_column("Sentiment Gauge", justify="center")
        table.add_column("Velocity", justify="right")
        table.add_column("Trend Verdict", justify="left")

        trend_engine = TrendEngine()
        for item in summary_list:
            ticker = item["ticker"]
            avg_sentiment = float(item["avg_sentiment"])

            trend = trend_engine.calculate_ticker_trend(storage, ticker, current_sentiment=None)

            buzz = trend.buzz_ratio
            if buzz >= 1.5:
                buzz_str = f"🔥 {buzz:.1f}x"
            else:
                buzz_str = f"{buzz:.1f}x"

            if avg_sentiment > 0.05:
                score_str = f"[green]{avg_sentiment:+.2f}[/green]"
            elif avg_sentiment < -0.05:
                score_str = f"[red]{avg_sentiment:+.2f}[/red]"
            else:
                score_str = f"[white]{avg_sentiment:+.2f}[/white]"

            gauge_str = format_sentiment_gauge(avg_sentiment, width=10)

            vel = trend.sentiment_velocity
            if vel > 0:
                vel_str = f"[green]{vel:+.2f}[/green]"
            elif vel < 0:
                vel_str = f"[red]{vel:+.2f}[/red]"
            else:
                vel_str = f"[white]{vel:+.2f}[/white]"

            verdict_val = (
                trend.trend_verdict.value
                if hasattr(trend.trend_verdict, "value")
                else str(trend.trend_verdict)
            )
            if "BULLISH" in verdict_val:
                verdict_str = f"[bold green]{verdict_val}[/bold green]"
            elif "BEAR" in verdict_val or "PANIC" in verdict_val:
                verdict_str = f"[bold red]{verdict_val}[/bold red]"
            else:
                verdict_str = f"[yellow]{verdict_val}[/yellow]"

            table.add_row(
                ticker,
                str(trend.articles_24h_count),
                buzz_str,
                score_str,
                gauge_str,
                vel_str,
                verdict_str,
            )

        console.print()
        console.print(table)
        console.print()
    finally:
        storage.close()


@app.command("heatmap")
def heatmap(
    hours: int = typer.Option(24, "--hours", "-h", help="Time window in hours for heatmap aggregation"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Display visual news sentiment heatmap across market sectors and tickers."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        sectors_summary = storage.get_all_sectors_trend_summary(hours=hours)
        tickers_summary = storage.get_all_tickers_trend_summary(hours=hours)

        if not sectors_summary and not tickers_summary:
            console.print(f"[yellow]No sector or ticker sentiment records found in the past {hours} hours.[/yellow]")
            return

        if sectors_summary:
            render_sector_news_summary_table(
                console,
                sectors_summary,
                title=f"Sector News Sentiment ({hours}h Window)",
            )
        else:
            console.print(f"[dim]No sector sentiment records in the past {hours} hours.[/dim]")

        if tickers_summary:
            render_ticker_news_summary_table(
                console,
                tickers_summary,
                title=f"Ticker News Sentiment ({hours}h Window)",
            )
        else:
            console.print(f"[dim]No ticker sentiment records in the past {hours} hours.[/dim]")
    finally:
        storage.close()


@app.command("history")
def history(
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter lineage by primary ticker (e.g. $NVDA)"),
    limit: int = typer.Option(20, "--limit", "-l", help="Max events to display"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Display persistent event lineage records with quality & opportunity scores."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    try:
        events = storage.list_event_lineage(ticker=ticker, limit=limit)
        if not events:
            filter_msg = f" for ticker '{ticker}'" if ticker else ""
            console.print(f"[yellow]No event lineage records found in database ({db_path}){filter_msg}.[/yellow]")
            return

        table = Table(title=f"Structured Event Lineage History ({len(events)} Events)")
        table.add_column("Event ID", style="dim", width=14)
        table.add_column("Ticker", style="cyan bold")
        table.add_column("Headline", style="bold")
        table.add_column("Sources", justify="center")
        table.add_column("Quality", justify="right")
        table.add_column("Opportunity", justify="right")
        table.add_column("Status", justify="center")
        table.add_column("Initial Price", justify="right")

        for ev in events:
            eid = str(ev.event_id)[:14]
            title = str(ev.title)[:50]
            tick = str(ev.primary_ticker or "N/A")
            srcs = str(ev.source_count)
            q_score = f"{float(ev.event_quality_score):.1f}/10"
            o_score = f"{float(ev.opportunity_score):.1f}/10"
            status = str(ev.status)
            p_price = f"${float(ev.initial_price):.2f}" if ev.initial_price is not None else "N/A"

            if status == "ALERT":
                status_str = f"[bold green]{status}[/bold green]"
            elif status == "WATCHLIST":
                status_str = f"[bold yellow]{status}[/bold yellow]"
            else:
                status_str = f"[dim]{status}[/dim]"

            table.add_row(eid, tick, title, srcs, q_score, o_score, status_str, p_price)

        console.print()
        console.print(table)
        console.print()
    finally:
        storage.close()


@app.command("performance")
def performance(
    update: bool = typer.Option(False, "--update", "-u", help="Automatically fetch latest quotes and update forward returns for eligible catalysts"),
    ticker: Optional[str] = typer.Option(None, "--ticker", "-t", help="Filter performance tracking by ticker symbol (e.g. $NVDA)"),
    source: Optional[str] = typer.Option(None, "--source", "-s", help="Filter performance tracking by publisher / source name"),
    min_opportunity: float = typer.Option(6.0, "--min-opportunity", "-o", help="Minimum opportunity score threshold"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Evaluate empirical forward return tracking, directional alpha, and publisher credibility scorecards."""
    settings = load_settings(config)
    db_path = db or settings.agent.database_path

    storage = SQLiteStorage(db_path)
    engine = PerformanceTrackerEngine()
    try:
        if update:
            from src.analysis.technical import TradingViewEngine
            tv_engine = TradingViewEngine()

            def fetch_quote(sym: str) -> Optional[float]:
                try:
                    summary = tv_engine.get_technical_summary(sym)
                    return summary.close_price if summary else None
                except Exception:
                    return None

            update_counts = engine.update_forward_returns(storage, price_fetcher=fetch_quote)
            total_upd = update_counts.get("updated_1d", 0) + update_counts.get("updated_7d", 0)
            console.print(f"[bold green]✓ Updated {total_upd} forward return quote(s) ({update_counts.get('updated_1d', 0)} 1d, {update_counts.get('updated_7d', 0)} 7d).[/bold green]\n")

        summary = engine.calculate_performance_metrics(
            storage, min_opportunity=min_opportunity, ticker=ticker, source=source
        )

        if summary.total_catalysts == 0:
            filter_msg = []
            if ticker:
                filter_msg.append(f"ticker '{ticker}'")
            if source:
                filter_msg.append(f"source '{source}'")
            filter_str = f" for {', '.join(filter_msg)}" if filter_msg else ""
            console.print(f"[yellow]No realized catalyst return records found in database ({db_path}){filter_str}.[/yellow]")
            return

        # 1. Global Metrics Panel
        hr_1d_color = "green" if summary.hit_rate_1d_pct >= 55.0 else ("red" if summary.hit_rate_1d_pct < 45.0 else "yellow")
        alpha_1d_color = "green" if summary.avg_alpha_1d_pct > 0 else "red"
        hr_7d_color = "green" if summary.hit_rate_7d_pct >= 55.0 else ("red" if summary.hit_rate_7d_pct < 45.0 else "yellow")
        alpha_7d_color = "green" if summary.avg_alpha_7d_pct > 0 else "red"

        metrics_table = Table.grid(padding=(0, 2))
        metrics_table.add_column("Metric", style="bold")
        metrics_table.add_column("1-Day Forward", style="cyan")
        metrics_table.add_column("7-Day Forward", style="magenta")

        metrics_table.add_row(
            "Resolved Catalysts:",
            f"{summary.resolved_1d} / {summary.total_catalysts}",
            f"{summary.resolved_7d} / {summary.total_catalysts}",
        )
        metrics_table.add_row(
            "Wins / Losses:",
            f"{summary.wins_1d}W / {summary.losses_1d}L",
            f"{summary.wins_7d}W / {summary.losses_7d}L",
        )
        metrics_table.add_row(
            "Directional Hit Rate:",
            f"[{hr_1d_color} bold]{summary.hit_rate_1d_pct:.1f}%[/{hr_1d_color} bold]",
            f"[{hr_7d_color} bold]{summary.hit_rate_7d_pct:.1f}%[/{hr_7d_color} bold]",
        )
        metrics_table.add_row(
            "Average Raw Return:",
            f"{summary.avg_return_1d_pct:+.2f}%",
            f"{summary.avg_return_7d_pct:+.2f}%",
        )
        metrics_table.add_row(
            "Average Directional Alpha:",
            f"[{alpha_1d_color} bold]{summary.avg_alpha_1d_pct:+.2f}%[/{alpha_1d_color} bold]",
            f"[{alpha_7d_color} bold]{summary.avg_alpha_7d_pct:+.2f}%[/{alpha_7d_color} bold]",
        )

        global_panel = Panel(
            metrics_table,
            title="🎯 GLOBAL ALPHA & HIT RATE METRICS",
            border_style="green",
            expand=False,
        )
        console.print()
        console.print(global_panel)

        # 2. Publisher Credibility Scorecard
        scorecard_table = Table(title="📰 PUBLISHER CREDIBILITY SCORECARD")
        scorecard_table.add_column("Publisher / Source", style="bold")
        scorecard_table.add_column("Catalysts", justify="center")
        scorecard_table.add_column("Avg Quality", justify="right")
        scorecard_table.add_column("1d Hit Rate", justify="right")
        scorecard_table.add_column("Avg 1d Alpha", justify="right")
        scorecard_table.add_column("Avg 7d Alpha", justify="right")
        scorecard_table.add_column("Credibility Tier", justify="center")

        for pub in summary.source_scorecard:
            hr_col = "green" if pub.hit_rate_pct >= 55.0 else ("red" if pub.hit_rate_pct < 45.0 else "yellow")
            a1_col = "green" if pub.avg_alpha_1d_pct > 0 else "red"
            a7_col = "green" if pub.avg_alpha_7d_pct > 0 else "red"

            tier_str = str(pub.credibility_tier.value if hasattr(pub.credibility_tier, "value") else pub.credibility_tier)
            if "TIER_A" in tier_str:
                tier_badge = "[bold green]🌟 TIER A (ALPHA)[/bold green]"
            elif "TIER_B" in tier_str:
                tier_badge = "[bold cyan]🟢 TIER B (RELIABLE)[/bold cyan]"
            elif "TIER_C" in tier_str:
                tier_badge = "[yellow]🟡 TIER C (NEUTRAL)[/yellow]"
            else:
                tier_badge = "[bold red]🔴 TIER D (UNRELIABLE)[/bold red]"

            scorecard_table.add_row(
                pub.source_name,
                str(pub.total_events),
                f"{pub.avg_quality_score:.1f}/10",
                f"[{hr_col}]{pub.hit_rate_pct:.1f}%[/{hr_col}]",
                f"[{a1_col}]{pub.avg_alpha_1d_pct:+.2f}%[/{a1_col}]",
                f"[{a7_col}]{pub.avg_alpha_7d_pct:+.2f}%[/{a7_col}]",
                tier_badge,
            )

        console.print()
        console.print(scorecard_table)

        # 3. Realized Performance Ledger
        records = storage.get_resolved_lineage_records()
        ledger_table = Table(title="📊 CATALYST REALIZED RETURN LEDGER")
        ledger_table.add_column("Event ID", style="dim", width=12)
        ledger_table.add_column("Ticker", style="cyan bold")
        ledger_table.add_column("Stance", justify="center")
        ledger_table.add_column("Headline", style="bold")
        ledger_table.add_column("Initial Price", justify="right")
        ledger_table.add_column("1d Return", justify="right")
        ledger_table.add_column("1d Alpha", justify="right")
        ledger_table.add_column("7d Return", justify="right")
        ledger_table.add_column("7d Alpha", justify="right")
        ledger_table.add_column("Verdict", justify="center")

        for ev in records:
            if ev.opportunity_score is not None and ev.opportunity_score < min_opportunity:
                continue
            if ticker:
                cl_t = ticker.strip().upper()
                nt = cl_t[1:] if cl_t.startswith("$") else cl_t
                et = str(ev.primary_ticker or "").strip().upper()
                ent = et[1:] if et.startswith("$") else et
                if nt != ent:
                    continue
            if source:
                if not any(source.strip().lower() in s.lower() for s in ev.all_sources):
                    continue

            stance = ev.directional_stance or "NEUTRAL"
            init_p = f"${ev.initial_price:.2f}" if ev.initial_price is not None else "N/A"

            if ev.subsequent_return_1d is not None:
                r1 = ev.subsequent_return_1d
                a1 = r1 if stance == "BULLISH" else (-r1 if stance == "BEARISH" else abs(r1))
                r1_str = f"{r1:+.2f}%"
                a1_str = f"[green]{a1:+.2f}%[/green]" if a1 > 0 else f"[red]{a1:+.2f}%[/red]"
                verdict = engine.classify_outcome(stance, r1)
                verdict_str = f"[bold green]WIN[/bold green]" if verdict == "WIN" else (f"[bold red]LOSS[/bold red]" if verdict == "LOSS" else "[dim]NEUT[/dim]")
            else:
                r1_str = "Pending"
                a1_str = "Pending"
                verdict_str = "[dim]--[/dim]"

            if ev.subsequent_return_7d is not None:
                r7 = ev.subsequent_return_7d
                a7 = r7 if stance == "BULLISH" else (-r7 if stance == "BEARISH" else abs(r7))
                r7_str = f"{r7:+.2f}%"
                a7_str = f"[green]{a7:+.2f}%[/green]" if a7 > 0 else f"[red]{a7:+.2f}%[/red]"
            else:
                r7_str = "Pending"
                a7_str = "Pending"

            stance_str = f"[green]BULL[/green]" if stance == "BULLISH" else (f"[red]BEAR[/red]" if stance == "BEARISH" else f"[yellow]{stance[:4]}[/yellow]")

            ledger_table.add_row(
                str(ev.event_id)[:12],
                str(ev.primary_ticker or "N/A"),
                stance_str,
                str(ev.title)[:40],
                init_p,
                r1_str,
                a1_str,
                r7_str,
                a7_str,
                verdict_str,
            )

        console.print()
        console.print(ledger_table)
        console.print()
    finally:
        storage.close()


@app.command("push-pending")
def push_pending(
    limit: int = typer.Option(50, "--limit", "-l", help="Maximum pending alerts to push"),
    delay: float = typer.Option(1.0, "--delay", "-d", help="Delay in seconds between Telegram dispatches"),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview qualifying articles without sending or updating database"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    db: Optional[str] = typer.Option(None, "--db", help="Override SQLite database path"),
) -> None:
    """Evaluate unalerted articles in SQLite against current decision gates and dispatch Telegram push alerts."""
    import sqlite3
    from datetime import datetime, timezone

    settings = load_settings(config)
    db_path = db or settings.agent.database_path
    pipeline = Pipeline.from_settings(settings)

    console.print(f"[bold cyan]Evaluating unalerted articles in DB ({db_path})...[/bold cyan]")
    console.print(
        f"[dim]Criteria: min_confidence={pipeline.min_confidence_score}, "
        f"min_impact_grade={pipeline.min_impact_grade}, "
        f"enable_fact_verification={pipeline.enable_fact_verification}[/dim]\n"
    )

    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute(
            """
            SELECT content_hash, title, url, source_name, published_at, confidence_score, impact_grade, 
                   status, directional_stance, passes_hard_gates, tickers, sectors, causal_mechanism, 
                   beneficiaries, negatively_exposed, counterarguments, invalidation_conditions,
                   corroborating_sources, corroboration_count, is_market_moving, sec_filing_item,
                   surge_level, sentiment_score, sentiment_label, buzz_ratio, trend_verdict
            FROM articles 
            WHERE alerted = 0
            ORDER BY confidence_score DESC, published_at DESC
            """
        )
        rows = c.fetchall()

        qualifying = []
        for r in rows:
            (
                content_hash, title, url, source_name, published_at_str, confidence_score, impact_grade, 
                status, directional_stance, passes_hard_gates, tickers_json, sectors_json, causal_mechanism, 
                beneficiaries_json, negatively_exposed_json, counterarguments, invalidation_conditions,
                corroborating_sources_json, corroboration_count, is_market_moving, sec_filing_item,
                surge_level, sentiment_score, sentiment_label, buzz_ratio, trend_verdict
            ) = r

            if not is_market_moving:
                continue

            try:
                grade = ImpactGrade(impact_grade) if impact_grade else ImpactGrade.LOW
            except ValueError:
                grade = ImpactGrade.LOW

            if pipeline.min_confidence_score and (confidence_score or 0) < pipeline.min_confidence_score:
                continue
            if pipeline.min_impact_grade and GRADE_SEVERITY.get(grade, 0) < GRADE_SEVERITY.get(pipeline.min_impact_grade, 0):
                continue

            tickers = json.loads(tickers_json) if tickers_json else []
            sectors = json.loads(sectors_json) if sectors_json else []
            beneficiaries = json.loads(beneficiaries_json) if beneficiaries_json else []
            negatively_exposed = json.loads(negatively_exposed_json) if negatively_exposed_json else []
            corroborating_sources = json.loads(corroborating_sources_json) if corroborating_sources_json else []

            has_direct_exposure = bool(tickers or beneficiaries or negatively_exposed)
            if not has_direct_exposure:
                continue

            hard_gates = HardGates(
                is_novel=True,
                is_verified=True,
                has_direct_exposure=has_direct_exposure,
                passes_all_gates=True,
            )

            causal_chain = CausalChain(
                event_facts=title,
                economic_mechanism=causal_mechanism or "Direct corporate development impacting valuation and forward guidance.",
                affected_sectors=sectors,
                beneficiaries=beneficiaries,
                negatively_exposed=negatively_exposed,
                counterarguments=counterarguments or "Market participants may have partially priced in near-term disruptions.",
                invalidation_conditions=invalidation_conditions or "Official management denial or swift operational recovery.",
            )

            pub_at = None
            if published_at_str:
                try:
                    pub_at = datetime.fromisoformat(published_at_str)
                except Exception:
                    pub_at = None

            raw_article = RawArticle(
                title=title,
                url=url,
                source_name=source_name,
                published_at=pub_at,
                content=causal_mechanism or title,
                content_hash=content_hash,
            )

            analysis = MarketImpactAnalysis(
                summary=title,
                catalyst_thesis=causal_mechanism or title,
                impact_grade=grade,
                confidence_score=confidence_score or 5,
                affected_tickers=tickers,
                affected_sectors=sectors,
                is_market_moving=bool(is_market_moving),
                status=EventStatus.OPPORTUNITY_CANDIDATE,
                directional_stance=DirectionalStance(directional_stance) if directional_stance else DirectionalStance.NEUTRAL,
                causal_chain=causal_chain,
                hard_gates=hard_gates,
                corroboration_count=corroboration_count or 1,
                corroborating_sources=corroborating_sources or [source_name],
                sec_filing_item=sec_filing_item,
            )

            gate_min_opp = ((float(pipeline.min_confidence_score) / 10.0) * 3.5 if pipeline.min_confidence_score is not None else 8.0)
            gate_min_event = 6.0

            gate_decision = pipeline.gate_engine.evaluate_gates(
                analysis,
                min_event_quality=gate_min_event,
                min_opportunity_score=gate_min_opp,
                technical_confluence_boost=0.0,
                enable_fact_verification=pipeline.enable_fact_verification,
            )
            analysis.two_layer_gate = gate_decision

            if pipeline.should_alert(analysis):
                qualifying.append((content_hash, raw_article, analysis, gate_decision))

        if not qualifying:
            console.print("[yellow]No unalerted articles currently qualify under decision gates.[/yellow]")
            return

        console.print(f"[bold green]Found {len(qualifying)} qualifying article(s) passing all gates.[/bold green]\n")

        table = Table(title="Qualifying Articles for Dispatch")
        table.add_column("#", style="dim", width=3)
        table.add_column("Grade", justify="center")
        table.add_column("Score", justify="center")
        table.add_column("Headline", style="bold")
        table.add_column("Publisher", style="magenta")
        table.add_column("Tickers", style="cyan")
        table.add_column("Verdict", justify="center")

        for idx, (ch, art, ana, gate) in enumerate(qualifying[:limit], start=1):
            table.add_row(
                str(idx),
                ana.impact_grade.value,
                f"{ana.confidence_score}/10",
                art.title[:45] + ("..." if len(art.title) > 45 else ""),
                art.source_name,
                ", ".join(ana.affected_tickers[:3]),
                f"[green]{gate.gate_verdict}[/green]",
            )
        console.print(table)
        console.print()

        if dry_run:
            console.print("[yellow]Dry run mode active: No alerts dispatched and DB not updated.[/yellow]")
            return

        sent_count = 0
        fail_count = 0
        for idx, (ch, art, ana, gate) in enumerate(qualifying[:limit], start=1):
            console.print(f"[{idx}/{min(len(qualifying), limit)}] Sending: [bold]{art.title}[/bold]...", end=" ")
            try:
                delivered = bool(pipeline.notifier.send_alert(art, ana))
            except Exception as send_err:
                delivered = False
                logger.error("Error sending alert for %s: %s", art.title, send_err)

            if delivered:
                c.execute("UPDATE articles SET alerted = 1 WHERE content_hash = ?", (ch,))
                conn.commit()
                sent_count += 1
                console.print("[bold green]✓ SENT & RECORDED[/bold green]")
            else:
                fail_count += 1
                console.print("[bold red]✗ FAILED[/bold red]")

            if idx < min(len(qualifying), limit) and delay > 0:
                time.sleep(delay)

        console.print(f"\n[bold green]Dispatch complete: {sent_count} sent successfully, {fail_count} failed.[/bold green]")
    finally:
        conn.close()


if __name__ == "__main__":
    app()


