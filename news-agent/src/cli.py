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
from src.pipeline import Pipeline
from src.storage.sqlite import SQLiteStorage
from src.analysis.digest import MarketDigestGenerator
from src.analysis.trend import TrendEngine
from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade


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

            if not beneficiaries and analysis.affected_tickers:
                stance_str = str(analysis.directional_stance)
                if "BULLISH" in stance_str:
                    beneficiaries = analysis.affected_tickers
                elif "BEARISH" in stance_str:
                    negatively_exposed = analysis.affected_tickers
                else:
                    beneficiaries = analysis.affected_tickers

            sectors_str = ", ".join(analysis.affected_sectors) if analysis.affected_sectors else "Broader Market / Macro"
            beneficiaries_str = ", ".join(f"[bold green]{t}[/bold green]" for t in beneficiaries) if beneficiaries else "[italic dim]None direct[/italic dim]"
            neg_str = ", ".join(f"[bold red]{t}[/bold red]" for t in negatively_exposed) if negatively_exposed else "[italic dim]None direct[/italic dim]"

            stance_str = str(analysis.directional_stance).replace("DirectionalStance.", "")
            stance_color = "green" if "BULLISH" in stance_str else ("red" if "BEARISH" in stance_str else "yellow")
            stance_icon = "📈" if "BULLISH" in stance_str else ("📉" if "BEARISH" in stance_str else "⚖️")

            grade_val = analysis.impact_grade.value if hasattr(analysis.impact_grade, "value") else str(analysis.impact_grade)
            grade_color = "red bold" if grade_val == "CRITICAL" else ("yellow bold" if grade_val == "HIGH" else "cyan")

            card_body = (
                f"[bold white]{article.title}[/bold white]\n"
                f"[dim]Source: {article.source_name} | Published: {article.published_at or 'Breaking'} | Grade: [{grade_color}]{grade_val}[/{grade_color}] (Score: {analysis.confidence_score}/10)[/dim]\n\n"
                f"{stance_icon} [bold]Directional Stance:[/] [{stance_color} bold]{stance_str}[/{stance_color} bold]  |  ⏱ [bold]Horizon:[/] {str(analysis.time_horizon or 'SHORT_TERM (1-7 days)').replace('TimeHorizon.', '')}\n"
                f"🏭 [bold]Sectors in Play:[/] [cyan]{sectors_str}[/cyan]\n\n"
                f"🟢 [bold green]STOCKS TO INVEST IN (BENEFICIARIES):[/bold green] {beneficiaries_str}\n"
                f"🔴 [bold red]STOCKS TO AVOID / SHORT (EXPOSED):[/bold red] {neg_str}\n\n"
                f"💡 [bold underline]Causal Transmission Thesis:[/bold underline]\n"
                f"[dim white]{mechanism or analysis.rationale or article.summary}[/dim white]\n"
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
                        b_str = ", ".join(f"[bold green]{tk}[/bold green]" for tk in b_tickers) if b_tickers else "[dim]Sector-wide[/dim]"
                        active_theses.append(f"• [bold white]{t_title}[/bold white] ([cyan]{t_grade}[/cyan]) -> Stance: [green]{t_stance}[/green] | Beneficiaries: {b_str}")
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


@app.command("run-once")
def run_once(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
    max_age_minutes: Optional[int] = typer.Option(None, "--max-age-minutes", "-m", help="Filter articles older than N minutes (overrides settings.yaml)"),
) -> None:
    """Execute a single news scan, analysis, alerting, and persistence cycle."""
    console.print("[bold cyan]Executing single scan cycle...[/bold cyan]")
    settings = load_settings(config)
    sources_list = load_sources(sources)

    pipeline = Pipeline(
        settings=settings,
        sources=sources_list,
        alert_on_all=alert_on_all or settings.filter.alert_on_all,
        max_article_age_minutes=max_age_minutes,
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
        render_actionable_investment_bottom_line(console, summary, pipeline.storage)
    finally:
        pipeline.close()


@app.command("run")
def run(
    daemon: bool = typer.Option(True, "--daemon/--no-daemon", "-d", help="Run continuous background monitoring daemon"),
    interval: Optional[int] = typer.Option(None, "--interval", "-i", help="Polling interval in minutes (defaults to settings.yaml)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
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


if __name__ == "__main__":
    app()


