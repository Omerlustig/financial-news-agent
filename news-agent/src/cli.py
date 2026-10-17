import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import requests
import typer
from rich.console import Console
from rich.table import Table

# Ensure news-agent root is in sys.path
news_agent_root = Path(__file__).resolve().parent.parent
if str(news_agent_root) not in sys.path:
    sys.path.insert(0, str(news_agent_root))

from src.config import load_settings, load_sources, Settings
from src.ingestion.rss import RSSFetcher
from src.notification.telegram import TelegramNotifier
from src.pipeline import Pipeline
from src.storage.sqlite import SQLiteStorage
from src.analysis.digest import MarketDigestGenerator

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


@app.command("run-once")
def run_once(
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
) -> None:
    """Execute a single news scan, analysis, alerting, and persistence cycle."""
    console.print("[bold cyan]Executing single scan cycle...[/bold cyan]")
    settings = load_settings(config)
    sources_list = load_sources(sources)

    pipeline = Pipeline(
        settings=settings,
        sources=sources_list,
        alert_on_all=alert_on_all or settings.filter.alert_on_all,
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
    finally:
        pipeline.close()


@app.command("run")
def run(
    daemon: bool = typer.Option(True, "--daemon/--no-daemon", "-d", help="Run continuous background monitoring daemon"),
    interval: Optional[int] = typer.Option(None, "--interval", "-i", help="Polling interval in minutes (defaults to settings.yaml)"),
    config: Optional[str] = typer.Option(None, "--config", "-c", help="Path to settings.yaml"),
    sources: Optional[str] = typer.Option(None, "--sources", "-s", help="Path to sources.yaml"),
    alert_on_all: bool = typer.Option(False, "--alert-on-all", help="Bypass impact and confidence thresholds for testing"),
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


if __name__ == "__main__":
    app()

