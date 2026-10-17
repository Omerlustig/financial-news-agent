import html
import re
import sys
from typing import Any, Dict, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    SentimentLabel,
    SurgeLevel,
)
from src.notification.base import BaseNotifier


def _strip_html(raw_html: str) -> str:
    """Strip basic HTML markup tags and unescape entities for terminal rendering."""
    if not raw_html:
        return ""
    clean = re.sub(r"<[^>]+>", "", raw_html)
    return html.unescape(clean)


class ConsoleNotifier(BaseNotifier):
    """Local terminal notifier for developer testing and local execution without Telegram keys."""

    def __init__(self, console: Optional[Console] = None) -> None:
        self.console = console or Console()

    def send_alert(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        """Format and print an institutional news catalyst alert directly to the terminal."""
        grade_name = (
            getattr(analysis.impact_grade, "value", str(analysis.impact_grade)).upper()
            if analysis.impact_grade
            else "UNKNOWN"
        )
        score = analysis.confidence_score if analysis.confidence_score is not None else 0

        # Title / Header
        surge_badge = " [BREAKING CATALYST SURGE]" if getattr(analysis, "surge_level", None) == SurgeLevel.SURGE_BREAKING else ""
        header_title = f"🚨 OPPORTUNITY CANDIDATE{surge_badge} | Score: {score}/10 | {grade_name}"

        table = Table.grid(padding=(0, 1))
        table.add_column("Key", style="bold cyan", justify="right", width=20)
        table.add_column("Value", style="white")

        table.add_row("Headline:", article.title)
        table.add_row("Source:", f"{article.source_name} ({article.published_at or 'N/A'})")

        # Stance & Time Horizon
        stance = getattr(analysis, "directional_stance", None)
        stance_str = getattr(stance, "value", str(stance)) if stance else "NEUTRAL"
        horizon = getattr(analysis, "time_horizon", None)
        horizon_str = getattr(horizon, "value", str(horizon)) if horizon else "SHORT_TERM"
        table.add_row("Stance / Horizon:", f"{stance_str} | {horizon_str}")

        # Sectors
        sectors = []
        if analysis.causal_chain and analysis.causal_chain.affected_sectors:
            sectors = analysis.causal_chain.affected_sectors
        elif analysis.affected_sectors:
            sectors = analysis.affected_sectors
        table.add_row("Sectors:", ", ".join(sectors) if sectors else "None")

        # SEC / Regulatory Filing
        if getattr(analysis, "sec_filing_item", None):
            filing_val = getattr(analysis.sec_filing_item, "value", str(analysis.sec_filing_item))
            table.add_row("Regulatory Filing:", filing_val)

        # Multi-source corroboration
        corroborating = getattr(analysis, "corroborating_sources", None) or []
        if len(corroborating) >= 2:
            table.add_row("Corroboration:", f"{', '.join(corroborating)} ({len(corroborating)} sources)")

        # Sentiment Score & Gauge
        from src.notification.telegram import format_sentiment_gauge
        sentiment_score = (
            float(analysis.sentiment.score)
            if analysis.sentiment and analysis.sentiment.score is not None
            else 0.0
        )
        label_val = (
            getattr(analysis.sentiment.label, "value", str(analysis.sentiment.label))
            if analysis.sentiment and analysis.sentiment.label
            else "NEUTRAL"
        )
        gauge = format_sentiment_gauge(sentiment_score, width=10)
        table.add_row("Sentiment:", f"{label_val} | {gauge}")

        # Trend & Buzz metrics
        if getattr(analysis, "trend", None) is not None and analysis.trend is not None:
            buzz_str = f"{analysis.trend.buzz_ratio:.1f}x (24h Articles: {analysis.trend.articles_24h_count})"
            verdict_val = getattr(analysis.trend.trend_verdict, "value", str(analysis.trend.trend_verdict))
            vel_str = f"{verdict_val} (Velocity: {analysis.trend.sentiment_velocity:+.2f})"
            table.add_row("News Buzz:", buzz_str)
            table.add_row("Trend Verdict:", vel_str)

        table.add_row("", "")
        summary = analysis.summary or (analysis.causal_chain.event_facts if analysis.causal_chain else "")
        table.add_row("Executive Summary:", summary or "None")

        mechanism = (
            analysis.causal_chain.economic_mechanism
            if analysis.causal_chain and analysis.causal_chain.economic_mechanism
            else (analysis.catalyst_thesis or "None")
        )
        table.add_row("Causal Mechanism:", mechanism)

        # Beneficiaries vs Negatively Exposed
        beneficiaries = analysis.causal_chain.beneficiaries if analysis.causal_chain else analysis.affected_tickers
        negatively_exposed = analysis.causal_chain.negatively_exposed if analysis.causal_chain else []
        table.add_row("🟢 Beneficiaries:", ", ".join(beneficiaries) if beneficiaries else "None")
        table.add_row("🔴 Negatively Exp:", ", ".join(negatively_exposed) if negatively_exposed else "None")

        rec = getattr(analysis, "recommendation", None)
        if rec:
            table.add_row("💡 Recommendation:", rec)

        table.add_row("🔗 Article URL:", article.url)
        table.add_row("", "")
        table.add_row("⚠️ Disclaimer:", analysis.disclaimer)

        panel = Panel(
            table,
            title=f"[bold green]{header_title}[/bold green]",
            subtitle="[dim]ConsoleNotifier (Local Mode)[/dim]",
            border_style="cyan",
            expand=False,
        )
        self.console.print()
        self.console.print(panel)
        self.console.print()
        return True

    def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        """Strip HTML tags and print message payload directly to the terminal."""
        cleaned_text = _strip_html(text)
        self.console.print()
        self.console.print(Panel(cleaned_text, title="[bold blue]News Agent Message[/bold blue]", border_style="blue"))
        self.console.print()
        return True
