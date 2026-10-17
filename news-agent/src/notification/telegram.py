import html
import logging
import time
from typing import Optional

import requests

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.notification.base import BaseNotifier

logger = logging.getLogger(__name__)

GRADE_BADGES = {
    ImpactGrade.CRITICAL: ("🚨", "CRITICAL"),
    ImpactGrade.HIGH: ("⚡", "HIGH"),
    ImpactGrade.AVERAGE: ("📊", "AVERAGE"),
    ImpactGrade.LOW: ("ℹ️", "LOW"),
}


def escape_html(text: Optional[str]) -> str:
    """Safely escape text for Telegram HTML parse mode.

    Replaces &, <, and > with their respective HTML entities to prevent
    Telegram parsing errors.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=False)


def format_telegram_message(
    article: RawArticle,
    analysis: MarketImpactAnalysis,
    directional_stance: Optional[str] = None,
) -> str:
    """Format a market impact analysis into a Telegram HTML message.

    Includes severity badge, score, directional stance (if present),
    impacted tickers, affected sectors, headline, source metadata,
    executive summary, analyst thesis, and direct source link.
    """
    badge_icon, grade_name = GRADE_BADGES.get(
        analysis.impact_grade,
        ("🔔", getattr(analysis.impact_grade, "value", str(analysis.impact_grade))),
    )
    score = analysis.confidence_score

    # Check for directional stance
    stance = directional_stance or getattr(analysis, "directional_stance", None) or getattr(analysis, "stance", None)
    stance_line = ""
    if stance:
        stance_str = str(stance).strip()
        lower_stance = stance_str.lower()
        if any(w in lower_stance for w in ["bull", "up", "pos", "long", "call"]):
            stance_icon = "📈"
        elif any(w in lower_stance for w in ["bear", "down", "neg", "short", "put"]):
            stance_icon = "📉"
        else:
            stance_icon = "⚖️"
        stance_line = f"{stance_icon} <b>Stance:</b> {escape_html(stance_str)}\n"

    # Publication date formatting
    if article.published_at:
        pub_time = article.published_at.strftime("%Y-%m-%d %H:%M UTC")
    else:
        pub_time = "N/A"

    # Tickers and Sectors
    tickers_str = ", ".join(analysis.affected_tickers) if analysis.affected_tickers else "None"
    sectors_str = ", ".join(analysis.affected_sectors) if analysis.affected_sectors else "None"

    # Escape dynamic text fields
    escaped_title = escape_html(article.title)
    escaped_source = escape_html(article.source_name)
    escaped_tickers = escape_html(tickers_str)
    escaped_sectors = escape_html(sectors_str)
    escaped_summary = escape_html(analysis.summary)
    escaped_thesis = escape_html(analysis.catalyst_thesis)
    escaped_url_attr = html.escape(article.url, quote=True)
    escaped_url_text = escape_html(article.url)

    lines = [
        f"{badge_icon} <b>[{grade_name} | Score: {score}/10]</b>",
    ]
    if stance_line:
        lines.append(stance_line.strip())

    lines.extend([
        f"<b>Headline:</b> {escaped_title}",
        f"<b>Source:</b> {escaped_source} | <b>Published:</b> {pub_time}",
        f"<b>Tickers:</b> {escaped_tickers}",
        f"<b>Sectors:</b> {escaped_sectors}",
        "",
        "<b>Executive Summary:</b>",
        escaped_summary,
        "",
        "<b>Analyst Thesis:</b>",
        escaped_thesis,
        "",
        f"🔗 <a href=\"{escaped_url_attr}\">{escaped_url_text}</a>",
    ])

    return "\n".join(lines)


class TelegramNotifier(BaseNotifier):
    """Dispatches market news alerts to Telegram channels or private chats via Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        parse_mode: str = "HTML",
    ) -> None:
        self.bot_token = bot_token.strip() if bot_token else None
        self.chat_id = chat_id.strip() if chat_id else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.parse_mode = parse_mode

    @property
    def is_configured(self) -> bool:
        """Check if bot token and target chat ID are provided and non-empty."""
        return bool(self.bot_token and self.chat_id)

    def send_alert(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        """Format and dispatch an alert for a market-moving event.

        Args:
            article: Normalized article metadata.
            analysis: Institutional catalyst impact analysis.

        Returns:
            bool: True if alert dispatched successfully, False otherwise.
        """
        try:
            message = format_telegram_message(article, analysis)
            return self.send_message(message)
        except Exception as exc:
            logger.error("Failed to construct or dispatch Telegram alert: %s", exc, exc_info=True)
            return False

    def send_message(self, text: str) -> bool:
        """Send a message string to Telegram with retry backoff for rate limits.

        Args:
            text: Formatted message payload.

        Returns:
            bool: True if Telegram API returns ok=True, False otherwise.
        """
        if not self.is_configured:
            logger.warning(
                "TelegramNotifier: bot_token or chat_id is missing or unconfigured. "
                "Skipping notification delivery."
            )
            return False

        endpoint = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": False,
        }

        retries = 0
        while retries <= self.max_retries:
            try:
                response = requests.post(endpoint, json=payload, timeout=self.timeout)

                # Handle HTTP 429 Too Many Requests
                if response.status_code == 429:
                    retries += 1
                    if retries > self.max_retries:
                        logger.error(
                            "TelegramNotifier: HTTP 429 rate limit exceeded max retries (%d). Dispatch aborted.",
                            self.max_retries,
                        )
                        return False

                    retry_after = 1.0
                    if "Retry-After" in response.headers:
                        try:
                            retry_after = float(response.headers["Retry-After"])
                        except (ValueError, TypeError):
                            retry_after = 1.0
                    else:
                        try:
                            res_json = response.json()
                            if isinstance(res_json, dict) and "parameters" in res_json:
                                retry_after = float(res_json["parameters"].get("retry_after", 1.0))
                        except Exception:
                            retry_after = 1.0

                    logger.warning(
                        "TelegramNotifier: Rate limited (HTTP 429). Retrying in %.2f seconds (attempt %d/%d)...",
                        retry_after,
                        retries,
                        self.max_retries,
                    )
                    time.sleep(retry_after)
                    continue

                # Handle HTTP 200
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                        if isinstance(res_json, dict) and res_json.get("ok") is True:
                            return True
                        logger.error(
                            "Telegram API returned 200 OK but ok=False: %s",
                            res_json,
                        )
                        return False
                    except Exception as json_err:
                        logger.error("Failed to decode Telegram JSON response: %s", json_err)
                        return False

                # Non-200 and Non-429 error codes
                logger.error(
                    "Telegram API dispatch failed with HTTP %d: %s",
                    response.status_code,
                    response.text,
                )
                return False

            except requests.exceptions.RequestException as req_err:
                logger.error("Telegram API network request error: %s", req_err)
                return False
            except Exception as exc:
                logger.error("Unexpected error dispatching Telegram message: %s", exc, exc_info=True)
                return False

        return False
