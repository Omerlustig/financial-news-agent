import html
import logging
import re
import time
from typing import Any, Dict, List, Optional

import requests

from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    SentimentLabel,
    SentimentScore,
    SurgeLevel,
    TelegramButton,
    TrendMetrics,
    TrendVerdict,
)
from src.notification.base import BaseNotifier


logger = logging.getLogger(__name__)

MAX_TELEGRAM_MESSAGE_LENGTH = 4000
MAX_RETRY_SLEEP_SECONDS = 60.0

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


def safe_truncate_html(text: str, max_length: int = MAX_TELEGRAM_MESSAGE_LENGTH) -> str:
    """Safely truncate HTML formatted text to max_length without breaking tags or entities."""
    if len(text) <= max_length:
        return text

    ellipsis = "..."
    available_len = max_length - len(ellipsis)
    truncated = text[:available_len]

    # Avoid cutting inside an HTML entity (e.g. &amp;)
    last_amp = truncated.rfind("&")
    last_semi = truncated.rfind(";")
    if last_amp != -1 and last_amp > last_semi:
        truncated = truncated[:last_amp]

    # Avoid cutting inside an HTML tag (<b... or </b...)
    last_lt = truncated.rfind("<")
    last_gt = truncated.rfind(">")
    if last_lt != -1 and last_lt > last_gt:
        truncated = truncated[:last_lt]

    tag_pattern = re.compile(r"<(/?)([a-zA-Z0-9]+)(?:\s+[^>]*)?>")

    def get_closing_tags(s: str) -> str:
        open_tags: List[str] = []
        for match in tag_pattern.finditer(s):
            is_closing = match.group(1) == "/"
            tag_name = match.group(2).lower()
            if not is_closing:
                open_tags.append(tag_name)
            else:
                if open_tags and open_tags[-1] == tag_name:
                    open_tags.pop()
        return "".join(f"</{tag}>" for tag in reversed(open_tags))

    closing = get_closing_tags(truncated)
    while len(truncated) + len(ellipsis) + len(closing) > max_length and len(truncated) > 0:
        truncated = truncated[:-1]
        last_lt = truncated.rfind("<")
        last_gt = truncated.rfind(">")
        if last_lt != -1 and last_lt > last_gt:
            truncated = truncated[:last_lt]
        last_amp = truncated.rfind("&")
        last_semi = truncated.rfind(";")
        if last_amp != -1 and last_amp > last_semi:
            truncated = truncated[:last_amp]
        closing = get_closing_tags(truncated)

    return truncated + ellipsis + closing


def format_sentiment_gauge(score: float, width: int = 10) -> str:
    """Generate a visual unicode block gauge representing sentiment score.

    Rules:
    - If score > 0: `width` proportional green blocks 🟩 followed by white blocks ⬜
      e.g. for score 0.70 with width 10 -> 7 green, 3 white: `[🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜] (+0.70)`
    - If score < 0: `width` proportional red blocks 🟥 followed by white blocks ⬜
      e.g. for score -0.50 with width 10 -> 5 red, 5 white: `[🟥🟥🟥🟥🟥⬜⬜⬜⬜⬜] (-0.50)`
    - If score == 0: all white blocks `[⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜] (0.00)`
    """
    if score is None:
        score = 0.0

    try:
        val = float(score)
    except (ValueError, TypeError):
        val = 0.0

    clamped = max(-1.0, min(1.0, val))

    if abs(clamped) < 0.005:
        return f"[{'⬜' * width}] (0.00)"

    if clamped > 0:
        filled = min(width, max(0, round(clamped * width)))
        empty = width - filled
        return f"[{'🟩' * filled}{'⬜' * empty}] (+{clamped:.2f})"
    else:
        filled = min(width, max(0, round(abs(clamped) * width)))
        empty = width - filled
        return f"[{'🟥' * filled}{'⬜' * empty}] ({clamped:.2f})"


TICKER_COMPANY_MAP: Dict[str, str] = {
    "$FSLR": "First Solar",
    "$RUN": "Sunrun",
    "$ENPH": "Enphase Energy",
    "$SEDG": "SolarEdge",
    "$VRNS": "Varonis Systems",
    "$PANW": "Palo Alto Networks",
    "$CRWD": "CrowdStrike",
    "$CHKP": "Check Point",
    "$FTNT": "Fortinet",
    "$CIBR": "Cybersecurity ETF",
    "$NVDA": "Nvidia",
    "$MSFT": "Microsoft",
    "$ADBE": "Adobe",
    "$AAPL": "Apple",
    "$GOOGL": "Alphabet",
    "$AMZN": "Amazon",
    "$TSLA": "Tesla",
    "$INTC": "Intel",
    "$AMD": "AMD",
    "$TSM": "TSMC",
    "$AVGO": "Broadcom",
    "$QCOM": "Qualcomm",
    "$DAL": "Delta Air Lines",
    "$UAL": "United Airlines",
    "$AAL": "American Airlines",
    "$LUV": "Southwest Airlines",
    "$JETS": "Airline ETF",
    "$BA": "Boeing",
    "$LMT": "Lockheed Martin",
    "$RTX": "RTX Corp",
    "$ESLT": "Elbit Systems",
    "$TEVA": "Teva Pharmaceutical",
    "$NICE": "NICE Systems",
    "$WIX": "Wix.com",
    "$MNDY": "Monday.com",
    "$ACN": "Accenture",
    "$INFY": "Infosys",
    "$XLK": "Tech ETF",
    "$SMH": "Semiconductor ETF",
    "$XLE": "Energy ETF",
    "$SPY": "S&P 500 ETF",
    "$QQQ": "Nasdaq 100 ETF",
}


def format_ticker_with_company(ticker: str) -> str:
    """Format ticker symbol with full company name if available (e.g. $FSLR (First Solar))."""
    cleaned = ticker.strip().upper()
    company = TICKER_COMPANY_MAP.get(cleaned)
    if company:
        return f"{cleaned} ({company})"
    return cleaned


def get_directional_simple_explanation(stance_val: Optional[str], sectors: List[str]) -> str:
    """Generate a plain-English directional impact summary."""
    s = (stance_val or "").upper()
    sec_name = sectors[0] if sectors and sectors != ["None"] else "the sector"
    if "BULL" in s:
        return f"Positive for {sec_name} and domestic leaders"
    elif "BEAR" in s:
        return f"Headwinds and margin pressure for {sec_name}"
    elif "VOL" in s:
        return f"Positive for domestic producers, negative for import-dependent users"
    return "Neutral / selective positioning"


def get_simple_reason_for_beneficiary(ticker: str, analysis: MarketImpactAnalysis) -> str:
    """Generate a clear 'Because' explanation for why a stock/sector is favored."""
    cleaned = ticker.strip().upper()
    if cleaned == "$FSLR":
        return "Because it already manufactures domestically in the US and gains competitive advantage from trade restrictions."
    if cleaned in ("$VRNS",):
        return "Because the buyout offer provides immediate equity re-rating toward the acquisition price."
    if cleaned in ("$PANW", "$CRWD", "$CHKP", "$FTNT", "$CIBR"):
        return "Because expanding sector M&A valuation multiples lift competitor valuations across cybersecurity peers."
    if cleaned in ("$NVDA", "$TSM", "$AVGO"):
        return "Because strong enterprise AI demand and backlog acceleration directly expand operating margins."
    if cleaned in ("$ACN", "$INFY"):
        return "Because US visa restrictions accelerate enterprise offshoring and IT consultancy contract awards."
    if cleaned in ("$DAL", "$UAL", "$AAL"):
        return "Because capacity discipline and premium route demand support higher passenger yields."
    if analysis.causal_chain and analysis.causal_chain.economic_mechanism:
        mech = analysis.causal_chain.economic_mechanism.strip()
        if len(mech) > 130:
            mech = mech[:127] + "..."
        return f"Because {mech[0].lower() + mech[1:]}"
    return "Because it stands to capture market share and benefit from the positive catalyst."


def get_simple_reason_for_exposed(ticker: str, analysis: MarketImpactAnalysis) -> str:
    """Generate a clear 'Because' explanation for why a stock/sector should be avoided or hedged."""
    cleaned = ticker.strip().upper()
    if cleaned in ("$RUN", "$ENPH", "$SEDG"):
        return "Because they install residential panels and may face higher equipment costs and supply shortages."
    if cleaned in ("$INTC",):
        return "Because of foundry packaging bottlenecks, delayed roadmap execution, and margin pressure."
    if cleaned in ("$MSFT", "$ADBE"):
        return "Because tightening immigration/visa programs elevates engineering wage inflation and project overhead."
    if analysis.causal_chain and analysis.causal_chain.counterarguments:
        counter = analysis.causal_chain.counterarguments.strip()
        if len(counter) > 130:
            counter = counter[:127] + "..."
        return f"Because of {counter[0].lower() + counter[1:]}"
    return "Because it faces margin pressure, rising supply costs, or regulatory headwinds."


def format_telegram_message(
    article: RawArticle,
    analysis: MarketImpactAnalysis,
    directional_stance: Optional[str] = None,
) -> str:

    """Format an institutional market impact analysis into a Telegram HTML alert.

    Follows the institutional causal alert specification:
    - Header: 🚨 [OPPORTUNITY CANDIDATE | Score: {score}/10 | {grade}]
    - Stance & Time Horizon: 📈 Stance: {stance} and ⏱ Horizon: {time_horizon} (if available)
    - Sectors: 🏭 Sectors: {sectors}
    - Headline, Publisher, and UTC Timestamp
    - Executive Summary: 2-3 sentence neutral facts brief
    - Causal Mechanism: Detailed transmission channel from causal_chain.economic_mechanism
    - Beneficiaries vs Negatively Exposed:
      * 🟢 Beneficiaries: {beneficiaries}
      * 🔴 Negatively Exposed: {negatively_exposed}
    - Counterarguments & Invalidation:
      * ⚖️ Counterarguments: {counterarguments}
      * 🛑 Invalidation Condition: {invalidation_conditions}
    - Direct URL Link to original source
    - Mandatory compliance hypothesis disclaimer

    Maintains robust HTML escaping and 4000-character safe truncation.
    """
    grade_name = (
        getattr(analysis.impact_grade, "value", str(analysis.impact_grade)).upper()
        if analysis.impact_grade
        else "UNKNOWN"
    )
    score = analysis.confidence_score if analysis.confidence_score is not None else 0

    lines: List[str] = []
    if getattr(analysis, "surge_level", None) == SurgeLevel.SURGE_BREAKING:
        lines.append("<b>⚡ [BREAKING CATALYST SURGE]</b>")

    if getattr(analysis, "is_watchlist_match", False):
        if getattr(analysis, "matched_watchlist_items", None):
            items_str = ", ".join(analysis.matched_watchlist_items)
            lines.append(f"⭐ <b>[WATCHLIST FOCUS: {escape_html(items_str)}]</b>")
        else:
            lines.append("⭐ <b>[WATCHLIST FOCUS]</b>")

    lines.append(
        f"🚨 <b>[OPPORTUNITY CANDIDATE | Score: {score}/10 | {escape_html(grade_name)}]</b>"
    )

    # Stance & Time Horizon
    stance = directional_stance or getattr(analysis, "directional_stance", None) or getattr(analysis, "stance", None)
    stance_val = getattr(stance, "value", str(stance)).strip() if stance else None

    time_horizon = getattr(analysis, "time_horizon", None)
    horizon_val = getattr(time_horizon, "value", str(time_horizon)).strip() if time_horizon else None

    meta_parts: List[str] = []
    if stance_val:
        lower_stance = stance_val.lower()
        if any(w in lower_stance for w in ["bear", "down", "neg", "short"]):
            stance_icon = "📉"
        elif any(w in lower_stance for w in ["volatile"]):
            stance_icon = "⚡"
        elif any(w in lower_stance for w in ["neutral"]):
            stance_icon = "⚖️"
        else:
            stance_icon = "📈"
        meta_parts.append(f"{stance_icon} <b>Stance:</b> {escape_html(stance_val)}")

    if horizon_val:
        meta_parts.append(f"⏱ <b>Horizon:</b> {escape_html(horizon_val)}")

    if meta_parts:
        lines.append(" | ".join(meta_parts))

    # Sectors
    sectors: List[str] = []
    if analysis.causal_chain and analysis.causal_chain.affected_sectors:
        sectors = analysis.causal_chain.affected_sectors
    elif analysis.affected_sectors:
        sectors = analysis.affected_sectors
    sectors_str = ", ".join(sectors) if sectors else "None"
    lines.append(f"🏭 <b>Sectors:</b> {escape_html(sectors_str)}")

    # SEC / TASE Regulatory Disclosure
    if getattr(analysis, "sec_filing_item", None):
        filing_val = getattr(analysis.sec_filing_item, "value", str(analysis.sec_filing_item))
        lines.append(f"🏛️ <b>Filing:</b> {escape_html(filing_val)}")


    # Headline, Publisher, and UTC Timestamp
    pub_time = article.published_at.strftime("%Y-%m-%d %H:%M UTC") if article.published_at else "N/A"
    lines.append(f"<b>Headline:</b> {escape_html(article.title)}")
    lines.append(f"<b>Publisher:</b> {escape_html(article.source_name)} | <b>Published:</b> {pub_time}")

    # Multi-Source Corroboration badge if corroborated across independent publishers
    corroborating = getattr(analysis, "corroborating_sources", None) or []
    if len(corroborating) >= 2:
        sources_str = ", ".join(corroborating)
        lines.append(f"📡 <b>Corroborated by:</b> {escape_html(sources_str)} ({len(corroborating)} independent sources)")

    # Sentiment & Trend metrics
    sentiment_score = 0.0
    sentiment_label_val = "NEUTRAL"
    if getattr(analysis, "sentiment", None) is not None and analysis.sentiment is not None:
        if analysis.sentiment.score is not None:
            sentiment_score = float(analysis.sentiment.score)
        if analysis.sentiment.label is not None:
            sentiment_label_val = getattr(
                analysis.sentiment.label, "value", str(analysis.sentiment.label)
            ).upper().replace(" ", "_")

    # Visual Sentiment & Trend metrics (Compact)
    emoji_badges = {
        "VERY_BULLISH": "🚀 VERY BULLISH",
        "BULLISH": "🟢 BULLISH",
        "NEUTRAL": "⚪ NEUTRAL",
        "BEARISH": "🔴 BEARISH",
        "VERY_BEARISH": "🚨 VERY BEARISH",
    }
    emoji_badge = emoji_badges.get(sentiment_label_val, f"⚪ {escape_html(sentiment_label_val)}")
    gauge = format_sentiment_gauge(sentiment_score, width=10)
    lines.append(f"📊 <b>Sentiment:</b> {emoji_badge} | {gauge}")

    if getattr(analysis, "trend", None) is not None and analysis.trend is not None:
        lines.append(
            f"🔥 <b>News Buzz:</b> {analysis.trend.buzz_ratio:.1f}x (24h Articles: {analysis.trend.articles_24h_count})"
        )
        verdict_val = (
            analysis.trend.trend_verdict.value
            if hasattr(analysis.trend.trend_verdict, "value")
            else str(analysis.trend.trend_verdict)
        )
        lines.append(
            f"📈 <b>Trend Verdict:</b> {escape_html(verdict_val)} (Velocity: {analysis.trend.sentiment_velocity:+.2f})"
        )
        if getattr(analysis.trend, "sentiment_acceleration", 0.0) != 0.0:
            lines.append(
                f"⚡ <b>Acceleration:</b> {analysis.trend.sentiment_acceleration:+.2f} ({escape_html(verdict_val)})"
            )

    # Beneficiaries vs Negatively Exposed
    if analysis.causal_chain:
        beneficiaries = analysis.causal_chain.beneficiaries
        negatively_exposed = analysis.causal_chain.negatively_exposed
    else:
        if stance_val and any(w in stance_val.lower() for w in ["bear", "down", "neg", "short"]):
            beneficiaries = []
            negatively_exposed = analysis.affected_tickers
        else:
            beneficiaries = analysis.affected_tickers
            negatively_exposed = []

    # Clean empty lists
    beneficiaries = [b for b in beneficiaries if b and b != "None"]
    negatively_exposed = [e for e in negatively_exposed if e and e != "None"]

    # Recommendation / Actionable Takeaway
    rec = getattr(analysis, "recommendation", None)
    if not rec:
        if beneficiaries:
            rec = f"Monitor {', '.join(beneficiaries)} for catalyst follow-through over {horizon_val or 'short term'}; watch counterarguments."
        elif sectors and sectors != ["None"] and "BULL" in str(stance_val):
            rec = f"Evaluate overweight positioning on {sectors_str} sector over {horizon_val or 'short term'}."
        elif sectors and sectors != ["None"] and "BEAR" in str(stance_val):
            rec = f"Exercise caution or evaluate hedge protection on {sectors_str} sector."
        elif negatively_exposed:
            rec = f"Exercise caution or evaluate hedge protection on {', '.join(negatively_exposed)}."
        else:
            rec = "Track subsequent corporate disclosures and primary filings for tradeable confirmation."

    # ==============================================================
    # 🎯 1. BOTTOM LINE (Top priority: recommendation + positioning)
    # ==============================================================
    lines.append("")
    lines.append("🎯 <b>BOTTOM LINE:</b>")
    lines.append(f"💡 <b>Recommendation:</b> {escape_html(rec)}")

    action_summary_items: List[str] = []
    if beneficiaries:
        action_summary_items.append(f"🟢 <b>Favor:</b> {', '.join(beneficiaries)}")
    if negatively_exposed:
        action_summary_items.append(f"🔴 <b>Avoid/Hedge:</b> {', '.join(negatively_exposed)}")
    if not action_summary_items and sectors and sectors != ["None"]:
        action_summary_items.append(f"🏭 <b>In Play:</b> {sectors_str}")

    if action_summary_items:
        lines.append(f"👉 <b>Actionable Positioning:</b> {' | '.join(action_summary_items)}")

    # Concise single-line rationale for top tickers
    if beneficiaries:
        top_b = beneficiaries[0]
        b_with_co = format_ticker_with_company(top_b)
        b_reason = get_simple_reason_for_beneficiary(top_b, analysis)
        lines.append(f"   • 🟢 <b>{escape_html(b_with_co)}:</b> {escape_html(b_reason)}")

    if negatively_exposed:
        top_e = negatively_exposed[0]
        e_with_co = format_ticker_with_company(top_e)
        e_reason = get_simple_reason_for_exposed(top_e, analysis)
        lines.append(f"   • 🔴 <b>{escape_html(e_with_co)}:</b> {escape_html(e_reason)}")

    # ==============================================================
    # 📖 2. THE CATALYST (Short, focused explanation of what happened & why)
    # ==============================================================
    summary = analysis.summary
    if not summary and analysis.causal_chain and analysis.causal_chain.event_facts:
        summary = analysis.causal_chain.event_facts
    summary_text = summary if summary else "None"

    if analysis.causal_chain and analysis.causal_chain.economic_mechanism:
        mechanism = analysis.causal_chain.economic_mechanism
    elif analysis.catalyst_thesis:
        mechanism = analysis.catalyst_thesis
    else:
        mechanism = "None"

    lines.append("")
    lines.append("📖 <b>THE CATALYST:</b>")
    lines.append(f"• <b>Event:</b> {escape_html(summary_text)}")
    if mechanism and mechanism != "None" and mechanism != summary_text:
        lines.append(f"• <b>Why it matters:</b> {escape_html(mechanism)}")

    # ==============================================================
    # ⚖️ 3. RISK & INVALIDATION (Counterarguments & thesis invalidation)
    # ==============================================================
    counterarguments = (
        analysis.causal_chain.counterarguments
        if analysis.causal_chain and analysis.causal_chain.counterarguments and analysis.causal_chain.counterarguments != "None"
        else None
    )
    invalidation = (
        analysis.causal_chain.invalidation_conditions
        if analysis.causal_chain and analysis.causal_chain.invalidation_conditions and analysis.causal_chain.invalidation_conditions != "None"
        else None
    )

    if counterarguments or invalidation:
        lines.append("")
        lines.append("⚖️ <b>RISK & INVALIDATION:</b>")
        if counterarguments:
            lines.append(f"• <b>Risk:</b> {escape_html(counterarguments)}")
        if invalidation:
            lines.append(f"• <b>Invalidation:</b> {escape_html(invalidation)}")

    # Direct URL Link
    escaped_url_attr = html.escape(article.url, quote=True)
    escaped_url_text = escape_html(article.url)
    lines.append("")
    lines.append(f'🔗 <a href="{escaped_url_attr}">{escaped_url_text}</a>')

    # Mandatory Disclaimer at footer
    disclaimer_text = (
        getattr(analysis, "disclaimer", None)
        or "⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked."
    )
    lines.append("")
    lines.append(f"<i>{escape_html(disclaimer_text)}</i>")

    formatted = "\n".join(lines)
    return safe_truncate_html(formatted, max_length=MAX_TELEGRAM_MESSAGE_LENGTH)


class TelegramNotifier(BaseNotifier):
    """Dispatches market news alerts to Telegram channels or private chats via Bot API."""

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
        timeout: int = 10,
        max_retries: int = 3,
        parse_mode: str = "HTML",
        fallback_to_console: bool = False,
    ) -> None:
        self.bot_token = bot_token.strip() if bot_token else None
        self.chat_id = chat_id.strip() if chat_id else None
        self.timeout = timeout
        self.max_retries = max_retries
        self.parse_mode = parse_mode
        self.fallback_to_console = fallback_to_console

    @property
    def is_configured(self) -> bool:
        """Check if bot token and target chat ID are provided and non-empty."""
        return bool(self.bot_token and self.chat_id)

    def _redact_token(self, value: Any) -> str:
        """Ensure bot token is redacted from string representations in logs."""
        s = str(value)
        if self.bot_token and self.bot_token in s:
            return s.replace(self.bot_token, "[REDACTED_TOKEN]")
        return s

    def send_alert(self, article: RawArticle, analysis: MarketImpactAnalysis) -> bool:
        """Format and dispatch an alert for a market-moving event with interactive inline buttons.

        Args:
            article: Normalized article metadata.
            analysis: Institutional catalyst impact analysis.

        Returns:
            bool: True if alert dispatched successfully, False otherwise.
        """
        if not self.is_configured:
            if self.fallback_to_console:
                from src.notification.console import ConsoleNotifier
                return ConsoleNotifier().send_alert(article, analysis)
            logger.warning(
                "TelegramNotifier: bot_token or chat_id is missing or unconfigured. "
                "Skipping notification delivery."
            )
            return False

        try:
            message = format_telegram_message(article, analysis)

            button_row: List[Dict[str, str]] = []

            # 1. Read Full Article
            if article.url:
                button_row.append({
                    "text": "🔗 Read Full Article",
                    "url": article.url,
                })

            # 2. Charts for first 1-2 tradeable tickers
            if analysis.affected_tickers:
                for ticker in analysis.affected_tickers[:2]:
                    ticker_clean = ticker.lstrip("$").strip().upper()
                    if ticker_clean:
                        button_row.append({
                            "text": f"📊 Chart {ticker}",
                            "url": f"https://www.tradingview.com/symbols/{ticker_clean}/",
                        })

            # 3. SEC / TASE Regulatory Disclosure
            if analysis.sec_filing_item and article.url:
                button_row.append({
                    "text": "🏛️ SEC / TASE Disclosure",
                    "url": article.url,
                })

            reply_markup: Optional[Dict[str, Any]] = (
                {"inline_keyboard": [button_row]} if button_row else None
            )

            if not analysis.interactive_buttons and button_row:
                analysis.interactive_buttons = [
                    TelegramButton(text=b["text"], url=b["url"]) for b in button_row
                ]

            return self.send_message(message, reply_markup=reply_markup)
        except Exception as exc:
            logger.error("Failed to construct or dispatch Telegram alert: %s", self._redact_token(exc))
            return False

    def send_message(self, text: str, reply_markup: Optional[Dict[str, Any]] = None) -> bool:
        """Send a message string to Telegram with retry backoff for rate limits.

        Args:
            text: Formatted message payload.
            reply_markup: Optional Telegram reply markup dict (e.g. inline_keyboard).

        Returns:
            bool: True if Telegram API returns ok=True, False otherwise.
        """
        if not self.is_configured:
            if self.fallback_to_console:
                from src.notification.console import ConsoleNotifier
                return ConsoleNotifier().send_message(text, reply_markup=reply_markup)
            logger.warning(
                "TelegramNotifier: bot_token or chat_id is missing or unconfigured. "
                "Skipping notification delivery."
            )
            return False


        endpoint = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        payload: Dict[str, Any] = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": self.parse_mode,
            "disable_web_page_preview": False,
        }
        if reply_markup is not None:
            payload["reply_markup"] = reply_markup


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

                    raw_retry_after = 1.0
                    if "Retry-After" in response.headers:
                        try:
                            raw_retry_after = float(response.headers["Retry-After"])
                        except (ValueError, TypeError):
                            raw_retry_after = 1.0
                    else:
                        try:
                            res_json = response.json()
                            if isinstance(res_json, dict) and "parameters" in res_json:
                                raw_retry_after = float(res_json["parameters"].get("retry_after", 1.0))
                        except Exception:
                            raw_retry_after = 1.0

                    # Clamp retry delay to avoid blocking indefinitely
                    sleep_time = min(max(0.0, raw_retry_after), MAX_RETRY_SLEEP_SECONDS)

                    logger.warning(
                        "TelegramNotifier: Rate limited (HTTP 429). Retrying in %.2f seconds (attempt %d/%d)...",
                        sleep_time,
                        retries,
                        self.max_retries,
                    )
                    time.sleep(sleep_time)
                    continue

                # Handle HTTP 200
                if response.status_code == 200:
                    try:
                        res_json = response.json()
                        if isinstance(res_json, dict) and res_json.get("ok") is True:
                            return True
                        logger.error(
                            "Telegram API returned 200 OK but ok=False: %s",
                            self._redact_token(res_json),
                        )
                        return False
                    except Exception as json_err:
                        logger.error("Failed to decode Telegram JSON response: %s", self._redact_token(json_err))
                        return False

                # Non-200 and Non-429 error codes
                logger.error(
                    "Telegram API dispatch failed with HTTP %d: %s",
                    response.status_code,
                    self._redact_token(response.text),
                )
                return False

            except requests.exceptions.RequestException as req_err:
                logger.error("Telegram API network request error: %s", self._redact_token(req_err))
                return False
            except Exception as exc:
                logger.error("Unexpected error dispatching Telegram message: %s", self._redact_token(exc))
                return False

        return False
