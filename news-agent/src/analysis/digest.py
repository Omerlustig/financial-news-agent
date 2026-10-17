import html
import json
import logging
from typing import Any, Dict, List, Optional

from src.storage.base import BaseStorage

logger = logging.getLogger(__name__)


def _escape_html(text: Optional[str]) -> str:
    """Safely escape text for Telegram and HTML rendering."""
    if text is None:
        return ""
    return html.escape(str(text), quote=False)


def _parse_json_list(val: Any) -> List[str]:
    """Parse JSON string or list into a list of strings."""
    if not val:
        return []
    if isinstance(val, list):
        return [str(item).strip() for item in val if str(item).strip()]
    if isinstance(val, str):
        try:
            parsed = json.loads(val)
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
            if isinstance(parsed, str):
                return [parsed.strip()]
        except Exception:
            return [s.strip() for s in val.split(",") if s.strip()]
    return [str(val).strip()]


class MarketDigestGenerator:
    """Generates an executive briefing catalyst digest across sectors over a configurable time window."""

    def __init__(self, default_disclaimer: Optional[str] = None) -> None:
        self.disclaimer = (
            default_disclaimer
            or "⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked."
        )

    def generate_digest(self, storage: BaseStorage, hours: int = 24) -> str:
        """Query storage for high-impact catalysts within `hours` and produce an executive digest.

        Executive briefing components:
        - Header: 📊 <b>DAILY MARKET CATALYST DIGEST</b> (Past {hours}h)
        - Overview: Total catalysts, top bullish vs bearish count
        - Sector grouping: Grouped by affected_sectors with bullet points:
          (tickers, score, headline, mechanism)
        - Footer: Mandatory hypothesis disclaimer
        """
        if hasattr(storage, "get_recent_catalysts"):
            catalysts = storage.get_recent_catalysts(hours=hours)
        else:
            catalysts = []

        total_count = len(catalysts)
        bullish_count = 0
        bearish_count = 0

        for c in catalysts:
            stance = str(c.get("directional_stance") or "").upper()
            if "BULL" in stance:
                bullish_count += 1
            elif "BEAR" in stance:
                bearish_count += 1

        lines: List[str] = [
            f"📊 <b>DAILY MARKET CATALYST DIGEST</b> (Past {hours}h)",
            "",
            f"📈 <b>Total Catalysts:</b> {total_count} | 🟢 <b>Bullish:</b> {bullish_count} | 🔴 <b>Bearish:</b> {bearish_count}",
            "",
        ]

        if total_count == 0:
            lines.append("<i>No qualifying market catalysts detected in the evaluation window.</i>")
            lines.append("")
            lines.append(f"<i>{_escape_html(self.disclaimer)}</i>")
            return "\n".join(lines)

        # Group by sector
        by_sector: Dict[str, List[Dict[str, Any]]] = {}
        for c in catalysts:
            sectors = _parse_json_list(c.get("sectors"))
            if not sectors:
                sectors = ["Macro / Cross-Sector"]

            primary_sector = sectors[0]
            by_sector.setdefault(primary_sector, []).append(c)

        for sector, items in sorted(by_sector.items()):
            lines.append(f"📁 <b>{_escape_html(sector)}</b>")
            for item in items:
                tickers = _parse_json_list(item.get("tickers"))
                if not tickers:
                    tickers = _parse_json_list(item.get("beneficiaries"))

                tickers_str = ", ".join(tickers) if tickers else "Macro"
                score = item.get("confidence_score") if item.get("confidence_score") is not None else 0
                title = item.get("title") or "Untitled Catalyst"
                mechanism = (
                    item.get("causal_mechanism")
                    or item.get("catalyst_thesis")
                    or "Mechanism unrecorded."
                )

                lines.append(
                    f"• <b>{_escape_html(tickers_str)}</b> (Score: {score}/10) - {_escape_html(title)}"
                )
                lines.append(
                    f"  ↳ <i>Mechanism:</i> {_escape_html(mechanism)}"
                )
            lines.append("")

        lines.append(f"<i>{_escape_html(self.disclaimer)}</i>")
        return "\n".join(lines)
