"""
TradingView Technical Analysis Engine for Market News Agent.
Enriches fundamental catalyst alerts with live technical confluence metrics
(Consensus Recommendation, RSI, 50-day / 200-day SMAs, and Confluence Verdict).
"""
from __future__ import annotations

import logging
import time
from typing import Dict, Optional, Tuple

from src.models import DirectionalStance, TechnicalIndicatorSummary

logger = logging.getLogger("technical_engine")

try:
    from tradingview_ta import TA_Handler, Interval, Exchange
    _TV_AVAILABLE = True
except ImportError:
    _TV_AVAILABLE = False
    TA_Handler = None
    Interval = None
    Exchange = None


class TradingViewEngine:
    """Queries TradingView scanner for real-time technical indicators and calculates confluence."""

    def __init__(self, cache_ttl_seconds: int = 900) -> None:
        self.cache_ttl = cache_ttl_seconds
        # In-memory cache: symbol -> (timestamp, TechnicalIndicatorSummary)
        self._cache: Dict[str, Tuple[float, TechnicalIndicatorSummary]] = {}

    def clean_symbol(self, raw_symbol: str) -> str:
        """Strip dollar signs, dots, and whitespace to normalize symbol."""
        cleaned = raw_symbol.strip().upper()
        if cleaned.startswith("$"):
            cleaned = cleaned[1:]
        # Normalize BRK.B to BRK_B or BRK.B
        return cleaned

    def compute_confluence_note(
        self,
        stance: Optional[str],
        rec: str,
        rsi: Optional[float],
        close: Optional[float],
        sma50: Optional[float],
    ) -> str:
        """Synthesize fundamental news stance with TradingView technical setup."""
        stance_str = (stance or "NEUTRAL").upper()
        rec_upper = (rec or "NEUTRAL").upper()

        if "BULL" in stance_str:
            if rsi is not None and rsi >= 75:
                return "⚠️ Overbought warning: RSI > 75, wait for technical pullback."
            elif any(w in rec_upper for w in ["BUY"]):
                return "🟢 Bullish confluence: Technical momentum confirms catalyst."
            elif any(w in rec_upper for w in ["SELL"]):
                return "⚡ Divergence: Bullish news catalyst facing technical resistance / downtrend."
            return "📈 Constructive: Catalyst aligns with stable technical backdrop."

        elif "BEAR" in stance_str:
            if rsi is not None and rsi <= 30:
                return "⚠️ Oversold bounce risk: RSI < 30, technical snapback possible."
            elif any(w in rec_upper for w in ["SELL"]):
                return "🔴 Bearish confluence: Technical breakdown confirms catalyst."
            elif any(w in rec_upper for w in ["BUY"]):
                return "⚡ Divergence: Bearish catalyst against prevailing technical uptrend."
            return "📉 Headwind: Technical backdrop reflects softening demand."

        # Neutral or Volatile
        if rec_upper in ("STRONG_BUY", "BUY"):
            return f"🟢 Technicals favor upside: {rec_upper}"
        elif rec_upper in ("STRONG_SELL", "SELL"):
            return f"🔴 Technicals favor downside: {rec_upper}"
        return f"⚖️ Neutral technical setup: {rec_upper}"

    def get_technical_summary(
        self,
        symbol: str,
        directional_stance: Optional[str] = None,
    ) -> Optional[TechnicalIndicatorSummary]:
        """Query TradingView for a given symbol and calculate technical confluence."""
        if not _TV_AVAILABLE or not symbol:
            return None

        clean_sym = self.clean_symbol(symbol)
        if not clean_sym:
            return None

        # Check in-memory cache
        now = time.time()
        if clean_sym in self._cache:
            cached_time, cached_summary = self._cache[clean_sym]
            if now - cached_time < self.cache_ttl:
                # Update confluence note for the current specific stance
                cached_summary.confluence_note = self.compute_confluence_note(
                    directional_stance,
                    cached_summary.recommendation,
                    cached_summary.rsi,
                    cached_summary.close_price,
                    cached_summary.sma50,
                )
                return cached_summary

        # Try common US exchanges: NASDAQ first, then NYSE, then AMEX
        candidate_exchanges = ["NASDAQ", "NYSE", "AMEX"]
        analysis = None

        for ex in candidate_exchanges:
            try:
                handler = TA_Handler(
                    symbol=clean_sym,
                    screener="america",
                    exchange=ex,
                    interval=Interval.INTERVAL_1_DAY,
                )
                analysis = handler.get_analysis()
                if analysis and analysis.summary:
                    break
            except Exception:
                continue

        # If not found in US, try Israel TASE screener if symbol matches Israeli stock pattern
        if analysis is None or not getattr(analysis, "summary", None):
            try:
                handler = TA_Handler(
                    symbol=clean_sym,
                    screener="israel",
                    exchange="TASE",
                    interval=Interval.INTERVAL_1_DAY,
                )
                analysis = handler.get_analysis()
            except Exception:
                pass

        if analysis is None or not getattr(analysis, "summary", None):
            logger.debug("TradingView technical analysis unavailable for symbol: %s", clean_sym)
            return None

        try:
            summary_dict = analysis.summary or {}
            indicators = analysis.indicators or {}

            rec = str(summary_dict.get("RECOMMENDATION", "NEUTRAL")).upper()
            buy_votes = int(summary_dict.get("BUY", 0))
            sell_votes = int(summary_dict.get("SELL", 0))
            neutral_votes = int(summary_dict.get("NEUTRAL", 0))

            rsi_val = indicators.get("RSI")
            rsi = round(float(rsi_val), 1) if rsi_val is not None else None

            close_val = indicators.get("close")
            close_price = round(float(close_val), 2) if close_val is not None else None

            sma50_val = indicators.get("SMA50")
            sma50 = round(float(sma50_val), 2) if sma50_val is not None else None

            sma200_val = indicators.get("SMA200")
            sma200 = round(float(sma200_val), 2) if sma200_val is not None else None

            confluence = self.compute_confluence_note(
                directional_stance, rec, rsi, close_price, sma50
            )

            tech_summary = TechnicalIndicatorSummary(
                symbol=f"${clean_sym}",
                recommendation=rec,
                buy_votes=buy_votes,
                sell_votes=sell_votes,
                neutral_votes=neutral_votes,
                rsi=rsi,
                close_price=close_price,
                sma50=sma50,
                sma200=sma200,
                confluence_note=confluence,
            )

            self._cache[clean_sym] = (now, tech_summary)
            return tech_summary

        except Exception as exc:
            logger.warning("Error parsing TradingView indicators for %s: %s", clean_sym, exc)
            return None
