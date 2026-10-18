"""
TradingView Technical Analysis Engine for Market News Agent.
Enriches fundamental catalyst alerts with live technical confluence metrics
(Consensus Recommendation, RSI, 50-day / 200-day SMAs, and Confluence Verdict).
"""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, Optional, Tuple

from src.models import DirectionalStance, TechnicalIndicatorSummary

logger = logging.getLogger("technical_engine")

# Quantitative Indicator Thresholds
RSI_OVERBOUGHT = 70.0
RSI_SEVERE_OVERBOUGHT = 75.0
RSI_OVERSOLD = 30.0
RSI_EXTREME_OVERSOLD = 25.0

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

    RSI_OVERBOUGHT = RSI_OVERBOUGHT
    RSI_SEVERE_OVERBOUGHT = RSI_SEVERE_OVERBOUGHT
    RSI_OVERSOLD = RSI_OVERSOLD
    RSI_EXTREME_OVERSOLD = RSI_EXTREME_OVERSOLD

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

    def calculate_technical_confluence(
        self,
        stance: Optional[str],
        rec: str,
        rsi: Optional[float],
        close: Optional[float],
        sma50: Optional[float],
        sma200: Optional[float] = None,
    ) -> Tuple[float, str, Dict[str, Any]]:
        """Calculate quantitative technical confluence score modifier (-2.0 to +2.0),

        summary verdict string, and structural technical flags.
        """
        stance_str = (stance or "NEUTRAL").upper()
        rec_upper = (rec or "NEUTRAL").upper()

        technical_flags: Dict[str, Any] = {
            "is_overbought": (rsi >= self.RSI_OVERBOUGHT) if rsi is not None else False,
            "is_oversold": (rsi <= self.RSI_OVERSOLD) if rsi is not None else False,
            "golden_cross": (sma50 > sma200) if (sma50 is not None and sma200 is not None) else False,
            "death_cross": (sma50 < sma200) if (sma50 is not None and sma200 is not None) else False,
            "above_sma50": (close > sma50) if (close is not None and sma50 is not None) else False,
            "above_sma200": (close > sma200) if (close is not None and sma200 is not None) else False,
        }

        confluence_score_mod = 0.0
        verdict_summary = ""

        is_bull_stance = "BULL" in stance_str
        is_bear_stance = "BEAR" in stance_str

        if is_bull_stance:
            if rsi is not None and rsi >= self.RSI_SEVERE_OVERBOUGHT:
                confluence_score_mod = -1.0
                verdict_summary = "⚠️ Overbought warning: RSI >= 75 indicates severe overbought risk despite bullish news."
            elif any(w in rec_upper for w in ["SELL"]):
                confluence_score_mod = -1.5
                verdict_summary = "⚡ Divergence: Bullish news catalyst facing technical resistance / downtrend."
            elif any(w in rec_upper for w in ["BUY"]) and (rsi is None or rsi < self.RSI_OVERBOUGHT) and technical_flags["above_sma50"]:
                if "STRONG" in rec_upper or technical_flags["above_sma200"] or technical_flags["golden_cross"]:
                    confluence_score_mod = 2.0
                else:
                    confluence_score_mod = 1.5
                verdict_summary = "🟢 Bullish confluence: Technical momentum confirms catalyst."
            else:
                confluence_score_mod = 0.0
                verdict_summary = "📈 Constructive: Catalyst aligns with stable technical backdrop."

        elif is_bear_stance:
            if rsi is not None and rsi <= self.RSI_EXTREME_OVERSOLD:
                confluence_score_mod = -1.0
                verdict_summary = "⚠️ Oversold bounce risk: RSI <= 25 indicates extreme oversold snapback risk."
            elif any(w in rec_upper for w in ["BUY"]):
                confluence_score_mod = -1.5
                verdict_summary = "⚡ Divergence: Bearish catalyst against prevailing technical uptrend."
            elif any(w in rec_upper for w in ["SELL"]) and (rsi is None or rsi > self.RSI_OVERSOLD) and (close is not None and sma50 is not None and close < sma50):
                below_sma200 = (close is not None and sma200 is not None and close < sma200)
                if "STRONG" in rec_upper or below_sma200 or technical_flags["death_cross"]:
                    confluence_score_mod = 2.0
                else:
                    confluence_score_mod = 1.5
                verdict_summary = "🔴 Bearish confluence: Technical breakdown confirms catalyst."
            else:
                confluence_score_mod = 0.0
                verdict_summary = "📉 Headwind: Technical backdrop reflects softening demand."

        else:
            confluence_score_mod = 0.0
            if "STRONG_BUY" in rec_upper or "BUY" in rec_upper:
                verdict_summary = f"🟢 Technicals favor upside: {rec_upper}"
            elif "STRONG_SELL" in rec_upper or "SELL" in rec_upper:
                verdict_summary = f"🔴 Technicals favor downside: {rec_upper}"
            else:
                verdict_summary = f"⚖️ Neutral technical setup: {rec_upper}"

        confluence_score_mod = max(-2.0, min(2.0, float(confluence_score_mod)))
        return confluence_score_mod, verdict_summary, technical_flags

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
