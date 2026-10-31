import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import requests

logger = logging.getLogger(__name__)


@dataclass
class MarketContext:
    """Quantitative technical and orderflow context for a specific ticker."""
    ticker: str
    close_price: Optional[float] = None
    change_1d_pct: Optional[float] = None
    relative_volume: Optional[float] = None
    rsi_14: Optional[float] = None
    dist_sma50_pct: Optional[float] = None
    dist_sma200_pct: Optional[float] = None
    short_float_pct: Optional[float] = None


@dataclass
class MarketDataConfig:
    """Configuration for live market data grounding."""
    enabled: bool = True
    provider: str = "tradingview"
    fetch_timeout_sec: float = 3.0
    cache_ttl_sec: int = 300  # 5 minutes cache


class TradingViewMarketDataEngine:
    """Fetches real-time technicals and quantitative statistics directly from TradingView screener
    endpoints or custom fetcher adapters to ground catalyst analysis in market reality.
    """

    def __init__(
        self,
        raw_fetcher: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None,
        config: Optional[MarketDataConfig] = None,
    ) -> None:
        self.raw_fetcher = raw_fetcher or self._default_tv_fetcher
        self.config = config or MarketDataConfig()
        self._cache: Dict[str, tuple[float, Optional[MarketContext]]] = {}

    def _default_tv_fetcher(self, ticker: str) -> Optional[Dict[str, Any]]:
        """Default public TradingView scanner POST request for US equities."""
        url = "https://scanner.tradingview.com/america/scan"
        payload = {
            "symbols": {"tickers": [f"NASDAQ:{ticker.upper()}", f"NYSE:{ticker.upper()}"], "query": {"types": []}},
            "columns": [
                "close",
                "change",
                "volume",
                "average_volume_10d_calc",
                "RSI",
                "SMA50",
                "SMA200",
                "short_shares_percent_of_float",
            ],
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.fetch_timeout_sec)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("data", [])
                if results and "d" in results[0]:
                    cols = results[0]["d"]
                    return {
                        "close": cols[0],
                        "change": cols[1],
                        "volume": cols[2],
                        "average_volume_10d_calc": cols[3],
                        "RSI": cols[4],
                        "SMA50": cols[5],
                        "SMA200": cols[6],
                        "short_shares_percent_of_float": cols[7],
                    }
        except Exception as e:
            logger.debug("TradingView fetch failed for %s: %s", ticker, e)
        return None

    def fetch_ticker_context(self, ticker: str) -> Optional[MarketContext]:
        """Retrieves and calculates technical indicators for a given ticker symbol."""
        if not self.config.enabled:
            return None

        symbol = ticker.strip().upper()
        now = time.time()
        if symbol in self._cache:
            ts, cached_ctx = self._cache[symbol]
            if now - ts < self.config.cache_ttl_sec:
                return cached_ctx

        try:
            raw = self.raw_fetcher(symbol)
            if not raw:
                self._cache[symbol] = (now, None)
                return None

            close = raw.get("close")
            change = raw.get("change")
            vol = raw.get("volume")
            avg_vol = raw.get("average_volume_10d_calc")
            rsi = raw.get("RSI")
            sma50 = raw.get("SMA50")
            sma200 = raw.get("SMA200")
            short_float = raw.get("short_shares_percent_of_float")

            # Calculate RVOL
            rvol = None
            if vol is not None and avg_vol is not None and avg_vol > 0:
                rvol = round(vol / avg_vol, 2)

            # Calculate distances to MAs
            dist_50 = None
            if close is not None and sma50 is not None and sma50 > 0:
                dist_50 = round(((close - sma50) / sma50) * 100, 2)

            dist_200 = None
            if close is not None and sma200 is not None and sma200 > 0:
                dist_200 = round(((close - sma200) / sma200) * 100, 2)

            ctx = MarketContext(
                ticker=symbol,
                close_price=float(close) if close is not None else None,
                change_1d_pct=float(change) if change is not None else None,
                relative_volume=rvol,
                rsi_14=float(rsi) if rsi is not None else None,
                dist_sma50_pct=dist_50,
                dist_sma200_pct=dist_200,
                short_float_pct=float(short_float) if short_float is not None else None,
            )
            self._cache[symbol] = (now, ctx)
            return ctx
        except Exception as e:
            logger.warning("Error computing market context for %s: %s", symbol, e)
            return None

    def format_prompt_context(self, tickers: List[str]) -> str:
        """Formats quantitative market context for one or more tickers as a clean prompt block."""
        if not self.config.enabled or not tickers:
            return ""

        blocks: List[str] = []
        for t in tickers:
            ctx = self.fetch_ticker_context(t)
            if not ctx:
                continue

            parts = [f"- {ctx.ticker}:"]
            if ctx.close_price is not None:
                chg = f" ({ctx.change_1d_pct:+.1f}%)" if ctx.change_1d_pct is not None else ""
                parts.append(f"${ctx.close_price:.2f}{chg}")
            if ctx.relative_volume is not None:
                parts.append(f"| RVOL: {ctx.relative_volume:.1f}x")
            if ctx.rsi_14 is not None:
                parts.append(f"| RSI(14): {ctx.rsi_14:.1f}")
            if ctx.dist_sma50_pct is not None:
                parts.append(f"| vs 50DMA: {ctx.dist_sma50_pct:+.1f}%")
            if ctx.short_float_pct is not None:
                parts.append(f"| Short Float: {ctx.short_float_pct:.1f}%")

            blocks.append(" ".join(parts))

        if not blocks:
            return ""

        header = "[QUANTITATIVE MARKET CONTEXT]"
        return f"{header}\n" + "\n".join(blocks)
