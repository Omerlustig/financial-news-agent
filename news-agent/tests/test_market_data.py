import pytest
from typing import Dict, Any, Optional
from src.analysis.market_data import TradingViewMarketDataEngine, MarketContext, MarketDataConfig


def mock_adapter(ticker: str) -> Optional[Dict[str, Any]]:
    if ticker.upper() == "NVDA":
        return {
            "close": 142.50,
            "change": 4.82,
            "volume": 75000000,
            "average_volume_10d_calc": 30000000,
            "RSI": 78.4,
            "SMA50": 126.8,
            "SMA200": 114.5,
            "short_shares_percent_of_float": 18.5,
        }
    elif ticker.upper() == "TSLA":
        return {
            "close": 210.00,
            "change": -2.10,
            "volume": 40000000,
            "average_volume_10d_calc": 45000000,
            "RSI": 42.1,
            "SMA50": 215.0,
            "SMA200": 195.0,
            "short_shares_percent_of_float": 3.2,
        }
    return None


def test_fetch_ticker_context_computes_rvol_and_distances() -> None:
    engine = TradingViewMarketDataEngine(raw_fetcher=mock_adapter)
    ctx = engine.fetch_ticker_context("NVDA")
    assert ctx is not None
    assert ctx.ticker == "NVDA"
    assert ctx.close_price == 142.50
    assert ctx.change_1d_pct == 4.82
    # RVOL: 75M / 30M = 2.5
    assert ctx.relative_volume == 2.5
    assert ctx.rsi_14 == 78.4
    # dist to SMA50: (142.5 - 126.8) / 126.8 * 100 ~ 12.38%
    assert ctx.dist_sma50_pct is not None
    assert round(ctx.dist_sma50_pct, 1) == 12.4
    assert ctx.short_float_pct == 18.5


def test_format_prompt_context_generates_grounding_block() -> None:
    engine = TradingViewMarketDataEngine(raw_fetcher=mock_adapter)
    context_str = engine.format_prompt_context(["NVDA", "TSLA"])
    assert "[QUANTITATIVE MARKET CONTEXT]" in context_str
    assert "NVDA: $142.50 (+4.8%)" in context_str
    assert "RVOL: 2.5x" in context_str
    assert "RSI(14): 78.4" in context_str
    assert "TSLA: $210.00 (-2.1%)" in context_str


def test_fetch_ticker_context_handles_unknown_ticker() -> None:
    engine = TradingViewMarketDataEngine(raw_fetcher=mock_adapter)
    ctx = engine.fetch_ticker_context("UNKNOWN_TICKER_XYZ")
    assert ctx is None
    context_str = engine.format_prompt_context(["UNKNOWN_TICKER_XYZ"])
    assert context_str == ""


def test_fetch_ticker_context_handles_fetch_exception() -> None:
    def crashing_fetcher(ticker: str) -> Dict[str, Any]:
        raise ConnectionError("Network timeout connecting to TradingView")

    engine = TradingViewMarketDataEngine(raw_fetcher=crashing_fetcher)
    ctx = engine.fetch_ticker_context("NVDA")
    assert ctx is None
    # Formatting prompt should not crash
    assert engine.format_prompt_context(["NVDA"]) == ""
