"""
Unit tests for TradingView Technical Analysis Engine and confluence checking.
"""
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from src.analysis.technical import TradingViewEngine
from src.models import (
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    TechnicalIndicatorSummary,
)
from src.notification.telegram import format_telegram_message


def test_clean_symbol():
    engine = TradingViewEngine()
    assert engine.clean_symbol("$NVDA") == "NVDA"
    assert engine.clean_symbol("aapl") == "AAPL"
    assert engine.clean_symbol(" $tsla ") == "TSLA"
    assert engine.clean_symbol("") == ""


def test_compute_confluence_note_bullish():
    engine = TradingViewEngine()
    # Bullish news + Buy technicals -> Bullish confluence
    note = engine.compute_confluence_note("BULLISH", "STRONG_BUY", rsi=55.0, close=100.0, sma50=90.0)
    assert "Bullish confluence" in note

    # Bullish news + Overbought RSI -> Overbought warning
    note_overbought = engine.compute_confluence_note("BULLISH", "BUY", rsi=80.0, close=100.0, sma50=90.0)
    assert "Overbought warning" in note_overbought

    # Bullish news + Sell technicals -> Divergence
    note_div = engine.compute_confluence_note("BULLISH", "SELL", rsi=40.0, close=100.0, sma50=110.0)
    assert "Divergence" in note_div


def test_compute_confluence_note_bearish():
    engine = TradingViewEngine()
    # Bearish news + Sell technicals -> Bearish confluence
    note = engine.compute_confluence_note("BEARISH", "SELL", rsi=45.0, close=100.0, sma50=110.0)
    assert "Bearish confluence" in note

    # Bearish news + Oversold RSI -> Oversold bounce risk
    note_oversold = engine.compute_confluence_note("BEARISH", "STRONG_SELL", rsi=25.0, close=100.0, sma50=110.0)
    assert "Oversold bounce risk" in note_oversold

    # Bearish news + Buy technicals -> Divergence
    note_div = engine.compute_confluence_note("BEARISH", "BUY", rsi=60.0, close=100.0, sma50=90.0)
    assert "Divergence" in note_div


def test_get_technical_summary_mocked():
    engine = TradingViewEngine()

    mock_analysis = MagicMock()
    mock_analysis.summary = {"RECOMMENDATION": "BUY", "BUY": 16, "SELL": 2, "NEUTRAL": 8}
    mock_analysis.indicators = {"RSI": 58.4, "close": 138.20, "SMA50": 130.00, "SMA200": 115.00}

    with patch("src.analysis.technical.TA_Handler") as mock_handler_cls:
        mock_handler_instance = MagicMock()
        mock_handler_instance.get_analysis.return_value = mock_analysis
        mock_handler_cls.return_value = mock_handler_instance

        summary = engine.get_technical_summary("$NVDA", directional_stance="BULLISH")

        assert summary is not None
        assert summary.symbol == "$NVDA"
        assert summary.recommendation == "BUY"
        assert summary.rsi == 58.4
        assert summary.close_price == 138.20
        assert summary.sma50 == 130.00
        assert summary.sma200 == 115.00
        assert "Bullish confluence" in summary.confluence_note


def test_get_technical_summary_cache():
    engine = TradingViewEngine(cache_ttl_seconds=60)
    engine._cache["AAPL"] = (
        datetime.now().timestamp(),
        TechnicalIndicatorSummary(
            symbol="$AAPL",
            recommendation="BUY",
            rsi=62.0,
            close_price=225.0,
            sma50=220.0,
            sma200=200.0,
        ),
    )
    cached = engine.get_technical_summary("$AAPL", directional_stance="BULLISH")
    assert cached is not None
    assert cached.symbol == "$AAPL"
    assert cached.rsi == 62.0


def test_format_telegram_message_with_technicals():
    article = RawArticle(
        title="Nvidia Announces Record AI Cluster Deployments",
        url="https://example.com/nvda",
        source_name="Reuters",
    )
    tech = TechnicalIndicatorSummary(
        symbol="$NVDA",
        recommendation="STRONG_BUY",
        buy_votes=18,
        sell_votes=1,
        neutral_votes=7,
        rsi=64.2,
        close_price=142.50,
        sma50=132.00,
        sma200=118.00,
        confluence_note="Bullish confluence: Technical momentum confirms catalyst.",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        summary="Nvidia expands data center capacity.",
        catalyst_thesis="Record enterprise AI cluster orders expand operating margins.",
        technicals=[tech],
    )
    msg = format_telegram_message(article, analysis)

    assert "TRADINGVIEW TECHNICAL CONFLUENCE" in msg
    assert "$NVDA" in msg
    assert "STRONG_BUY" in msg
    assert "RSI: 64.2" in msg
    assert "$142.50" in msg
    assert "Above 50-SMA $132.00" in msg
    assert "Bullish confluence: Technical momentum confirms catalyst." in msg
