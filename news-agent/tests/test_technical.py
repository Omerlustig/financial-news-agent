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


def test_calculate_technical_confluence_bullish():
    engine = TradingViewEngine()

    # 1. Full Bullish Confluence: Strong Buy, RSI 58.0, Above SMA50 & SMA200, Golden Cross
    score, summary, flags = engine.calculate_technical_confluence(
        stance="BULLISH",
        rec="STRONG_BUY",
        rsi=58.0,
        close=150.0,
        sma50=140.0,
        sma200=120.0,
    )
    assert score == 2.0
    assert "Bullish confluence" in summary
    assert flags["is_overbought"] is False
    assert flags["is_oversold"] is False
    assert flags["golden_cross"] is True
    assert flags["death_cross"] is False
    assert flags["above_sma50"] is True
    assert flags["above_sma200"] is True

    # 2. Moderate Bullish Confluence: Buy, Above SMA50 but not above SMA200
    score_mod, summary_mod, flags_mod = engine.calculate_technical_confluence(
        stance="BULLISH",
        rec="BUY",
        rsi=62.0,
        close=135.0,
        sma50=130.0,
        sma200=140.0,
    )
    assert score_mod == 1.5
    assert "Bullish confluence" in summary_mod
    assert flags_mod["above_sma50"] is True
    assert flags_mod["above_sma200"] is False

    # 3. Severe Overbought Risk: RSI >= 75
    score_ob, summary_ob, flags_ob = engine.calculate_technical_confluence(
        stance="BULLISH",
        rec="STRONG_BUY",
        rsi=78.5,
        close=160.0,
        sma50=140.0,
        sma200=120.0,
    )
    assert score_ob == -1.0
    assert "Overbought warning" in summary_ob
    assert flags_ob["is_overbought"] is True


def test_calculate_technical_confluence_bearish():
    engine = TradingViewEngine()

    # 1. Full Bearish Confluence: Strong Sell, RSI 40.0, Below SMA50 & SMA200, Death Cross
    score, summary, flags = engine.calculate_technical_confluence(
        stance="BEARISH",
        rec="STRONG_SELL",
        rsi=40.0,
        close=95.0,
        sma50=110.0,
        sma200=130.0,
    )
    assert score == 2.0
    assert "Bearish confluence" in summary
    assert flags["is_oversold"] is False
    assert flags["death_cross"] is True
    assert flags["golden_cross"] is False
    assert flags["above_sma50"] is False
    assert flags["above_sma200"] is False

    # 2. Moderate Bearish Confluence: Sell, Below SMA50
    score_mod, summary_mod, flags_mod = engine.calculate_technical_confluence(
        stance="BEARISH",
        rec="SELL",
        rsi=38.0,
        close=105.0,
        sma50=110.0,
        sma200=100.0,
    )
    assert score_mod == 1.5
    assert "Bearish confluence" in summary_mod

    # 3. Extreme Oversold Bounce Risk: RSI <= 25
    score_os, summary_os, flags_os = engine.calculate_technical_confluence(
        stance="BEARISH",
        rec="STRONG_SELL",
        rsi=21.0,
        close=85.0,
        sma50=110.0,
        sma200=130.0,
    )
    assert score_os == -1.0
    assert "Oversold bounce risk" in summary_os
    assert flags_os["is_oversold"] is True


def test_calculate_technical_confluence_divergences():
    engine = TradingViewEngine()

    # Bullish news vs Strong Sell technicals
    score_div_bull, sum_div_bull, _ = engine.calculate_technical_confluence(
        stance="BULLISH",
        rec="STRONG_SELL",
        rsi=45.0,
        close=100.0,
        sma50=110.0,
        sma200=120.0,
    )
    assert score_div_bull == -1.5
    assert "Divergence" in sum_div_bull

    # Bearish news vs Strong Buy technicals
    score_div_bear, sum_div_bear, _ = engine.calculate_technical_confluence(
        stance="BEARISH",
        rec="STRONG_BUY",
        rsi=65.0,
        close=150.0,
        sma50=140.0,
        sma200=130.0,
    )
    assert score_div_bear == -1.5
    assert "Divergence" in sum_div_bear


def test_calculate_technical_confluence_neutral():
    engine = TradingViewEngine()

    score, summary, flags = engine.calculate_technical_confluence(
        stance="NEUTRAL",
        rec="NEUTRAL",
        rsi=50.0,
        close=100.0,
        sma50=100.0,
        sma200=100.0,
    )
    assert score == 0.0
    assert "Neutral technical setup" in summary
    assert flags["golden_cross"] is False
    assert flags["death_cross"] is False

