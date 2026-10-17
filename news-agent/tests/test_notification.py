from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest
import requests

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.notification.base import BaseNotifier
from src.notification.telegram import (
    TelegramNotifier,
    format_telegram_message,
    escape_html,
)


def test_base_notifier_cannot_be_instantiated():
    """Verify BaseNotifier is an abstract class requiring implementation."""
    with pytest.raises(TypeError):
        BaseNotifier()  # type: ignore[abstract]


def test_telegram_message_formatting():
    """Verify message formatting contains badge, score, tickers, sectors, summary, thesis, url."""
    article = RawArticle(
        title="Fed cuts rates by 50bps",
        url="https://cnbc.com/rates",
        source_name="CNBC",
        published_at=datetime(2026, 10, 17, 14, 30, tzinfo=timezone.utc),
        summary="Fed surprise move",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$SPY", "$QQQ"],
        affected_sectors=["Financials", "Real Estate"],
        summary="The Federal Reserve implemented an emergency 50bps rate cut.",
        catalyst_thesis="Lower borrowing costs will immediately re-rate equity multiples.",
    )
    msg = format_telegram_message(article, analysis)

    assert "[CRITICAL | Score: 9/10]" in msg
    assert "🚨" in msg
    assert "$SPY" in msg
    assert "$QQQ" in msg
    assert "Financials" in msg
    assert "Real Estate" in msg
    assert "Fed cuts rates by 50bps" in msg
    assert "CNBC" in msg
    assert "2026-10-17 14:30" in msg
    assert "The Federal Reserve implemented an emergency 50bps rate cut." in msg
    assert "Lower borrowing costs will immediately re-rate equity multiples." in msg
    assert "https://cnbc.com/rates" in msg


def test_telegram_message_formatting_all_grades():
    """Verify all impact grades produce distinct badges."""
    article = RawArticle(
        title="Sample News",
        url="https://example.com/news",
        source_name="Reuters",
    )
    grades_and_emojis = [
        (ImpactGrade.LOW, 2, "ℹ️", "LOW"),
        (ImpactGrade.AVERAGE, 5, "📊", "AVERAGE"),
        (ImpactGrade.HIGH, 8, "⚡", "HIGH"),
        (ImpactGrade.CRITICAL, 10, "🚨", "CRITICAL"),
    ]
    for grade, score, emoji, label in grades_and_emojis:
        analysis = MarketImpactAnalysis(
            is_market_moving=True,
            confidence_score=score,
            impact_grade=grade,
            affected_tickers=["$AAPL"],
            affected_sectors=["Technology"],
            summary="Test summary.",
            catalyst_thesis="Test thesis.",
        )
        msg = format_telegram_message(article, analysis)
        expected_badge = f"[{label} | Score: {score}/10]"
        assert expected_badge in msg
        assert emoji in msg


def test_telegram_message_formatting_with_directional_stance():
    """Verify directional stance is included when provided."""
    article = RawArticle(
        title="Bullish Momentum",
        url="https://example.com/bull",
        source_name="Bloomberg",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Nvidia beats estimates by 40%.",
        catalyst_thesis="Massive data center demand acceleration.",
    )
    # Passed via directional_stance parameter
    msg = format_telegram_message(article, analysis, directional_stance="Bullish")
    assert "Bullish" in msg
    assert "Stance" in msg


def test_telegram_message_formatting_empty_fields_and_none_published_at():
    """Verify graceful handling when tickers, sectors are empty and published_at is None."""
    article = RawArticle(
        title="Macro Indicator Update",
        url="https://example.com/macro",
        source_name="WSJ",
        published_at=None,
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=[],
        summary="Routine labor index published.",
        catalyst_thesis="Inline with expectations, minimal delta.",
    )
    msg = format_telegram_message(article, analysis)
    assert "N/A" in msg  # published_at or empty field fallback
    assert "https://example.com/macro" in msg


def test_telegram_message_formatting_escapes_special_characters():
    """Verify HTML special characters (<, >, &) are escaped properly."""
    article = RawArticle(
        title="AT&T <T> & S&P > 5000: 'Breakout' & More",
        url="https://example.com/at&t?a=1&b=2",
        source_name="Finance & Co",
        summary="Operating income > $5B & subscribers < 100M.",
        catalyst_thesis="EPS beat & strong churn < expected.",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=7,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$T"],
        affected_sectors=["Telecom & Utilities"],
        summary="Operating income > $5B & subscribers < 100M.",
        catalyst_thesis="EPS beat & strong churn < expected.",
    )
    msg = format_telegram_message(article, analysis)

    # Entities should be escaped
    assert "&lt;T&gt;" in msg
    assert "&gt; 5000" in msg
    assert "&amp;" in msg
    # Raw unescaped dangerous tags shouldn't appear
    assert "<T>" not in msg
    assert "> 5000" not in msg


def test_escape_html_helper():
    """Verify HTML escaping utility helper."""
    assert escape_html(None) == ""
    assert escape_html("") == ""
    assert escape_html("Tom & Jerry <123> 'test' \"quoted\"") == "Tom &amp; Jerry &lt;123&gt; 'test' \"quoted\""


def test_telegram_send_alert_success():
    """Verify TelegramNotifier sends alert message successfully."""
    notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345")
    article = RawArticle(
        title="Rate cut announcement",
        url="https://example.com/cut",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$SPY"],
        affected_sectors=["Financials"],
        summary="Emergency rate cut.",
        catalyst_thesis="Immediate liquidity injection.",
    )

    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True, "result": {"message_id": 999}}

        success = notifier.send_alert(article, analysis)
        assert success is True

        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args[1]
        assert call_kwargs["timeout"] == 10
        assert "api.telegram.org/botfake_token/sendMessage" in mock_post.call_args[0][0]
        json_payload = call_kwargs["json"]
        assert json_payload["chat_id"] == "12345"
        assert json_payload["parse_mode"] == "HTML"
        assert "[CRITICAL | Score: 9/10]" in json_payload["text"]


def test_telegram_send_message_unconfigured():
    """Verify unconfigured notifier logs a warning and returns False safely without HTTP calls."""
    unconfigured_cases = [
        TelegramNotifier(bot_token=None, chat_id=None),
        TelegramNotifier(bot_token="", chat_id="12345"),
        TelegramNotifier(bot_token="fake_token", chat_id=""),
        TelegramNotifier(bot_token="   ", chat_id="12345"),
        TelegramNotifier(bot_token="fake_token", chat_id="   "),
    ]
    with patch("requests.post") as mock_post:
        for notifier in unconfigured_cases:
            res = notifier.send_message("Test message")
            assert res is False
            res_alert = notifier.send_alert(
                RawArticle(title="T", url="http://a", source_name="S"),
                MarketImpactAnalysis(
                    is_market_moving=True,
                    confidence_score=5,
                    impact_grade=ImpactGrade.AVERAGE,
                    summary="S",
                    catalyst_thesis="T",
                ),
            )
            assert res_alert is False
        mock_post.assert_not_called()


def test_telegram_send_message_api_returns_ok_false():
    """Verify handling when Telegram API returns 200 but ok is False."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": False, "description": "Chat not found"}

        success = notifier.send_message("Hello")
        assert success is False


def test_telegram_send_message_http_429_retry_with_header():
    """Verify HTTP 429 rate limit triggers Retry-After backoff and succeeds on retry."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123", max_retries=2)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "2"}
    resp_429.json.return_value = {"ok": False, "error_code": 429, "parameters": {"retry_after": 2}}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"ok": True}

    with patch("requests.post", side_effect=[resp_429, resp_200]) as mock_post, \
         patch("time.sleep") as mock_sleep:
        success = notifier.send_message("Rate limited message")
        assert success is True
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(2.0)


def test_telegram_send_message_http_429_retry_with_json_fallback():
    """Verify HTTP 429 without header falls back to JSON parameters retry_after."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123", max_retries=2)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {}
    resp_429.json.return_value = {"ok": False, "error_code": 429, "parameters": {"retry_after": 3}}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"ok": True}

    with patch("requests.post", side_effect=[resp_429, resp_200]) as mock_post, \
         patch("time.sleep") as mock_sleep:
        success = notifier.send_message("Rate limited message")
        assert success is True
        assert mock_post.call_count == 2
        mock_sleep.assert_called_once_with(3.0)


def test_telegram_send_message_http_429_exceeds_max_retries():
    """Verify HTTP 429 returns False after exhausting max retries."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123", max_retries=2)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "1"}
    resp_429.json.return_value = {"ok": False, "error_code": 429}

    with patch("requests.post", return_value=resp_429) as mock_post, \
         patch("time.sleep"):
        success = notifier.send_message("Exceeds retries")
        assert success is False
        assert mock_post.call_count == 3  # initial + 2 retries


def test_telegram_send_message_http_error_response():
    """Verify non-200, non-429 HTTP errors return False."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123")

    resp_400 = MagicMock()
    resp_400.status_code = 400
    resp_400.text = "Bad Request: can't parse entities"

    with patch("requests.post", return_value=resp_400):
        success = notifier.send_message("Bad payload")
        assert success is False


def test_telegram_send_message_network_exception():
    """Verify requests network exception is logged and returns False without crashing."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123")

    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Network drop")):
        success = notifier.send_message("Connection failure")
        assert success is False
