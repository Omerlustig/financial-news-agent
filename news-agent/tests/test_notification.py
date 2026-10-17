from datetime import datetime, timezone
from unittest.mock import patch, MagicMock
import pytest
import requests

from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    CausalChain,
    DirectionalStance,
    TimeHorizon,
    EventStatus,
    HardGates,
)
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

    assert "[OPPORTUNITY CANDIDATE | Score: 9/10 | CRITICAL]" in msg
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
        (ImpactGrade.LOW, 2, "LOW"),
        (ImpactGrade.AVERAGE, 5, "AVERAGE"),
        (ImpactGrade.HIGH, 8, "HIGH"),
        (ImpactGrade.CRITICAL, 10, "CRITICAL"),
    ]
    for grade, score, label in grades_and_emojis:
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
        expected_badge = f"[OPPORTUNITY CANDIDATE | Score: {score}/10 | {label}]"
        assert expected_badge in msg
        assert "🚨" in msg


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
        assert "[OPPORTUNITY CANDIDATE | Score: 9/10 | CRITICAL]" in json_payload["text"]


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


def test_telegram_message_truncation_on_oversized_input():
    """Verify messages exceeding 4000 characters are safely truncated with ellipsis."""
    massive_summary = "A" * 3000
    massive_thesis = "B" * 2000
    article = RawArticle(
        title="Massive Headline That Is Very Long",
        url="https://example.com/massive",
        source_name="Financial Gazette",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$BIG"],
        affected_sectors=["MegaCap"],
        summary=massive_summary,
        catalyst_thesis=massive_thesis,
    )
    msg = format_telegram_message(article, analysis)
    assert len(msg) <= 4000
    assert "..." in msg
    assert "[OPPORTUNITY CANDIDATE | Score: 8/10 | HIGH]" in msg


def test_telegram_notifier_redacts_bot_token_in_logs(caplog):
    """Verify bot token is redacted as [REDACTED_TOKEN] in error and warning logs."""
    import logging
    secret_token = "SECRET_BOT_TOKEN_98765"
    notifier = TelegramNotifier(bot_token=secret_token, chat_id="12345")

    with caplog.at_level(logging.WARNING):
        # 1. Network exception containing token in url
        err_msg = f"Failed to connect to https://api.telegram.org/bot{secret_token}/sendMessage"
        with patch("requests.post", side_effect=requests.exceptions.ConnectionError(err_msg)):
            res = notifier.send_message("Test")
            assert res is False
            assert secret_token not in caplog.text
            assert "[REDACTED_TOKEN]" in caplog.text

        caplog.clear()

        # 2. HTTP error response containing token
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.text = f"Invalid token: {secret_token}"
        with patch("requests.post", return_value=mock_resp):
            res = notifier.send_message("Test")
            assert res is False
            assert secret_token not in caplog.text
            assert "[REDACTED_TOKEN]" in caplog.text


def test_telegram_send_message_http_429_clamped_sleep():
    """Verify HTTP 429 delays are clamped to MAX_RETRY_SLEEP_SECONDS (60.0s)."""
    notifier = TelegramNotifier(bot_token="token", chat_id="123", max_retries=1)

    resp_429 = MagicMock()
    resp_429.status_code = 429
    resp_429.headers = {"Retry-After": "300"}  # 5 minutes
    resp_429.json.return_value = {"ok": False, "error_code": 429}

    resp_200 = MagicMock()
    resp_200.status_code = 200
    resp_200.json.return_value = {"ok": True}

    with patch("requests.post", side_effect=[resp_429, resp_200]), \
         patch("time.sleep") as mock_sleep:
        success = notifier.send_message("Test clamped sleep")
        assert success is True
        mock_sleep.assert_called_once_with(60.0)


def test_telegram_message_formatting_institutional_causal_chain():
    """Verify Telegram formatting contains causal mechanism, beneficiaries,
    negatively exposed, counterarguments, invalidation conditions, and disclaimer.
    """
    article = RawArticle(
        title="US Grants Advanced Semiconductor Export Exemptions to Allied Defense Consortium",
        url="https://reuters.com/business/semiconductors-allied-exemption",
        source_name="Reuters",
        published_at=datetime(2026, 10, 17, 16, 45, tzinfo=timezone.utc),
        summary="Department of Commerce published new rule granting expedited licensing.",
    )
    causal_chain = CausalChain(
        event_facts="Commerce Department enacted expedited licensing rule for defense AI chip shipments to allied nations.",
        economic_mechanism="Eliminates 6-9 month export review cycle, accelerating booked backlog recognition and gross margins.",
        affected_sectors=["Semiconductors", "Defense Tech"],
        beneficiaries=["$NVDA", "$AVGO"],
        negatively_exposed=["$INTC"],
        counterarguments="Legislative challenge in Congress could delay implementation window by 60 days.",
        invalidation_conditions="Federal register rule revocation or injunction issued by trade court.",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        time_horizon=TimeHorizon.SHORT_TERM,
        affected_tickers=["$NVDA", "$AVGO", "$INTC"],
        affected_sectors=["Semiconductors", "Defense Tech"],
        summary="US Commerce Department granted expedited semiconductor export licensing to defense partners.",
        catalyst_thesis="Backlog clearance will re-rate high-margin enterprise data center earnings.",
        causal_chain=causal_chain,
        hard_gates=HardGates(
            is_novel=True,
            is_verified=True,
            has_direct_exposure=True,
            passes_all_gates=True,
        ),
    )

    msg = format_telegram_message(article, analysis)

    # 1. Header
    assert "🚨 <b>[OPPORTUNITY CANDIDATE | Score: 9/10 | CRITICAL]</b>" in msg

    # 2. Stance & Time Horizon
    assert "📈 <b>Stance:</b> BULLISH" in msg
    assert "⏱ <b>Horizon:</b> SHORT_TERM" in msg

    # 3. Sectors
    assert "🏭 <b>Sectors:</b> Semiconductors, Defense Tech" in msg

    # 4. Headline, Publisher, and UTC Timestamp
    assert "<b>Headline:</b> US Grants Advanced Semiconductor Export Exemptions to Allied Defense Consortium" in msg
    assert "<b>Publisher:</b> Reuters | <b>Published:</b> 2026-10-17 16:45 UTC" in msg

    # 5. Executive Summary
    assert "<b>Executive Summary:</b>" in msg
    assert "US Commerce Department granted expedited semiconductor export licensing to defense partners." in msg

    # 6. Causal Mechanism
    assert "<b>Causal Mechanism:</b>" in msg
    assert "Eliminates 6-9 month export review cycle, accelerating booked backlog recognition and gross margins." in msg

    # 7. Beneficiaries vs Negatively Exposed
    assert "🟢 <b>Beneficiaries:</b> $NVDA, $AVGO" in msg
    assert "🔴 <b>Negatively Exposed:</b> $INTC" in msg

    # Recommendation
    assert "💡 <b>Recommendation:</b>" in msg

    # 8. Counterarguments & Invalidation
    assert "⚖️ <b>Counterarguments:</b> Legislative challenge in Congress could delay implementation window by 60 days." in msg
    assert "🛑 <b>Invalidation Condition:</b> Federal register rule revocation or injunction issued by trade court." in msg

    # 9. Direct URL Link
    assert '<a href="https://reuters.com/business/semiconductors-allied-exemption">https://reuters.com/business/semiconductors-allied-exemption</a>' in msg

    # 10. Mandatory Disclaimer at footer
    assert "<i>⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.</i>" in msg


def test_telegram_message_formatting_with_explicit_recommendation():
    article = RawArticle(
        title="Check Point Software Acquires Cyber Leader",
        url="https://globes.co.il/news/chkp-deal",
        source_name="Globes",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$CHKP"],
        affected_sectors=["Cybersecurity"],
        summary="Check Point completed accretive acquisition.",
        catalyst_thesis="Expanding ARR and AI cyber defense margins.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        time_horizon="MEDIUM_TERM",
        recommendation="Accumulate $CHKP on opening dip; set trailing stops at 50-day moving average.",
    )
    msg = format_telegram_message(article, analysis)
    assert "💡 <b>Recommendation:</b> Accumulate $CHKP on opening dip; set trailing stops at 50-day moving average." in msg


def test_telegram_message_formatting_corroboration_badge():
    article = RawArticle(
        title="Regulators Approve Landmark Biotech Merger",
        url="https://marketwatch.com/biotech-deal",
        source_name="MarketWatch",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$BIIB"],
        affected_sectors=["Biotech"],
        summary="Antitrust cleared merger.",
        catalyst_thesis="Merger approved.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        corroborating_sources=["Bloomberg", "MarketWatch", "Reuters"],
        corroboration_count=3,
    )
    msg = format_telegram_message(article, analysis)
    assert "📡 <b>Corroborated by:</b> Bloomberg, MarketWatch, Reuters (3 independent sources)" in msg


