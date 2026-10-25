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
    SurgeLevel,
    SECFilingItem,
    SentimentLabel,
    SentimentScore,
    TrendMetrics,
    TrendVerdict,
)

from src.notification.base import BaseNotifier
from src.notification.telegram import (
    TelegramNotifier,
    format_telegram_message,
    format_sentiment_gauge,
    escape_html,
)
from src.notification.console import ConsoleNotifier



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

    # 5. Bottom Line & Actionable Positioning
    assert "🎯 <b>BOTTOM LINE:</b>" in msg
    assert "💡 <b>Recommendation:</b>" in msg
    assert "👉 <b>Actionable Positioning:</b>" in msg
    assert "🟢 <b>Favor:</b> $NVDA, $AVGO" in msg
    assert "🔴 <b>Avoid/Hedge:</b> $INTC" in msg

    # 6. The Catalyst (Event & Why it matters)
    assert "📖 <b>THE CATALYST:</b>" in msg
    assert "• <b>Event:</b> US Commerce Department granted expedited semiconductor export licensing to defense partners." in msg
    assert "• <b>Why it matters:</b> Eliminates 6-9 month export review cycle, accelerating booked backlog recognition and gross margins." in msg

    # 7. Risk & Invalidation
    assert "⚖️ <b>RISK & INVALIDATION:</b>" in msg
    assert "• <b>Risk:</b> Legislative challenge in Congress could delay implementation window by 60 days." in msg
    assert "• <b>Invalidation:</b> Federal register rule revocation or injunction issued by trade court." in msg

    # 8. Direct URL Link
    assert '<a href="https://reuters.com/business/semiconductors-allied-exemption">https://reuters.com/business/semiconductors-allied-exemption</a>' in msg

    # 9. Mandatory Disclaimer at footer
    assert "<i>⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.</i>" in msg


def test_telegram_message_includes_behind_the_lines_subtext():
    """Verify that decoded subtext / reading between the lines is included in Telegram alert."""
    article = RawArticle(
        title="Volkswagen plans global salary freeze through 2030 - report",
        url="https://seekingalpha.com/news/vw-salary-freeze",
        source_name="Seeking Alpha",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$VWAGY", "$TSLA"],
        affected_sectors=["Automotive", "Electric Vehicles"],
        summary="Volkswagen plans unprecedented global salary freeze through 2030 across European operations.",
        catalyst_thesis="Structural European cost crisis and declining China market share force fixed-cost cuts.",
        subtext_analysis="Management is testing union resolve ahead of collective bargaining, pre-empting acute cash-flow bleed and unannounced factory closures.",
        directional_stance=DirectionalStance.BEARISH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
    )
    msg = format_telegram_message(article, analysis)
    assert "Behind the Lines (Subtext):" in msg
    assert "Management is testing union resolve ahead of collective bargaining" in msg


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


def test_telegram_message_formatting_surge_breaking_badge():
    """Verify SURGE_BREAKING level produces breaking badge in header."""
    article = RawArticle(
        title="Breaking Semiconductor Breakthrough",
        url="https://reuters.com/semi-breakthrough",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Major breakthrough.",
        catalyst_thesis="Disruptive advance.",
        surge_level=SurgeLevel.SURGE_BREAKING,
    )
    msg = format_telegram_message(article, analysis)
    assert "⚡ [BREAKING CATALYST SURGE]" in msg


def test_telegram_message_formatting_sec_filing_item():
    """Verify SEC filing classification item is rendered when present."""
    article = RawArticle(
        title="Form 8-K Definitive Agreement Filed",
        url="https://sec.gov/filing/123",
        source_name="SEC EDGAR",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$MSFT"],
        affected_sectors=["Technology"],
        summary="Definitive agreement disclosed.",
        catalyst_thesis="Material acquisition.",
        sec_filing_item=SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT,
    )
    msg = format_telegram_message(article, analysis)
    assert "🏛️ <b>Filing:</b> Item 1.01: Entry into Material Definitive Agreement (M&amp;A / Partnership)" in msg



def test_telegram_send_alert_inline_keyboard_payload():
    """Verify send_alert constructs inline_keyboard with Read Article, Chart buttons, and SEC Disclosure."""
    article = RawArticle(
        title="Major Tech Merger Announced",
        url="https://reuters.com/tech-merger",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA", "$AMD", "$INTC"],
        affected_sectors=["Semiconductors"],
        summary="Merger agreement entered.",
        catalyst_thesis="Synergies.",
        sec_filing_item=SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT,
    )

    notifier = TelegramNotifier(bot_token="test-bot-token", chat_id="123456")

    with patch("src.notification.telegram.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        ok = notifier.send_alert(article, analysis)
        assert ok is True
        assert mock_post.called

        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})

        assert "reply_markup" in payload
        reply_markup = payload["reply_markup"]
        assert "inline_keyboard" in reply_markup

        row = reply_markup["inline_keyboard"][0]
        # Button 1: Read Full Article
        assert row[0]["text"] == "🔗 Read Full Article"
        assert row[0]["url"] == "https://reuters.com/tech-merger"

        # Button 2 & 3: First 1-2 tickers
        assert row[1]["text"] == "📊 Chart $NVDA"
        assert row[1]["url"] == "https://www.tradingview.com/symbols/NVDA/"
        assert row[2]["text"] == "📊 Chart $AMD"
        assert row[2]["url"] == "https://www.tradingview.com/symbols/AMD/"

        # Button 4: SEC Disclosure
        assert row[3]["text"] == "🏛️ SEC / TASE Disclosure"
        assert row[3]["url"] == "https://reuters.com/tech-merger"

        # 3rd ticker ($INTC) should be omitted
        button_texts = [b["text"] for b in row]
        assert not any("INTC" in t for t in button_texts)


def test_telegram_send_message_with_reply_markup():
    """Verify send_message forwards reply_markup payload to Telegram API."""
    notifier = TelegramNotifier(bot_token="test-bot-token", chat_id="123456")

    with patch("src.notification.telegram.requests.post") as mock_post:
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"ok": True}
        mock_post.return_value = mock_response

        markup = {"inline_keyboard": [[{"text": "Custom", "url": "https://custom.com"}]]}
        ok = notifier.send_message("Test with markup", reply_markup=markup)
        assert ok is True

        _, kwargs = mock_post.call_args
        payload = kwargs.get("json", {})
        assert payload.get("reply_markup") == markup


def test_format_sentiment_gauge_values():
    """Verify format_sentiment_gauge handles positive, negative, zero, and boundary clamp values."""
    # Positive
    gauge_pos = format_sentiment_gauge(0.70, width=10)
    assert gauge_pos == "[🟩🟩🟩🟩🟩🟩🟩⬜⬜⬜] (+0.70)"

    # Negative
    gauge_neg = format_sentiment_gauge(-0.50, width=10)
    assert gauge_neg == "[🟥🟥🟥🟥🟥⬜⬜⬜⬜⬜] (-0.50)"

    # Zero
    gauge_zero = format_sentiment_gauge(0.0, width=10)
    assert gauge_zero == "[⬜⬜⬜⬜⬜⬜⬜⬜⬜⬜] (0.00)"

    # Boundary clamps
    gauge_upper = format_sentiment_gauge(1.5, width=10)
    assert gauge_upper == "[🟩🟩🟩🟩🟩🟩🟩🟩🟩🟩] (+1.00)"

    gauge_lower = format_sentiment_gauge(-2.0, width=10)
    assert gauge_lower == "[🟥🟥🟥🟥🟥🟥🟥🟥🟥🟥] (-1.00)"


def test_format_telegram_message_with_sentiment_and_trend():
    """Verify format_telegram_message includes visual sentiment gauge, news buzz, and trend verdict."""
    article = RawArticle(
        title="Nvidia Reports Historic Datacenter Revenue Growth",
        url="https://reuters.com/nvda-earnings",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Nvidia Blackwell revenue surged 200%.",
        catalyst_thesis="Accelerating AI demand expanding margins.",
        sentiment=SentimentScore(score=0.85, label=SentimentLabel.VERY_BULLISH),
        trend=TrendMetrics(
            buzz_ratio=3.2,
            sentiment_velocity=0.45,
            trend_verdict=TrendVerdict.STRONG_BULLISH,
            articles_24h_count=18,
        ),
    )

    msg = format_telegram_message(article, analysis)

    # Sentiment Line
    assert "📊 <b>Sentiment:</b> 🚀 VERY BULLISH | [🟩🟩🟩🟩🟩🟩🟩🟩⬜⬜] (+0.85)" in msg


    # Buzz Line
    assert "🔥 <b>News Buzz:</b> 3.2x (24h Articles: 18)" in msg

    # Trend Verdict Line
    assert "📈 <b>Trend Verdict:</b> STRONG_BULLISH (Velocity: +0.45)" in msg


def test_console_notifier_send_alert_and_message():
    """Verify ConsoleNotifier prints formatted alerts and messages cleanly to console."""
    from rich.console import Console
    import io

    output_stream = io.StringIO()
    test_console = Console(file=output_stream, force_terminal=True, width=120)
    notifier = ConsoleNotifier(console=test_console)

    article = RawArticle(
        title="Broadcom Inks AI Infrastructure Pact",
        url="https://example.com/avgo-deal",
        source_name="Bloomberg",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$AVGO"],
        affected_sectors=["Semiconductors"],
        summary="Broadcom expands enterprise ASIC agreements.",
        catalyst_thesis="High margin custom silicon growth.",
        sentiment=SentimentScore(score=0.60, label=SentimentLabel.BULLISH),
        trend=TrendMetrics(
            buzz_ratio=2.0,
            sentiment_velocity=0.15,
            trend_verdict=TrendVerdict.STRONG_BULLISH,
            articles_24h_count=5,
        ),
    )

    # 1. Test send_alert
    success = notifier.send_alert(article, analysis)
    assert success is True
    alert_output = output_stream.getvalue()
    assert "Broadcom Inks AI Infrastructure Pact" in alert_output
    assert "OPPORTUNITY CANDIDATE" in alert_output
    assert "$AVGO" in alert_output
    assert "ConsoleNotifier (Local Mode)" in alert_output

    # 2. Test send_message
    msg_success = notifier.send_message("<b>Market Update:</b> Pipeline operational.")
    assert msg_success is True
    assert "Market Update: Pipeline operational." in output_stream.getvalue()


def test_telegram_notifier_fallback_to_console():
    """Verify TelegramNotifier delegates to ConsoleNotifier when unconfigured and fallback_to_console=True."""
    notifier = TelegramNotifier(fallback_to_console=True)
    assert not notifier.is_configured

    article = RawArticle(
        title="Local Mode Test Headline",
        url="https://example.com/test",
        source_name="TestFeed",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$AAPL"],
        affected_sectors=["Technology"],
        summary="Local mode test.",
        catalyst_thesis="Local hypothesis.",
    )

    with patch("src.notification.console.ConsoleNotifier.send_alert", return_value=True) as mock_alert:
        ok = notifier.send_alert(article, analysis)
        assert ok is True
        mock_alert.assert_called_once_with(article, analysis)

    with patch("src.notification.console.ConsoleNotifier.send_message", return_value=True) as mock_msg:
        ok = notifier.send_message("Hello from local")
        assert ok is True
        mock_msg.assert_called_once()


def test_format_telegram_message_watchlist_badge():
    """Verify watchlist focus badge is displayed in header when is_watchlist_match is True."""
    article = RawArticle(
        title="Delta Air Lines Cuts Flights Due to Fuel",
        url="https://wsj.com/delta-fuel",
        source_name="Wall Street Journal",
        published_at=datetime(2026, 10, 18, 8, 0, tzinfo=timezone.utc),
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$DAL"],
        affected_sectors=["Aviation"],
        summary="Delta cuts flights schedule due to fuel costs.",
        catalyst_thesis="Fuel spike cuts into quarterly operating margins.",
        is_watchlist_match=True,
        matched_watchlist_items=["$DAL", "Aviation"],
    )
    msg = format_telegram_message(article, analysis)

    assert "⭐ <b>[WATCHLIST FOCUS: $DAL, Aviation]</b>" in msg


def test_format_telegram_message_with_sentiment_acceleration():
    """Verify sentiment acceleration indicator is displayed when trend acceleration is non-zero."""
    article = RawArticle(
        title="Reversal Catalyst in Energy",
        url="https://reuters.com/oil-reversal",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$XOM"],
        affected_sectors=["Energy"],
        summary="Crude inventory drops precipitously.",
        catalyst_thesis="Rapid inventory drain.",
        trend=TrendMetrics(
            buzz_ratio=2.1,
            sentiment_velocity=0.35,
            sentiment_acceleration=0.45,
            trend_verdict=TrendVerdict.BULLISH_REVERSAL,
            articles_24h_count=5,
        ),
    )
    msg = format_telegram_message(article, analysis)

    assert "⚡ <b>Acceleration:</b> +0.45 (BULLISH_REVERSAL)" in msg


def test_format_telegram_message_institutional_two_layer_gate_and_4d_sentiment():
    """Verify telegram formatting renders Event Quality, Opportunity Score, 4D sentiment, and Adversarial Defense."""
    from src.models import (
        AdversarialReview,
        FourDimensionalSentiment,
        TwoLayerGateDecision,
    )

    article = RawArticle(
        title="Nvidia announces breakthrough Blackwell deployment with Hyperscalers",
        url="https://reuters.com/nvda-breakthrough",
        source_name="Reuters",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA"],
        affected_sectors=["Semiconductors"],
        summary="Nvidia deploys first wave of Blackwell GB200 racks to major cloud hyperscalers.",
        catalyst_thesis="Accelerating revenue recognition and datacenter margin expansion.",
        four_d_sentiment=FourDimensionalSentiment(
            linguistic_sentiment=0.8,
            fundamental_impact=0.7,
            expectation_surprise=0.65,
            price_implication=0.69,
            surprise_verdict="BEAT",
        ),
        adversarial_review=AdversarialReview(
            bearish_counter_thesis="Thermal management and liquid cooling yields could constrain volume ramp.",
            priced_in_risk="Hyperscaler capex guidance was already raised last quarter.",
            macro_confounding_risk="Broad semiconductor index correlation.",
            falsification_threshold="Quarterly datacenter revenue growth falls below 15% sequentially.",
            is_thesis_robust=True,
        ),
        two_layer_gate=TwoLayerGateDecision(
            event_quality_score=9.2,
            opportunity_score=8.4,
            passes_event_quality_gate=True,
            passes_opportunity_gate=True,
            gate_verdict="ALERT",
        ),
    )

    msg = format_telegram_message(article, analysis)

    assert "🚨 <b>[OPPORTUNITY CANDIDATE | Event Quality: 9/10 | Opportunity: 8/10]</b>" in msg
    assert "📊 <b>4D Sentiment:</b> Editorial: +0.80 | Cash-Flow: +0.70 | Surprise: +0.65 (BEAT) | Price Imp: +0.69" in msg
    assert "⚔️ <b>ADVERSARIAL CHALLENGER & RISK DEFENSE:</b>" in msg
    assert "• <b>Contrarian Counter-Thesis:</b> Thermal management and liquid cooling yields could constrain volume ramp." in msg
    assert "• <b>Invalidation Threshold:</b> Quarterly datacenter revenue growth falls below 15% sequentially." in msg


def test_format_batch_telegram_message_single():
    from src.notification.telegram import format_batch_telegram_message
    from src.models import CausalChain

    art = RawArticle(
        title="US Conducts Airstrikes in Iran; Oil Shipping Chokepoints at Risk",
        url="https://reuters.com/world/middle-east/iran-strikes",
        source_name="Reuters",
    )
    ana = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$XOM", "$CVX"],
        affected_sectors=["Energy", "Oil & Gas"],
        summary="US strikes military targets in Iran; Strait of Hormuz tanker routes threatened.",
        catalyst_thesis="Crude supply disruption elevates spot oil prices.",
        subtext_analysis="Escalation risk premium pricing in; upstream E&P spot margins expand immediately while passenger airlines face acute jet fuel inflation.",
        directional_stance=DirectionalStance.BULLISH,
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        causal_chain=CausalChain(
            event_facts="US airstrikes conducted.",
            economic_mechanism="Crude price surge expands upstream E&P margins.",
            affected_sectors=["Energy", "Airlines"],
            beneficiaries=["$XOM", "$CVX", "$XLE"],
            negatively_exposed=["$DAL", "$UAL", "$JETS"],
            counterarguments="Rapid diplomatic de-escalation.",
            invalidation_conditions="Crude drops below $75/bbl or shipping guarantees restored.",
        ),
    )

    msg = format_batch_telegram_message([(art, ana)])
    assert "🚨 <b>MARKET OPPORTUNITY (Score: 8/10)</b>" in msg
    assert "Score: 8/10 (HIGH)" in msg
    assert "$XOM, $CVX, $XLE" in msg
    assert "• <b>Catalyst:</b>" in msg
    assert "• <b>Behind The Lines:</b> Escalation risk premium pricing in" in msg
    assert "• <b>Action:</b> 🟢 <b>Buy:</b> $XOM, $CVX, $XLE" in msg
    assert "Avoid/Hedge" not in msg
    assert "• <b>Risk / Invalidation:</b> Crude drops below $75/bbl" in msg
    assert '<a href="https://reuters.com/world/middle-east/iran-strikes">Reuters</a>' in msg

    # Verify include_avoid=True opt-in behavior
    msg_with_avoid = format_batch_telegram_message([(art, ana)], include_avoid=True)
    assert "🔴 <b>Avoid/Hedge:</b> $DAL, $UAL, $JETS" in msg_with_avoid


def test_format_batch_telegram_message_multiple():
    from src.notification.telegram import format_batch_telegram_message
    from src.models import CausalChain

    art1 = RawArticle(
        title="US Conducts Airstrikes in Iran",
        url="https://reuters.com/iran-strikes",
        source_name="Reuters",
    )
    ana1 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$XOM", "$CVX"],
        affected_sectors=["Energy"],
        summary="Airstrikes threaten Persian Gulf oil flows.",
        catalyst_thesis="Crude supply disruption elevates spot oil prices.",
        subtext_analysis="Upstream margins expand immediately while airlines face jet fuel spikes.",
        causal_chain=CausalChain(
            event_facts="Airstrikes conducted.",
            economic_mechanism="Oil spike.",
            affected_sectors=["Energy"],
            beneficiaries=["$XOM", "$CVX"],
            negatively_exposed=["$DAL"],
            invalidation_conditions="Crude below $75/bbl.",
        ),
    )

    art2 = RawArticle(
        title="Volkswagen plans global salary freeze through 2030 - report",
        url="https://seekingalpha.com/vw-freeze",
        source_name="Seeking Alpha",
    )
    ana2 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=7,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$TSLA"],
        affected_sectors=["Automotive"],
        summary="Volkswagen plans unprecedented global wage freeze.",
        catalyst_thesis="Structural cost reduction.",
        subtext_analysis="Management is testing union resolve to avert plant closures.",
        directional_stance=DirectionalStance.BEARISH,
        causal_chain=CausalChain(
            event_facts="Salary freeze through 2030.",
            economic_mechanism="Fixed-cost reduction under Chinese EV pressure.",
            affected_sectors=["Automotive"],
            beneficiaries=["$TSLA"],
            negatively_exposed=["$VWAGY"],
            invalidation_conditions="Margin recovery above 8%.",
        ),
    )

    msg = format_batch_telegram_message([(art1, ana1), (art2, ana2)])
    assert "🚨 <b>MARKET OPPORTUNITIES (2 In Focus)</b>" in msg
    assert "━━━━━━━━━━━━━━━━━━━━━━━━━━" in msg
    assert "1. $XOM, $CVX | 🟢 Energy | Score: 8/10 (HIGH)" in msg
    assert "2. $TSLA | 🔴 Automotive | Score: 7/10 (HIGH)" in msg
    assert "• <b>Behind The Lines:</b>" in msg
    assert "Seeking Alpha" in msg
    assert "Reuters" in msg


def test_telegram_send_batch_alerts_dispatches_single_push():
    from unittest.mock import MagicMock
    from src.notification.telegram import TelegramNotifier

    notifier = TelegramNotifier(bot_token="123:ABC", chat_id="456")
    notifier.send_message = MagicMock(return_value=True)

    art1 = RawArticle(title="Art 1", url="https://a.com", source_name="Source A")
    ana1 = MarketImpactAnalysis(is_market_moving=True, confidence_score=8, impact_grade=ImpactGrade.HIGH, summary="Sum 1", catalyst_thesis="Thesis 1")
    art2 = RawArticle(title="Art 2", url="https://b.com", source_name="Source B")
    ana2 = MarketImpactAnalysis(is_market_moving=True, confidence_score=7, impact_grade=ImpactGrade.HIGH, summary="Sum 2", catalyst_thesis="Thesis 2")

    ok = notifier.send_batch_alerts([(art1, ana1), (art2, ana2)])
    assert ok is True
    # Exactly ONE call to send_message!
    assert notifier.send_message.call_count == 1
    sent_text = notifier.send_message.call_args[0][0]
    assert "MARKET OPPORTUNITIES (2 In Focus)" in sent_text






