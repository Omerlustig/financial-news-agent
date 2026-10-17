from datetime import datetime, timedelta, timezone
from unittest.mock import patch, MagicMock
import pytest
from typer.testing import CliRunner

from src.analysis.digest import MarketDigestGenerator
from src.cli import app
from src.models import (
    DirectionalStance,
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
    EventStatus,
    CausalChain,
)
from src.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_db(tmp_path):
    db_file = tmp_path / "test_digest.db"
    storage = SQLiteStorage(str(db_file))
    yield storage, str(db_file)
    storage.close()


def test_digest_empty_storage(temp_db):
    """Verify digest handles empty storage gracefully with zero counts and disclaimer."""
    storage, _ = temp_db
    generator = MarketDigestGenerator()
    digest = generator.generate_digest(storage, hours=24)

    assert "📊 <b>DAILY MARKET CATALYST DIGEST</b> (Past 24h)" in digest
    assert "Total Catalysts:</b> 0" in digest
    assert "Bullish:</b> 0" in digest
    assert "Bearish:</b> 0" in digest
    assert "No qualifying market catalysts detected" in digest
    assert "News-derived hypothesis" in digest


def test_digest_populated_storage_and_sector_grouping(temp_db):
    """Verify digest correctly queries, groups by sector, counts stances, and includes mechanisms."""
    storage, _ = temp_db
    now = datetime.now(timezone.utc)

    # 1. Bullish Semiconductor catalyst (alerted=True)
    art1 = RawArticle(
        title="Hyperscalers Double AI Chip Orders",
        url="https://reuters.com/ai-chips",
        source_name="Reuters",
        published_at=now - timedelta(hours=2),
    )
    analysis1 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA", "$AMD"],
        affected_sectors=["Semiconductors"],
        directional_stance=DirectionalStance.BULLISH,
        summary="Hyperscalers aggressively expanded data center allocations.",
        catalyst_thesis="Capex surge directly inflates high-margin backlog recognition.",
        causal_chain=CausalChain(
            event_facts="Hyperscalers doubled procurement.",
            economic_mechanism="Capex surge directly inflates high-margin backlog recognition.",
            affected_sectors=["Semiconductors"],
            beneficiaries=["$NVDA", "$AMD"],
        ),
        status=EventStatus.OPPORTUNITY_CANDIDATE,
    )
    storage.record_article(art1, analysis1, alerted=True)

    # 2. Bearish Aerospace catalyst (status=OPPORTUNITY_CANDIDATE, alerted=False)
    art2 = RawArticle(
        title="Aerospace Supply Chain Halts Deliveries",
        url="https://bloomberg.com/aerospace-halt",
        source_name="Bloomberg",
        published_at=now - timedelta(hours=5),
    )
    analysis2 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$BA"],
        affected_sectors=["Aerospace & Defense"],
        directional_stance=DirectionalStance.BEARISH,
        summary="Titanium component shortage disrupts jet engine assembly.",
        catalyst_thesis="Delivery delays impair operating margins and delay billings.",
        causal_chain=CausalChain(
            event_facts="Titanium suppliers announced force majeure.",
            economic_mechanism="Delivery delays impair operating margins and delay billings.",
            affected_sectors=["Aerospace & Defense"],
            negatively_exposed=["$BA"],
        ),
        status=EventStatus.OPPORTUNITY_CANDIDATE,
    )
    storage.record_article(art2, analysis2, alerted=False)

    # 3. Routine Watchlist Article (alerted=False, status=WATCHLIST) -> Should be EXCLUDED
    art3 = RawArticle(
        title="Routine Corporate Governance Update",
        url="https://example.com/corp",
        source_name="PR Newswire",
        published_at=now - timedelta(hours=1),
    )
    analysis3 = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=["$MSFT"],
        affected_sectors=["Software"],
        directional_stance=DirectionalStance.NEUTRAL,
        summary="Board committee assignment announced.",
        catalyst_thesis="Non-material routine disclosure.",
        status=EventStatus.WATCHLIST,
    )
    storage.record_article(art3, analysis3, alerted=False)

    generator = MarketDigestGenerator()
    digest = generator.generate_digest(storage, hours=24)

    assert "📊 <b>DAILY MARKET CATALYST DIGEST</b> (Past 24h)" in digest
    assert "Total Catalysts:</b> 2" in digest
    assert "Bullish:</b> 1" in digest
    assert "Bearish:</b> 1" in digest

    # Sector headings and items
    assert "Semiconductors" in digest
    assert "$NVDA" in digest
    assert "Score: 9/10" in digest
    assert "Hyperscalers Double AI Chip Orders" in digest
    assert "Capex surge directly inflates high-margin backlog recognition" in digest

    assert "Aerospace &amp; Defense" in digest
    assert "$BA" in digest
    assert "Score: 8/10" in digest
    assert "Aerospace Supply Chain Halts Deliveries" in digest
    assert "Delivery delays impair operating margins" in digest


    # Watchlist article must NOT be present
    assert "Routine Corporate Governance Update" not in digest


def test_digest_time_window_filtering(temp_db):
    """Verify articles outside the requested hours window are excluded."""
    storage, _ = temp_db
    now = datetime.now(timezone.utc)

    # Article published 36 hours ago (outside 24h window)
    art_old = RawArticle(
        title="Ancient Rate Decision",
        url="https://old.com/rate",
        source_name="OldWire",
        published_at=now - timedelta(hours=36),
    )
    analysis_old = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$SPY"],
        affected_sectors=["Financials"],
        summary="Old rate decision.",
        catalyst_thesis="Historical interest rate impact.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
    )
    storage.record_article(art_old, analysis_old, alerted=True)

    # Manually backdate created_at in database to 36 hours ago
    old_iso = (now - timedelta(hours=36)).isoformat()
    with storage._conn:
        storage._conn.execute(
            "UPDATE articles SET created_at = ? WHERE content_hash = ?",
            (old_iso, art_old.content_hash),
        )

    generator = MarketDigestGenerator()
    digest = generator.generate_digest(storage, hours=24)

    assert "Total Catalysts:</b> 0" in digest
    assert "Ancient Rate Decision" not in digest


def test_cli_digest_command_output(temp_db):
    """Verify Typer CLI 'digest' command executes successfully and prints digest."""
    _, db_path = temp_db
    runner = CliRunner()
    result = runner.invoke(app, ["digest", "--hours", "12", "--db", db_path])

    assert result.exit_code == 0
    assert "DAILY MARKET CATALYST DIGEST" in result.stdout
    assert "Past 12h" in result.stdout


def test_cli_digest_command_with_send_telegram(temp_db):
    """Verify Typer CLI 'digest' command dispatches to Telegram when --send-telegram is flagged."""
    _, db_path = temp_db
    runner = CliRunner()

    with patch("src.cli.TelegramNotifier") as mock_notifier_cls:
        mock_instance = MagicMock()
        mock_instance.is_configured = True
        mock_instance.send_message.return_value = True
        mock_notifier_cls.return_value = mock_instance

        result = runner.invoke(app, ["digest", "--send-telegram", "--db", db_path])

        assert result.exit_code == 0
        mock_instance.send_message.assert_called_once()
        assert "dispatched to Telegram successfully" in result.stdout
