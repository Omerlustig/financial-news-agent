from datetime import datetime, timezone
import pytest
from typer.testing import CliRunner

from src.cli import app
from src.storage.sqlite import SQLiteStorage


@pytest.fixture
def temp_scanner_db(tmp_path):
    db_file = tmp_path / "test_scanner.db"
    storage = SQLiteStorage(str(db_file))
    yield storage, str(db_file)
    storage.close()


def test_scanner_cli_empty_storage(temp_scanner_db):
    """Verify scanner cleanly informs user when no ticker sentiments exist."""
    _, db_path = temp_scanner_db
    runner = CliRunner()
    result = runner.invoke(app, ["scanner", "--hours", "24", "--db", db_path])

    assert result.exit_code == 0
    assert "No ticker sentiment records found" in result.stdout


def test_scanner_cli_populated_storage(temp_scanner_db):
    """Verify scanner table displays tickers, buzz multipliers, gauges, velocities, and verdicts."""
    storage, db_path = temp_scanner_db
    now = datetime.now(timezone.utc)

    # Populate multiple tickers with varying sentiment
    storage.record_ticker_sentiment("$NVDA", 0.85, "VERY_BULLISH", "hash_nvda_1", published_at=now)
    storage.record_ticker_sentiment("$NVDA", 0.75, "VERY_BULLISH", "hash_nvda_2", published_at=now)
    storage.record_ticker_sentiment("$TSLA", -0.55, "VERY_BEARISH", "hash_tsla_1", published_at=now)
    storage.record_ticker_sentiment("$AAPL", 0.02, "NEUTRAL", "hash_aapl_1", published_at=now)

    runner = CliRunner()
    result = runner.invoke(app, ["scanner", "--hours", "24", "--db", db_path])

    assert result.exit_code == 0
    assert "Real-Time Market Scanner" in result.stdout
    assert "Ticker" in result.stdout
    assert "Buzz" in result.stdout
    assert "Sentiment" in result.stdout
    assert "Velocity" in result.stdout
    assert "Verdict" in result.stdout


    # Verify rows
    assert "$NVDA" in result.stdout
    assert "$TSLA" in result.stdout
    assert "$AAPL" in result.stdout


def test_scanner_cli_custom_hours_option(temp_scanner_db):
    """Verify scanner accepts custom --hours argument without errors."""
    storage, db_path = temp_scanner_db
    now = datetime.now(timezone.utc)
    storage.record_ticker_sentiment("$AMD", 0.60, "BULLISH", "hash_amd_1", published_at=now)

    runner = CliRunner()
    result = runner.invoke(app, ["scanner", "--hours", "12", "--db", db_path])

    assert result.exit_code == 0
    assert "Real-Time Market Scanner (12h Window)" in result.stdout
    assert "$AMD" in result.stdout
