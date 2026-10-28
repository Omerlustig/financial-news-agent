import pytest
from datetime import datetime, timezone
from src.models import (
    RecommendationRecord,
    RawArticle,
    MarketImpactAnalysis,
    CausalChain,
    DirectionalStance,
    TechnicalIndicatorSummary,
    EventStatus,
    TwoLayerGateDecision,
)
from src.storage.sqlite import SQLiteStorage
from src.pipeline import Pipeline


def test_recommendation_record_model():
    rec = RecommendationRecord(
        recommended_at=datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc),
        ticker="nvda",
        action="BUY",
        initial_price=135.50,
        confidence_score=9,
        opportunity_score=8.5,
        headline="NVIDIA Announces Breakthrough AI Chip Architecture",
        source="Reuters",
        article_hash="hash_nvda_123",
        event_id="evt_nvda_123",
    )
    assert rec.ticker == "$NVDA"
    assert rec.initial_price == 135.50
    assert rec.confidence_score == 9
    assert rec.opportunity_score == 8.5
    assert rec.status == "OPEN"
    assert rec.latest_price is None
    assert rec.return_pct is None


def test_storage_record_and_get_recommendations(tmp_path):
    db_file = tmp_path / "test_recs.db"
    storage = SQLiteStorage(db_file)

    rec1 = RecommendationRecord(
        recommended_at=datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc),
        ticker="$NVDA",
        action="BUY",
        initial_price=135.50,
        confidence_score=9,
        opportunity_score=8.5,
        headline="NVIDIA Breakthrough",
        source="Reuters",
        article_hash="hash_1",
    )
    rec2 = RecommendationRecord(
        recommended_at=datetime(2026, 10, 28, 12, 5, 0, tzinfo=timezone.utc),
        ticker="$AMD",
        action="BUY",
        initial_price=150.00,
        confidence_score=8,
        opportunity_score=7.0,
        headline="AMD Data Center Expansion",
        source="Bloomberg",
        article_hash="hash_2",
    )

    rec_id1 = storage.record_recommendation(rec1)
    rec_id2 = storage.record_recommendation(rec2)
    assert rec_id1 > 0
    assert rec_id2 > 0

    recs = storage.get_recommendations()
    assert len(recs) == 2
    # Ordered by recommended_at DESC
    assert recs[0].ticker == "$AMD"
    assert recs[1].ticker == "$NVDA"

    # Filter by ticker
    nvda_recs = storage.get_recommendations(ticker="NVDA")
    assert len(nvda_recs) == 1
    assert nvda_recs[0].ticker == "$NVDA"
    assert nvda_recs[0].initial_price == 135.50

    # Update price and return
    storage.update_recommendation_price(rec_id1, current_price=142.275, return_pct=5.0)
    updated_recs = storage.get_recommendations(ticker="NVDA")
    assert updated_recs[0].latest_price == 142.275
    assert updated_recs[0].return_pct == 5.0

    # Deduplication check: recording same article_hash and ticker should not duplicate
    dup_id = storage.record_recommendation(rec1)
    recs_after_dup = storage.get_recommendations()
    assert len(recs_after_dup) == 2

    storage.close()


def test_pipeline_records_recommendation_on_alert(tmp_path):
    db_file = tmp_path / "test_pipeline_recs.db"
    storage = SQLiteStorage(db_file)

    article = RawArticle(
        title="Taiwan Semi surges on record packaging demand",
        url="https://example.com/tsm-news",
        source_name="Financial Times",
        published_at=datetime.now(timezone.utc),
        raw_content="TSMC reports surging demand for advanced packaging.",
    )

    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade="HIGH",
        summary="TSMC reports surging demand for advanced packaging.",
        catalyst_thesis="Advanced packaging demand will boost forward EPS and margins.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        affected_tickers=["$TSM"],
        causal_chain=CausalChain(
            event_facts="TSMC packaging capacity booked out through 2027.",
            economic_mechanism="Surging demand -> high revenue",
            beneficiaries=["$TSM"],
            negatively_exposed=[],
        ),
        two_layer_gate=TwoLayerGateDecision(
            event_quality_score=9.0,
            opportunity_score=8.5,
            gate_verdict="OPPORTUNITY_CANDIDATE",
            audit_reason="Passed all hard gates.",
        ),
        technicals=[
            TechnicalIndicatorSummary(
                symbol="TSM",
                close_price=195.40,
                rsi=45.2,
                recommendation="BUY",
            )
        ],
    )

    class MockNotifier:
        def __init__(self):
            self.alerts = []

        def send_alert(self, art, ana):
            self.alerts.append((art, ana))
            return True

    notifier = MockNotifier()
    pipeline = Pipeline(
        sources=[],
        analyzer=None,
        storage=storage,
        notifier=notifier,
        min_confidence_score=7,
        buy_opportunities_only=True,
        batch_notifications=False,
    )

    # Simulate delivering the alert through helper
    pipeline._record_delivered_recommendation(article, analysis)

    recs = storage.get_recommendations()
    assert len(recs) == 1
    assert recs[0].ticker == "$TSM"
    assert recs[0].initial_price == 195.40
    assert recs[0].confidence_score == 9
    assert recs[0].opportunity_score == 8.5
    assert recs[0].headline == "Taiwan Semi surges on record packaging demand"
    assert recs[0].source == "Financial Times"

    storage.close()


def test_cli_performance_shows_recommendations(tmp_path):
    from typer.testing import CliRunner
    from src.cli import app

    runner = CliRunner()
    db_file = tmp_path / "test_cli_perf.db"
    storage = SQLiteStorage(db_file)

    rec = RecommendationRecord(
        recommended_at=datetime(2026, 10, 28, 14, 30, tzinfo=timezone.utc),
        ticker="$NVDA",
        action="BUY",
        initial_price=135.50,
        confidence_score=9,
        opportunity_score=8.5,
        headline="NVIDIA Breakthrough in AI",
        source="Reuters",
        article_hash="cli_hash_1",
    )
    storage.record_recommendation(rec)
    storage.close()

    result = runner.invoke(app, ["performance", "--db", str(db_file)])
    assert result.exit_code == 0
    assert "TELEGRAM BUY RECOMMENDATIONS TRACKING LEDGER" in result.output
    assert "$NVDA" in result.output
    assert "$135.50" in result.output
    assert "BUY" in result.output

