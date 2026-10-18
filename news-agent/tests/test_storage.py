import concurrent.futures
import datetime
import json
import sqlite3
import pytest

from src.storage.base import BaseStorage
from src.storage.sqlite import SQLiteStorage
from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    DirectionalStance,
    CausalChain,
    HardGates,
    SECFilingItem,
    SurgeLevel,
    TelegramButton,
    SentimentLabel,
    SentimentScore,
    TrendVerdict,
    TrendMetrics,
    SectorTrendMetrics,
    StatementType,
    ProvenanceLevel,
    NoveltyState,
    FourDimensionalSentiment,
    AdversarialReview,
    TwoLayerGateDecision,
    EventLineageRecord,
)


def test_base_storage_cannot_be_instantiated_directly():
    with pytest.raises(TypeError):
        BaseStorage()  # type: ignore[abstract]


def test_deduplication_and_unseen_filter(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(db_path)

    a1 = RawArticle(title="Article 1", url="https://news.com/1", source_name="Reuters", published_at=None, summary="Test")
    a2 = RawArticle(title="Article 2", url="https://news.com/2", source_name="CNBC", published_at=None, summary="Test")

    assert storage.is_article_seen(a1.content_hash) is False
    assert storage.is_article_seen("") is False

    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=[],
        summary="Low impact",
        catalyst_thesis="Routine",
    )
    storage.record_article(a1, analysis, alerted=False)

    assert storage.is_article_seen(a1.content_hash) is True
    assert storage.is_article_seen(a2.content_hash) is False

    unseen = storage.filter_unseen([a1, a2])
    assert len(unseen) == 1
    assert unseen[0].title == "Article 2"
    storage.close()


def test_filter_unseen_edge_cases(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "edges.db"))

    # Empty list
    assert storage.filter_unseen([]) == []

    a1 = RawArticle(title="Article 1", url="https://news.com/1", source_name="Reuters", published_at=None, summary="Test")
    a2 = RawArticle(title="Article 2", url="https://news.com/2", source_name="Reuters", published_at=None, summary="Test")

    # In-batch duplicate items
    batch_with_duplicates = [a1, a2, a1, a2]
    unseen = storage.filter_unseen(batch_with_duplicates)
    assert len(unseen) == 2
    assert unseen[0].title == "Article 1"
    assert unseen[1].title == "Article 2"

    # Record a1
    storage.record_article(a1)
    # Now filter same batch with duplicates
    unseen2 = storage.filter_unseen(batch_with_duplicates)
    assert len(unseen2) == 1
    assert unseen2[0].title == "Article 2"

    # All seen
    storage.record_article(a2)
    assert storage.filter_unseen([a1, a2]) == []
    storage.close()


def test_filter_unseen_large_batch_chunking(tmp_path):
    # Tests batch query chunking across > 1000 items
    storage = SQLiteStorage(str(tmp_path / "large.db"))

    articles = [
        RawArticle(
            title=f"Bulk Article {i}",
            url=f"https://news.com/bulk/{i}",
            source_name="Wire",
            published_at=None,
            summary=f"Summary {i}",
        )
        for i in range(1200)
    ]

    # Pre-record even indexed articles
    for i in range(0, 1200, 2):
        storage.record_article(articles[i])

    unseen = storage.filter_unseen(articles)
    assert len(unseen) == 600
    for art in unseen:
        index = int(art.url.split("/")[-1])
        assert index % 2 != 0
    storage.close()


def test_record_article_upsert_and_data_integrity(tmp_path):
    db_file = tmp_path / "record.db"
    storage = SQLiteStorage(str(db_file))

    pub_time = datetime.datetime(2026, 10, 17, 10, 30, tzinfo=datetime.timezone.utc)
    article = RawArticle(
        title="NVIDIA Q3 Record Revenue",
        url="https://finance.example.com/nvda-q3",
        source_name="Financial Times",
        published_at=pub_time,
        summary="Data center revenue surges.",
    )

    # 1. Record without analysis initially
    storage.record_article(article, analysis=None, alerted=False)
    assert storage.is_article_seen(article.content_hash) is True

    # Inspect SQLite row directly
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (article.content_hash,)).fetchone()
    assert row is not None
    assert row["title"] == "NVIDIA Q3 Record Revenue"
    assert row["source_name"] == "Financial Times"
    assert row["published_at"] == pub_time.isoformat()
    assert row["is_market_moving"] is None
    assert row["confidence_score"] is None
    assert row["impact_grade"] is None
    assert row["alerted"] == 0
    conn.close()

    # 2. Upsert with analysis and alerted=True
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["NVDA", "TSM"],
        affected_sectors=["Semiconductors", "Artificial Intelligence"],
        summary="Datacenter compute demand drove historic beats.",
        catalyst_thesis="Immediate gross margin expansion and earnings revision upward.",
    )
    storage.record_article(article, analysis=analysis, alerted=True)

    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    updated_row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (article.content_hash,)).fetchone()
    assert updated_row["is_market_moving"] == 1
    assert updated_row["confidence_score"] == 9
    assert updated_row["impact_grade"] == "CRITICAL"
    assert updated_row["alerted"] == 1
    assert json.loads(updated_row["tickers"]) == ["$NVDA", "$TSM"]
    assert json.loads(updated_row["sectors"]) == ["Semiconductors", "Artificial Intelligence"]
    assert updated_row["status"] == "OPPORTUNITY_CANDIDATE"
    assert updated_row["directional_stance"] == "NEUTRAL"
    assert updated_row["causal_mechanism"] is None
    assert updated_row["passes_hard_gates"] is None
    conn.close()

    # 3. Upsert again with alerted=False should NOT clear alerted status if already alerted
    storage.record_article(article, analysis=analysis, alerted=False)
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    row3 = conn.execute("SELECT alerted FROM articles WHERE content_hash = ?", (article.content_hash,)).fetchone()
    assert row3["alerted"] == 1
    conn.close()
    storage.close()


def test_get_stats_empty_and_populated(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "stats.db"))

    # Initial stats
    empty_stats = storage.get_stats()
    assert empty_stats == {
        "total_articles": 0,
        "alerted_count": 0,
        "market_moving_count": 0,
    }

    # Add 1: Routine news (not market moving, not alerted)
    a1 = RawArticle(title="Routine 1", url="https://n.com/1", source_name="Src", published_at=None, summary="")
    an1 = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=2,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=[],
        summary="Quiet",
        catalyst_thesis="None",
    )
    storage.record_article(a1, an1, alerted=False)

    # Add 2: Market moving, but below alert threshold (not alerted)
    a2 = RawArticle(title="Moving 2", url="https://n.com/2", source_name="Src", published_at=None, summary="")
    an2 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=5,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$SPY"],
        affected_sectors=["Broad Market"],
        summary="Moderate shock",
        catalyst_thesis="Rates shift",
    )
    storage.record_article(a2, an2, alerted=False)

    # Add 3: High impact, market moving, alerted
    a3 = RawArticle(title="Critical 3", url="https://n.com/3", source_name="Src", published_at=None, summary="")
    an3 = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$AAPL"],
        affected_sectors=["Consumer Tech"],
        summary="Breakthrough",
        catalyst_thesis="New product category",
    )
    storage.record_article(a3, an3, alerted=True)

    # Add 4: No analysis provided (e.g. filtered out before analysis)
    a4 = RawArticle(title="Unanalyzed 4", url="https://n.com/4", source_name="Src", published_at=None, summary="")
    storage.record_article(a4, analysis=None, alerted=False)

    stats = storage.get_stats()
    assert stats["total_articles"] == 4
    assert stats["alerted_count"] == 1
    assert stats["market_moving_count"] == 2
    storage.close()


def test_wal_mode_and_pragmas(tmp_path):
    db_path = str(tmp_path / "wal_test.db")
    storage = SQLiteStorage(db_path)

    conn = sqlite3.connect(db_path)
    journal_mode = conn.execute("PRAGMA journal_mode;").fetchone()[0]
    conn.close()

    assert journal_mode.lower() == "wal"
    # PRAGMA synchronous is connection-scoped; check on the active storage connection (1 = NORMAL)
    synchronous = storage._conn.execute("PRAGMA synchronous;").fetchone()[0]
    assert synchronous == 1
    storage.close()


def test_directory_auto_creation(tmp_path):
    nested_db = tmp_path / "nested" / "deep" / "dir" / "agent.db"
    assert not nested_db.parent.exists()

    with SQLiteStorage(str(nested_db)) as storage:
        assert nested_db.parent.exists()
        a = RawArticle(title="Test", url="https://n.com/test", source_name="Test", published_at=None, summary="")
        storage.record_article(a)
        assert storage.is_article_seen(a.content_hash) is True


def test_concurrent_access_and_thread_safety(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "concurrent.db"))

    total_threads = 10
    articles_per_thread = 20

    def worker(thread_id: int):
        for i in range(articles_per_thread):
            art = RawArticle(
                title=f"Thread {thread_id} News {i}",
                url=f"https://news.com/{thread_id}/{i}",
                source_name=f"Feed-{thread_id}",
                published_at=None,
                summary=f"Worker payload {i}",
            )
            # Check seen before
            _ = storage.is_article_seen(art.content_hash)
            # Batch filter
            _ = storage.filter_unseen([art])
            # Record
            analysis = MarketImpactAnalysis(
                is_market_moving=(i % 2 == 0),
                confidence_score=7,
                impact_grade=ImpactGrade.HIGH,
                affected_tickers=["$TEST"],
                affected_sectors=["Tech"],
                summary=f"Summary {i}",
                catalyst_thesis=f"Thesis {i}",
            )
            storage.record_article(art, analysis=analysis, alerted=(i % 4 == 0))
            # Read stats
            _ = storage.get_stats()

    with concurrent.futures.ThreadPoolExecutor(max_workers=total_threads) as executor:
        futures = [executor.submit(worker, tid) for tid in range(total_threads)]
        for f in concurrent.futures.as_completed(futures):
            f.result()  # Will re-raise exceptions if any occurred

    final_stats = storage.get_stats()
    expected_total = total_threads * articles_per_thread
    assert final_stats["total_articles"] == expected_total
    # i % 2 == 0 happens half the time
    assert final_stats["market_moving_count"] == expected_total // 2
    # i % 4 == 0 happens 1/4 of the time
    assert final_stats["alerted_count"] == expected_total // 4
    storage.close()


def test_storage_fail_fast_on_closed(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "closed.db"))
    a = RawArticle(title="Test", url="https://n.com/1", source_name="Src", published_at=None, summary="")
    storage.record_article(a)
    assert storage.is_article_seen(a.content_hash) is True

    storage.close()
    assert storage._conn is None

    # Any subsequent operation should fail fast
    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.is_article_seen(a.content_hash)

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.filter_unseen([a])

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.record_article(a)

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_stats()

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_feed_cache("https://example.com/rss")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.set_feed_cache("https://example.com/rss", "etag-1", "mod-1")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.record_llm_metric("hash-1", "gemini-1.5-flash", 100, 50, 150, 0.001)

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_metrics_summary()

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.record_ticker_sentiment("$NVDA", 0.5, "BULLISH", "hash-1")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_ticker_sentiment_window("$NVDA")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_all_tickers_trend_summary()

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.record_sector_sentiment("Semiconductors", 0.5, "BULLISH", "hash-1")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_sector_sentiment_window("Semiconductors")

    with pytest.raises(RuntimeError, match="Database connection is closed"):
        storage.get_all_sectors_trend_summary()


def test_transaction_rollback_on_failure(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "rollback.db"))

    a = RawArticle(title="Rollback Test", url="https://n.com/rb", source_name="Src", published_at=None, summary="")

    # Create a temporary abort trigger to simulate failure during transaction execution
    with storage._lock:
        storage._conn.execute(
            "CREATE TRIGGER abort_insert BEFORE INSERT ON articles BEGIN SELECT RAISE(ABORT, 'Simulated transaction error'); END;"
        )

    with pytest.raises(sqlite3.IntegrityError, match="Simulated transaction error"):
        storage.record_article(a)

    # Verify that article was NOT saved (transaction rolled back)
    assert storage.is_article_seen(a.content_hash) is False
    assert storage.get_stats()["total_articles"] == 0

    # Drop trigger and verify normal operations succeed without hung transaction
    with storage._lock:
        storage._conn.execute("DROP TRIGGER abort_insert;")

    storage.record_article(a)
    assert storage.is_article_seen(a.content_hash) is True
    assert storage.get_stats()["total_articles"] == 1
    storage.close()


def test_record_article_persists_institutional_fields(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "institutional.db"))
    article = RawArticle(
        title="Semiconductor export license approvals",
        url="https://example.com/semi",
        source_name="Reuters",
    )
    chain = CausalChain(
        event_facts="Commerce Department grants special export waivers.",
        economic_mechanism="Direct revenue resumption for enterprise AI chips.",
        affected_sectors=["Semiconductors"],
        beneficiaries=["$NVDA", "$AMD"],
        negatively_exposed=["$INTC"],
        counterarguments="Licenses subject to 90-day revocable review.",
        invalidation_conditions="Commerce Department retracts waiver on national security grounds.",
    )
    gates = HardGates(
        is_novel=True,
        is_verified=True,
        has_direct_exposure=True,
        passes_all_gates=True,
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA", "$AMD"],
        affected_sectors=["Semiconductors"],
        summary="Export waivers approved for top GPU makers.",
        catalyst_thesis="Unlocks immediate shipment backlog.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        causal_chain=chain,
        hard_gates=gates,
    )
    storage.record_article(article, analysis, alerted=True)

    conn = sqlite3.connect(storage.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (article.content_hash,)).fetchone()
    assert row is not None
    assert row["status"] == "OPPORTUNITY_CANDIDATE"
    assert row["directional_stance"] == "BULLISH"
    assert row["causal_mechanism"] == "Direct revenue resumption for enterprise AI chips."
    assert json.loads(row["beneficiaries"]) == ["$NVDA", "$AMD"]
    assert json.loads(row["negatively_exposed"]) == ["$INTC"]
    assert row["counterarguments"] == "Licenses subject to 90-day revocable review."
    assert row["invalidation_conditions"] == "Commerce Department retracts waiver on national security grounds."
    assert row["passes_hard_gates"] == 1
    conn.close()
    storage.close()


def test_storage_schema_migration_safety(tmp_path):
    db_file = tmp_path / "legacy_migration.db"

    # 1. Create a legacy table without the new columns
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE articles (
            content_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source_name TEXT NOT NULL,
            published_at TEXT,
            created_at TEXT NOT NULL,
            is_market_moving INTEGER,
            confidence_score INTEGER,
            impact_grade TEXT,
            tickers TEXT,
            sectors TEXT,
            alerted INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    # Insert a legacy article
    conn.execute(
        """
        INSERT INTO articles (
            content_hash, url, title, source_name, published_at, created_at,
            is_market_moving, confidence_score, impact_grade, tickers, sectors, alerted
        ) VALUES ('legacy_hash', 'https://example.com/legacy', 'Legacy News', 'OldSource', NULL, '2026-01-01', 1, 7, 'HIGH', '["$AAPL"]', '["Tech"]', 0);
        """
    )
    conn.commit()
    conn.close()

    # 2. Open with SQLiteStorage, triggering _init_db and migrations
    storage = SQLiteStorage(str(db_file))

    # Verify legacy article is intact
    assert storage.is_article_seen("legacy_hash") is True

    # 3. Verify all 8 columns were added via ALTER TABLE
    conn = sqlite3.connect(str(db_file))
    conn.row_factory = sqlite3.Row
    cursor = conn.execute("PRAGMA table_info(articles);")
    col_names = {row["name"] for row in cursor.fetchall()}
    for expected_col in [
        "status",
        "directional_stance",
        "causal_mechanism",
        "beneficiaries",
        "negatively_exposed",
        "counterarguments",
        "invalidation_conditions",
        "passes_hard_gates",
        "sec_filing_item",
        "surge_level",
        "sentiment_score",
        "sentiment_label",
        "buzz_ratio",
        "trend_verdict",
    ]:
        assert expected_col in col_names

    # Verify ticker_sentiment_history table exists
    tbl_cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ticker_sentiment_history';")
    assert tbl_cursor.fetchone() is not None

    # 4. Insert new article with institutional fields into migrated database
    new_art = RawArticle(
        title="Migrated New Article",
        url="https://example.com/migrated",
        source_name="NewSource",
    )
    chain = CausalChain(
        event_facts="Facts",
        economic_mechanism="Mechanism",
        affected_sectors=["Energy"],
        beneficiaries=["$XOM"],
        negatively_exposed=["$NEE"],
        counterarguments="Risk factors",
        invalidation_conditions="Policy reversal",
    )
    gates = HardGates(
        is_novel=True,
        is_verified=True,
        has_direct_exposure=True,
        passes_all_gates=True,
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$XOM"],
        affected_sectors=["Energy"],
        summary="Summary",
        catalyst_thesis="Thesis",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        directional_stance=DirectionalStance.BULLISH,
        causal_chain=chain,
        hard_gates=gates,
    )
    storage.record_article(new_art, analysis, alerted=True)

    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (new_art.content_hash,)).fetchone()
    assert row["status"] == "OPPORTUNITY_CANDIDATE"
    assert row["directional_stance"] == "BULLISH"
    assert row["causal_mechanism"] == "Mechanism"
    assert json.loads(row["beneficiaries"]) == ["$XOM"]
    assert json.loads(row["negatively_exposed"]) == ["$NEE"]
    assert row["counterarguments"] == "Risk factors"
    assert row["invalidation_conditions"] == "Policy reversal"
    assert row["passes_hard_gates"] == 1

    conn.close()
    storage.close()


def test_feed_cache_operations(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "feed_cache.db"))

    # Non-existent or empty URL returns (None, None)
    assert storage.get_feed_cache("https://example.com/rss") == (None, None)
    assert storage.get_feed_cache("") == (None, None)

    # Empty URL does nothing on set
    storage.set_feed_cache("", "etag-empty", "mod-empty")

    # Set feed cache
    feed_url = "https://feeds.reuters.com/businessNews"
    etag_1 = '"12345abcdef"'
    last_mod_1 = "Fri, 17 Oct 2026 12:00:00 GMT"
    storage.set_feed_cache(feed_url, etag=etag_1, last_modified=last_mod_1)

    cached_etag, cached_mod = storage.get_feed_cache(feed_url)
    assert cached_etag == etag_1
    assert cached_mod == last_mod_1

    # Update cache (upsert)
    etag_2 = '"67890ghijkl"'
    last_mod_2 = "Fri, 17 Oct 2026 13:00:00 GMT"
    storage.set_feed_cache(feed_url, etag=etag_2, last_modified=last_mod_2)

    cached_etag_upd, cached_mod_upd = storage.get_feed_cache(feed_url)
    assert cached_etag_upd == etag_2
    assert cached_mod_upd == last_mod_2

    # Verify table schema directly
    conn = sqlite3.connect(storage.db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM feed_cache WHERE url = ?", (feed_url,)).fetchone()
    assert row is not None
    assert row["url"] == feed_url
    assert row["etag"] == etag_2
    assert row["last_modified"] == last_mod_2
    assert row["updated_at"] is not None
    conn.close()
    storage.close()


def test_llm_metrics_operations(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "llm_metrics.db"))

    # Initial summary on empty database
    empty_summary = storage.get_metrics_summary()
    assert empty_summary["total_calls"] == 0
    assert empty_summary["total_prompt_tokens"] == 0
    assert empty_summary["total_completion_tokens"] == 0
    assert empty_summary["total_tokens"] == 0
    assert empty_summary["total_cost_usd"] == 0.0
    assert empty_summary["calls_by_model"] == {}
    assert empty_summary["spend_this_month"] == 0.0

    # Record metrics for gemini
    storage.record_llm_metric(
        content_hash="hash_art_1",
        model_name="gemini/gemini-1.5-flash",
        prompt_tokens=400,
        completion_tokens=100,
        total_tokens=500,
        cost_usd=0.0001,
    )
    storage.record_llm_metric(
        content_hash="hash_art_2",
        model_name="gemini/gemini-1.5-flash",
        prompt_tokens=600,
        completion_tokens=200,
        total_tokens=800,
        cost_usd=0.0002,
    )

    # Record metric for fallback model gpt-4o-mini
    storage.record_llm_metric(
        content_hash="hash_art_3",
        model_name="gpt-4o-mini",
        prompt_tokens=500,
        completion_tokens=150,
        total_tokens=650,
        cost_usd=0.00015,
    )

    summary = storage.get_metrics_summary()
    assert summary["total_calls"] == 3
    assert summary["total_prompt_tokens"] == 1500
    assert summary["total_completion_tokens"] == 450
    assert summary["total_tokens"] == 1950
    assert pytest.approx(summary["total_cost_usd"], rel=1e-5) == 0.00045
    assert summary["calls_by_model"] == {
        "gemini/gemini-1.5-flash": 2,
        "gpt-4o-mini": 1,
    }
    assert pytest.approx(summary["spend_this_month"], rel=1e-5) == 0.00045

    # Insert a metric record from a previous month directly to test monthly spend isolation
    conn = sqlite3.connect(storage.db_path)
    conn.execute(
        """
        INSERT INTO llm_metrics (
            content_hash, model_name, prompt_tokens, completion_tokens, total_tokens, cost_usd, created_at
        ) VALUES ('past_hash', 'gemini/gemini-1.5-flash', 1000, 500, 1500, 0.05, '2025-01-15T10:00:00')
        """
    )
    conn.commit()
    conn.close()

    updated_summary = storage.get_metrics_summary()
    # Total calls and total cost should include the past month record
    assert updated_summary["total_calls"] == 4
    assert pytest.approx(updated_summary["total_cost_usd"], rel=1e-5) == 0.05045
    # spend_this_month must ONLY reflect current month spend
    assert pytest.approx(updated_summary["spend_this_month"], rel=1e-5) == 0.00045

    storage.close()


def test_record_article_persists_corroboration_fields(tmp_path):
    db_path = str(tmp_path / "corroboration_test.db")
    storage = SQLiteStorage(db_path)

    art = RawArticle(
        title="Antitrust Agency Sues Tech Giant",
        url="https://marketwatch.com/sue-giant",
        source_name="MarketWatch",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$GOOGL"],
        affected_sectors=["Cloud Computing"],
        summary="Suit filed.",
        catalyst_thesis="Antitrust breakup.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        corroborating_sources=["Bloomberg", "MarketWatch", "Reuters"],
        corroboration_count=3,
    )

    storage.record_article(art, analysis, alerted=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (art.content_hash,)).fetchone()
    conn.close()

    assert row is not None
    assert row["corroboration_count"] == 3
    sources = json.loads(row["corroborating_sources"])
    assert sources == ["Bloomberg", "MarketWatch", "Reuters"]

    storage.close()


def test_record_article_persists_sec_filing_and_surge_level(tmp_path):
    db_path = str(tmp_path / "sec_surge_test.db")
    storage = SQLiteStorage(db_path)

    art = RawArticle(
        title="Acme Corp Files Item 1.01 Material Agreement",
        url="https://sec.gov/edgar/data/12345/acme-8k",
        source_name="SEC Edgar",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$ACME"],
        affected_sectors=["Technology"],
        summary="Acme enters into material definitive acquisition agreement.",
        catalyst_thesis="Acquisition expands total addressable market by 40%.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        sec_filing_item=SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT,
        surge_level=SurgeLevel.SURGE_BREAKING,
        interactive_buttons=[
            TelegramButton(text="View 8-K", url="https://sec.gov/edgar/data/12345/acme-8k"),
            TelegramButton(text="Charts", url="https://tradingview.com/chart/?symbol=ACME"),
        ],
    )

    storage.record_article(art, analysis, alerted=True)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (art.content_hash,)).fetchone()
    conn.close()

    assert row is not None
    assert row["sec_filing_item"] == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT.value
    assert row["surge_level"] == "SURGE_BREAKING"

    # Also test record without analysis preserves None
    art_no_analysis = RawArticle(
        title="Routine Press Release",
        url="https://example.com/routine",
        source_name="PR Newswire",
    )
    storage.record_article(art_no_analysis, analysis=None, alerted=False)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row_no_analysis = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (art_no_analysis.content_hash,)).fetchone()
    conn.close()

    assert row_no_analysis is not None
    assert row_no_analysis["sec_filing_item"] is None
    assert row_no_analysis["surge_level"] is None

    storage.close()


def test_sec_and_surge_model_contracts():
    # Verify SECFilingItem values
    assert SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT.value == "Item 1.01: Entry into Material Definitive Agreement (M&A / Partnership)"
    assert SECFilingItem.ITEM_1_02_TERMINATION_AGREEMENT.value == "Item 1.02: Termination of a Material Agreement"
    assert SECFilingItem.ITEM_2_02_EARNINGS_OPERATIONS.value == "Item 2.02: Results of Operations and Financial Condition"
    assert SECFilingItem.ITEM_4_02_ACCOUNTING_RESTATEMENT.value == "Item 4.02: Non-Reliance on Previously Issued Financial Statements (Restatement)"
    assert SECFilingItem.ITEM_5_02_EXECUTIVE_CHANGE.value == "Item 5.02: Departure / Election of Key Officers or Directors"
    assert SECFilingItem.ITEM_GENERAL.value == "General Corporate Disclosure"

    # Verify SurgeLevel values
    assert SurgeLevel.NORMAL.value == "NORMAL"
    assert SurgeLevel.ELEVATED.value == "ELEVATED"
    assert SurgeLevel.SURGE_BREAKING.value == "SURGE_BREAKING"

    # Verify default surge_level and buttons on MarketImpactAnalysis
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=2,
        impact_grade=ImpactGrade.LOW,
        summary="Minor update",
        catalyst_thesis="No material market impact expected.",
    )
    assert analysis.surge_level == SurgeLevel.NORMAL
    assert analysis.sec_filing_item is None
    assert analysis.interactive_buttons == []

    # Verify normalization with string values
    analysis_str = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        summary="Breaking earnings",
        catalyst_thesis="Revenue beat by 20%.",
        sec_filing_item="Item 2.02: Results of Operations and Financial Condition",
        surge_level="elevated",
        interactive_buttons=[TelegramButton(text="Details", url="https://example.com")],
    )
    assert analysis_str.surge_level == SurgeLevel.ELEVATED
    assert analysis_str.sec_filing_item == SECFilingItem.ITEM_2_02_EARNINGS_OPERATIONS.value
    assert len(analysis_str.interactive_buttons) == 1
    assert analysis_str.interactive_buttons[0].text == "Details"


def test_record_article_persists_sentiment_and_trend(tmp_path):
    db_path = str(tmp_path / "sentiment_trend_test.db")
    storage = SQLiteStorage(db_path)

    art = RawArticle(
        title="NVIDIA Unveils Revolutionary Next-Gen AI Chip Architecture",
        url="https://techfinance.com/nvda-next-gen",
        source_name="TechFinance",
        published_at=datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.timezone.utc),
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$NVDA", "$TSM"],
        affected_sectors=["Semiconductors"],
        summary="NVIDIA reveals next-gen architecture accelerating compute capabilities 4x.",
        catalyst_thesis="Massive revenue expansion expected in cloud compute providers.",
        status=EventStatus.OPPORTUNITY_CANDIDATE,
        sentiment=SentimentScore(
            score=0.85,
            label=SentimentLabel.VERY_BULLISH,
            positive=0.9,
            negative=0.05,
            neutral=0.05,
            confidence=0.95,
        ),
        trend=TrendMetrics(
            buzz_ratio=3.2,
            sentiment_velocity=0.45,
            trend_verdict=TrendVerdict.STRONG_BULLISH,
            articles_24h_count=12,
        ),
    )

    storage.record_article(art, analysis, alerted=True)

    # 1. Verify columns in articles table
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM articles WHERE content_hash = ?", (art.content_hash,)).fetchone()
    assert row is not None
    assert row["sentiment_score"] == pytest.approx(0.85)
    assert row["sentiment_label"] == "VERY_BULLISH"
    assert row["buzz_ratio"] == pytest.approx(3.2)
    assert row["trend_verdict"] == "STRONG_BULLISH"

    # 2. Verify rows in ticker_sentiment_history for both tickers
    hist_rows = conn.execute(
        "SELECT * FROM ticker_sentiment_history WHERE article_hash = ? ORDER BY ticker ASC",
        (art.content_hash,),
    ).fetchall()
    conn.close()

    assert len(hist_rows) == 2
    assert hist_rows[0]["ticker"] == "$NVDA"
    assert hist_rows[0]["sentiment_score"] == pytest.approx(0.85)
    assert hist_rows[0]["sentiment_label"] == "VERY_BULLISH"
    assert hist_rows[1]["ticker"] == "$TSM"
    assert hist_rows[1]["sentiment_score"] == pytest.approx(0.85)

    storage.close()


def test_record_ticker_sentiment_and_window_query(tmp_path):
    db_path = str(tmp_path / "ticker_window_test.db")
    storage = SQLiteStorage(db_path)

    now = datetime.datetime.now(datetime.timezone.utc)
    recent_time = now - datetime.timedelta(hours=2)
    old_time = now - datetime.timedelta(hours=30)

    # Record recent sentiment
    storage.record_ticker_sentiment(
        ticker="$AAPL",
        sentiment_score=0.6,
        sentiment_label=SentimentLabel.BULLISH,
        article_hash="hash_aapl_1",
        published_at=recent_time,
    )

    # Record older sentiment directly into DB
    conn = sqlite3.connect(db_path)
    conn.execute(
        """
        INSERT INTO ticker_sentiment_history (ticker, sentiment_score, sentiment_label, article_hash, published_at, created_at)
        VALUES ('$AAPL', -0.5, 'BEARISH', 'hash_aapl_old', ?, ?)
        """,
        (old_time.isoformat(), old_time.isoformat()),
    )
    conn.commit()
    conn.close()

    # Query with 24h window - should only include recent record
    window = storage.get_ticker_sentiment_window("$AAPL", hours=24)
    assert len(window) == 1
    assert window[0]["article_hash"] == "hash_aapl_1"
    assert window[0]["sentiment_score"] == pytest.approx(0.6)

    # Query without leading '$' should also match
    window_no_dollar = storage.get_ticker_sentiment_window("AAPL", hours=24)
    assert len(window_no_dollar) == 1
    assert window_no_dollar[0]["ticker"] == "$AAPL"

    # Query with 48h window - should include both
    window_48h = storage.get_ticker_sentiment_window("$AAPL", hours=48)
    assert len(window_48h) == 2

    # Query non-existent ticker
    empty_window = storage.get_ticker_sentiment_window("$MSFT", hours=24)
    assert empty_window == []

    storage.close()


def test_get_all_tickers_trend_summary(tmp_path):
    db_path = str(tmp_path / "trend_summary_test.db")
    storage = SQLiteStorage(db_path)

    # Empty table returns empty list
    assert storage.get_all_tickers_trend_summary(hours=24) == []

    # Insert multiple data points for tickers
    now = datetime.datetime.now(datetime.timezone.utc)
    storage.record_ticker_sentiment("$NVDA", 0.8, "VERY_BULLISH", "hash_1", published_at=now)
    storage.record_ticker_sentiment("$NVDA", 0.6, "BULLISH", "hash_2", published_at=now)
    storage.record_ticker_sentiment("$TSLA", -0.4, "BEARISH", "hash_3", published_at=now)
    storage.record_ticker_sentiment("$GOOGL", 0.05, "NEUTRAL", "hash_4", published_at=now)

    summary = storage.get_all_tickers_trend_summary(hours=24)
    assert len(summary) == 3

    # Check $NVDA aggregate: avg=(0.8+0.6)/2 = 0.7 -> VERY_BULLISH
    nvda = next(s for s in summary if s["ticker"] == "$NVDA")
    assert nvda["total_articles"] == 2
    assert nvda["avg_sentiment"] == pytest.approx(0.7)
    assert nvda["sentiment_label"] == "VERY_BULLISH"

    # Check $TSLA aggregate: avg=-0.4 -> BEARISH
    tsla = next(s for s in summary if s["ticker"] == "$TSLA")
    assert tsla["total_articles"] == 1
    assert tsla["avg_sentiment"] == pytest.approx(-0.4)
    assert tsla["sentiment_label"] == "BEARISH"

    # Check $GOOGL aggregate: avg=0.05 -> NEUTRAL
    googl = next(s for s in summary if s["ticker"] == "$GOOGL")
    assert googl["total_articles"] == 1
    assert googl["avg_sentiment"] == pytest.approx(0.05)
    assert googl["sentiment_label"] == "NEUTRAL"

    storage.close()


def test_sentiment_and_trend_model_contracts():
    # Verify SentimentLabel values
    assert SentimentLabel.VERY_BEARISH.value == "VERY_BEARISH"
    assert SentimentLabel.BEARISH.value == "BEARISH"
    assert SentimentLabel.NEUTRAL.value == "NEUTRAL"
    assert SentimentLabel.BULLISH.value == "BULLISH"
    assert SentimentLabel.VERY_BULLISH.value == "VERY_BULLISH"

    # Verify TrendVerdict values
    assert TrendVerdict.STRONG_BULLISH.value == "STRONG_BULLISH"
    assert TrendVerdict.MODERATE_BULLISH.value == "MODERATE_BULLISH"
    assert TrendVerdict.NEUTRAL_CONSOLIDATION.value == "NEUTRAL_CONSOLIDATION"
    assert TrendVerdict.MODERATE_BEARISH.value == "MODERATE_BEARISH"
    assert TrendVerdict.PANIC_CAPITULATION.value == "PANIC_CAPITULATION"

    # Default MarketImpactAnalysis has default sentiment and None trend
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        summary="Test summary",
        catalyst_thesis="Test thesis",
    )
    assert analysis.sentiment.score == 0.0
    assert analysis.sentiment.label == SentimentLabel.NEUTRAL
    assert analysis.trend is None

    # Dict backward compatibility
    analysis_dict = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        summary="Test summary",
        catalyst_thesis="Test thesis",
        sentiment={"score": -0.65, "label": "very_bearish"},
        trend={"buzz_ratio": 2.1, "trend_verdict": "panic_capitulation"},
    )
    assert analysis_dict.sentiment.score == pytest.approx(-0.65)
    assert analysis_dict.sentiment.label == SentimentLabel.VERY_BEARISH
    assert analysis_dict.trend is not None
    assert analysis_dict.trend.buzz_ratio == pytest.approx(2.1)
    assert analysis_dict.trend.trend_verdict == TrendVerdict.PANIC_CAPITULATION


def test_record_article_persists_sector_sentiment(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "sector_article.db"))
    article = RawArticle(
        title="Next-gen GaN semiconductor foundries announced",
        url="https://example.com/gan-foundries",
        source_name="TechWire",
    )
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=8,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA", "$TSM"],
        affected_sectors=["Semiconductors", "Commercial Aviation"],
        summary="High-efficiency semiconductor production capacity expanding.",
        catalyst_thesis="Accelerates aerospace power electronics transition.",
        sentiment=SentimentScore(score=0.8, label=SentimentLabel.VERY_BULLISH),
    )
    storage.record_article(article, analysis, alerted=True)

    # Verify sector sentiment entries for each affected sector
    semi_window = storage.get_sector_sentiment_window("Semiconductors", hours=24)
    assert len(semi_window) == 1
    assert semi_window[0]["sector"] == "Semiconductors"
    assert semi_window[0]["sentiment_score"] == pytest.approx(0.8)
    assert semi_window[0]["sentiment_label"] == "VERY_BULLISH"
    assert semi_window[0]["article_hash"] == article.content_hash

    # Case-insensitive query test
    aviation_window = storage.get_sector_sentiment_window("commercial aviation", hours=24)
    assert len(aviation_window) == 1
    assert aviation_window[0]["sector"] == "Commercial Aviation"
    assert aviation_window[0]["sentiment_score"] == pytest.approx(0.8)
    assert aviation_window[0]["sentiment_label"] == "VERY_BULLISH"

    storage.close()


def test_record_sector_sentiment_and_window_query(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "sector_sentiment.db"))
    now = datetime.datetime.now(datetime.timezone.utc)
    old_time = now - datetime.timedelta(hours=36)
    recent_time = now - datetime.timedelta(hours=2)

    # 1. Record old data point for Defense (>24h ago)
    storage.record_sector_sentiment(
        sector="Defense",
        sentiment_score=-0.6,
        sentiment_label=SentimentLabel.VERY_BEARISH,
        article_hash="old_defense_hash",
        published_at=old_time,
    )

    # 2. Record recent data point for Defense (within 24h)
    storage.record_sector_sentiment(
        sector="Defense",
        sentiment_score=0.7,
        sentiment_label=SentimentLabel.VERY_BULLISH,
        article_hash="recent_defense_hash",
        published_at=recent_time,
    )

    # Query 24h window: should only return the recent one
    window_24h = storage.get_sector_sentiment_window("Defense", hours=24)
    assert len(window_24h) == 1
    assert window_24h[0]["article_hash"] == "recent_defense_hash"
    assert window_24h[0]["sentiment_score"] == pytest.approx(0.7)

    # Query 48h window: should return both
    window_48h = storage.get_sector_sentiment_window("Defense", hours=48)
    assert len(window_48h) == 2

    # Query empty/unknown sector
    unknown = storage.get_sector_sentiment_window("NonexistentSector", hours=24)
    assert unknown == []

    # Query with empty string
    empty = storage.get_sector_sentiment_window("", hours=24)
    assert empty == []

    storage.close()


def test_get_all_sectors_trend_summary(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "sector_summary.db"))
    now = datetime.datetime.now(datetime.timezone.utc)

    # Insert multiple data points across sectors
    storage.record_sector_sentiment("Semiconductors", 0.8, "VERY_BULLISH", "h1", published_at=now)
    storage.record_sector_sentiment("Semiconductors", 0.6, "BULLISH", "h2", published_at=now)
    storage.record_sector_sentiment("Automotive", -0.4, "BEARISH", "h3", published_at=now)
    storage.record_sector_sentiment("Energy", 0.05, "NEUTRAL", "h4", published_at=now)

    summary = storage.get_all_sectors_trend_summary(hours=24)
    assert len(summary) == 3

    # Check Semiconductors: avg=(0.8+0.6)/2 = 0.7 -> VERY_BULLISH
    semi = next(s for s in summary if s["sector"] == "Semiconductors")
    assert semi["total_articles"] == 2
    assert semi["avg_sentiment"] == pytest.approx(0.7)
    assert semi["sentiment_label"] == "VERY_BULLISH"

    # Check Automotive: avg=-0.4 -> BEARISH
    auto = next(s for s in summary if s["sector"] == "Automotive")
    assert auto["total_articles"] == 1
    assert auto["avg_sentiment"] == pytest.approx(-0.4)
    assert auto["sentiment_label"] == "BEARISH"

    # Check Energy: avg=0.05 -> NEUTRAL
    energy = next(s for s in summary if s["sector"] == "Energy")
    assert energy["total_articles"] == 1
    assert energy["avg_sentiment"] == pytest.approx(0.05)
    assert energy["sentiment_label"] == "NEUTRAL"

    storage.close()


def test_storage_schema_migration_includes_sector_sentiment_history(tmp_path):
    db_file = tmp_path / "legacy_migration_sector.db"

    # 1. Create a legacy table without sector_sentiment_history
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE articles (
            content_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source_name TEXT NOT NULL,
            published_at TEXT,
            created_at TEXT NOT NULL,
            is_market_moving INTEGER,
            confidence_score INTEGER,
            impact_grade TEXT,
            tickers TEXT,
            sectors TEXT,
            alerted INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()

    # 2. Instantiate SQLiteStorage on this legacy DB
    storage = SQLiteStorage(str(db_file))

    # Verify sector_sentiment_history is auto-created and operations work seamlessly
    storage.record_sector_sentiment(
        sector="Biotechnology",
        sentiment_score=0.9,
        sentiment_label="VERY_BULLISH",
        article_hash="bio_hash_1",
    )
    records = storage.get_sector_sentiment_window("Biotechnology", hours=24)
    assert len(records) == 1
    assert records[0]["sentiment_score"] == pytest.approx(0.9)
    storage.close()


def test_trend_metrics_acceleration_and_divergence_contracts():
    # Defaults
    t = TrendMetrics()
    assert t.buzz_ratio == 1.0
    assert t.sentiment_velocity == 0.0
    assert t.sentiment_acceleration == 0.0
    assert t.is_divergent is False
    assert t.divergence_type is None
    assert t.trend_verdict == TrendVerdict.NEUTRAL_CONSOLIDATION

    # Custom divergence values
    t_div = TrendMetrics(
        buzz_ratio=3.5,
        sentiment_velocity=-0.45,
        sentiment_acceleration=-0.2,
        is_divergent=True,
        divergence_type="BEARISH_PANIC_LIQUIDATION",
        trend_verdict=TrendVerdict.PANIC_CAPITULATION,
    )
    assert t_div.is_divergent is True
    assert t_div.divergence_type == "BEARISH_PANIC_LIQUIDATION"
    assert t_div.sentiment_acceleration == pytest.approx(-0.2)
    assert t_div.trend_verdict == TrendVerdict.PANIC_CAPITULATION


def test_institutional_models_contract_and_serialization():
    # 1. Enums
    assert StatementType.VERIFIED_FACT == "VERIFIED_FACT"
    assert StatementType("attributed_claim") == StatementType.ATTRIBUTED_CLAIM
    assert ProvenanceLevel.PRIMARY_OFFICIAL == "PRIMARY_OFFICIAL"
    assert ProvenanceLevel("tier1_financial") == ProvenanceLevel.TIER1_FINANCIAL
    assert NoveltyState.GENUINE_SURPRISE == "GENUINE_SURPRISE"
    assert NoveltyState("in_line_confirmation") == NoveltyState.IN_LINE_CONFIRMATION

    # 2. FourDimensionalSentiment
    s4d = FourDimensionalSentiment(
        linguistic_sentiment=0.75,
        fundamental_impact=0.9,
        expectation_surprise=0.8,
        price_implication=0.85,
        surprise_verdict="BEAT",
    )
    assert s4d.linguistic_sentiment == pytest.approx(0.75)
    assert s4d.fundamental_impact == pytest.approx(0.9)
    assert s4d.expectation_surprise == pytest.approx(0.8)
    assert s4d.price_implication == pytest.approx(0.85)
    assert s4d.surprise_verdict == "BEAT"

    # Clamping behavior
    s4d_clamped = FourDimensionalSentiment(
        linguistic_sentiment=1.5,
        fundamental_impact=-2.0,
        expectation_surprise=10.0,
        price_implication=-5.0,
    )
    assert s4d_clamped.linguistic_sentiment == 1.0
    assert s4d_clamped.fundamental_impact == -1.0
    assert s4d_clamped.expectation_surprise == 1.0
    assert s4d_clamped.price_implication == -1.0

    # 3. AdversarialReview
    adv = AdversarialReview(
        bearish_counter_thesis="Margin pressure from high capex ramp",
        priced_in_risk="Stock already ran up 35% into the announcement",
        macro_confounding_risk="Broad sector beta rally driven by rate cut expectations",
        falsification_threshold="Gross margin dropping below 65% in Q3",
        is_thesis_robust=True,
    )
    assert adv.is_thesis_robust is True
    assert "Margin pressure" in adv.bearish_counter_thesis

    # 4. TwoLayerGateDecision
    gate = TwoLayerGateDecision(
        event_quality_score=8.5,
        opportunity_score=8.2,
        passes_event_quality_gate=True,
        passes_opportunity_gate=True,
        gate_verdict="ALERT",
        rejection_reason=None,
    )
    assert gate.passes_event_quality_gate is True
    assert gate.gate_verdict == "ALERT"

    # Score clamping
    gate_clamped = TwoLayerGateDecision(
        event_quality_score=15.0,
        opportunity_score=-3.0,
    )
    assert gate_clamped.event_quality_score == 10.0
    assert gate_clamped.opportunity_score == 0.0

    # 5. EventLineageRecord
    rec = EventLineageRecord(
        event_id="evt_test_123",
        title="NVIDIA announces Blackwell Ultra architecture",
        primary_ticker="NVDA",
        source_count=3,
        all_sources=["Reuters", "Bloomberg", "PR Newswire"],
        event_quality_score=9.0,
        opportunity_score=8.5,
        status="ALERT",
        initial_price=120.50,
        subsequent_return_1d=0.035,
        subsequent_return_7d=0.078,
    )
    assert rec.primary_ticker == "$NVDA"
    assert rec.source_count == 3
    assert len(rec.all_sources) == 3

    # 6. MarketImpactAnalysis backward compatibility & integration
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.HIGH,
        affected_tickers=["$NVDA"],
        summary="High conviction chip catalyst",
        catalyst_thesis="Next gen AI accelerator ramp",
        four_d_sentiment=s4d,
        adversarial_review=adv,
        two_layer_gate=gate,
        novelty_state=NoveltyState.GENUINE_SURPRISE,
        provenance_level=ProvenanceLevel.PRIMARY_OFFICIAL,
        event_id=rec.event_id,
    )
    assert analysis.four_d_sentiment.surprise_verdict == "BEAT"
    assert analysis.adversarial_review.is_thesis_robust is True
    assert analysis.two_layer_gate.gate_verdict == "ALERT"
    assert analysis.novelty_state == NoveltyState.GENUINE_SURPRISE
    assert analysis.provenance_level == ProvenanceLevel.PRIMARY_OFFICIAL
    assert analysis.event_id == "evt_test_123"

    # Serialization roundtrip
    dumped = analysis.model_dump()
    assert dumped["four_d_sentiment"]["linguistic_sentiment"] == pytest.approx(0.75)
    assert dumped["novelty_state"] == "GENUINE_SURPRISE"
    assert dumped["event_id"] == "evt_test_123"

    restored = MarketImpactAnalysis.model_validate(dumped)
    assert restored.event_id == "evt_test_123"
    assert restored.novelty_state == NoveltyState.GENUINE_SURPRISE


def test_event_lineage_crud_operations(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "lineage_crud.db"))

    # Initial query on non-existent event
    assert storage.get_event_lineage("non_existent_evt") is None

    # Create and record an event
    now = datetime.datetime.now(datetime.timezone.utc)
    record = EventLineageRecord(
        event_id="evt_aapl_q3",
        title="Apple reports record Q3 Services revenue",
        primary_ticker="AAPL",
        first_seen=now - datetime.timedelta(minutes=10),
        last_updated=now,
        source_count=1,
        all_sources=["Apple Investor Relations"],
        event_quality_score=9.5,
        opportunity_score=8.0,
        status="WATCHLIST",
        initial_price=225.0,
    )
    storage.record_event_lineage(record)

    # Fetch and verify
    fetched = storage.get_event_lineage("evt_aapl_q3")
    assert fetched is not None
    assert fetched.event_id == "evt_aapl_q3"
    assert fetched.title == "Apple reports record Q3 Services revenue"
    assert fetched.primary_ticker == "$AAPL"
    assert fetched.source_count == 1
    assert fetched.all_sources == ["Apple Investor Relations"]
    assert fetched.event_quality_score == pytest.approx(9.5)
    assert fetched.opportunity_score == pytest.approx(8.0)
    assert fetched.status == "WATCHLIST"
    assert fetched.initial_price == pytest.approx(225.0)
    assert fetched.subsequent_return_1d is None
    assert fetched.subsequent_return_7d is None

    # Update via record_event_lineage (corroboration update)
    updated_record = EventLineageRecord(
        event_id="evt_aapl_q3",
        title="Apple reports record Q3 Services revenue with AI upside",
        primary_ticker="$AAPL",
        first_seen=record.first_seen,
        last_updated=now + datetime.timedelta(minutes=5),
        source_count=2,
        all_sources=["Apple Investor Relations", "Reuters"],
        event_quality_score=9.8,
        opportunity_score=8.7,
        status="ALERT",
        initial_price=225.0,
    )
    storage.record_event_lineage(updated_record)

    fetched_updated = storage.get_event_lineage("evt_aapl_q3")
    assert fetched_updated is not None
    assert fetched_updated.title == "Apple reports record Q3 Services revenue with AI upside"
    assert fetched_updated.source_count == 2
    assert "Reuters" in fetched_updated.all_sources
    assert fetched_updated.opportunity_score == pytest.approx(8.7)
    assert fetched_updated.status == "ALERT"

    # Test update_event_return for 1d and 7d horizons
    storage.update_event_return(event_id="evt_aapl_q3", horizon="1d", return_pct=2.45)
    after_1d = storage.get_event_lineage("evt_aapl_q3")
    assert after_1d.subsequent_return_1d == pytest.approx(2.45)

    storage.update_event_return(event_id="evt_aapl_q3", horizon="7D", return_pct=5.12)
    after_7d = storage.get_event_lineage("evt_aapl_q3")
    assert after_7d.subsequent_return_7d == pytest.approx(5.12)

    # Invalid horizon raises ValueError
    with pytest.raises(ValueError):
        storage.update_event_return(event_id="evt_aapl_q3", horizon="30d", return_pct=10.0)

    storage.close()


def test_event_lineage_find_existing_event_and_token_matching(tmp_path):
    storage = SQLiteStorage(str(tmp_path / "lineage_matching.db"))

    now = datetime.datetime.now(datetime.timezone.utc)
    ev1 = EventLineageRecord(
        event_id="evt_nvda_blackwell",
        title="NVIDIA unveils Blackwell Ultra GPUs for sovereign AI clusters",
        primary_ticker="NVDA",
        first_seen=now - datetime.timedelta(hours=2),
        last_updated=now - datetime.timedelta(hours=2),
        source_count=2,
        all_sources=["Reuters", "Bloomberg"],
        event_quality_score=9.0,
        opportunity_score=8.5,
        status="ALERT",
    )
    storage.record_event_lineage(ev1)

    # 1. Matching: Same ticker, >= 2 overlapping title tokens ("blackwell", "ultra")
    match = storage.find_existing_event(
        ticker="NVDA",
        title_tokens=["blackwell", "ultra", "delays"],
        window_hours=48,
    )
    assert match is not None
    assert match.event_id == "evt_nvda_blackwell"

    # 2. Matching with '$' prefix in search ticker
    match_dollar = storage.find_existing_event(
        ticker="$NVDA",
        title_tokens=["Nvidia", "Sovereign", "Architecture"],
        window_hours=48,
    )
    assert match_dollar is not None
    assert match_dollar.event_id == "evt_nvda_blackwell"

    # 3. Non-matching: Only 1 token overlapping (< 2)
    single_overlap = storage.find_existing_event(
        ticker="NVDA",
        title_tokens=["NVIDIA", "Omniverse", "Robotics"],
        window_hours=48,
    )
    assert single_overlap is None

    # 4. Non-matching: Different ticker even with overlapping tokens
    diff_ticker = storage.find_existing_event(
        ticker="AMD",
        title_tokens=["Blackwell", "Ultra", "GPUs"],
        window_hours=48,
    )
    assert diff_ticker is None

    # 5. Outside time window:
    old_ev = EventLineageRecord(
        event_id="evt_msft_old",
        title="Microsoft announces OpenAI partnership expansion and Azure credits",
        primary_ticker="MSFT",
        first_seen=now - datetime.timedelta(hours=60),
        last_updated=now - datetime.timedelta(hours=60),
        source_count=1,
        all_sources=["Microsoft PR"],
        event_quality_score=8.0,
        opportunity_score=7.0,
    )
    storage.record_event_lineage(old_ev)

    # With window_hours=48 -> None (expired)
    assert storage.find_existing_event(
        ticker="MSFT",
        title_tokens=["OpenAI", "partnership", "expansion"],
        window_hours=48,
    ) is None

    # With window_hours=72 -> Matched
    found_old = storage.find_existing_event(
        ticker="MSFT",
        title_tokens=["OpenAI", "partnership", "expansion"],
        window_hours=72,
    )
    assert found_old is not None
    assert found_old.event_id == "evt_msft_old"

    # 6. Edge cases: empty ticker or empty tokens
    assert storage.find_existing_event("", ["test", "tokens"]) is None
    assert storage.find_existing_event("NVDA", []) is None
    assert storage.find_existing_event("NVDA", ["single"]) is None

    storage.close()


def test_storage_schema_migration_includes_event_lineage(tmp_path):
    db_file = tmp_path / "legacy_v2.db"
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE articles (
            content_hash TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source_name TEXT NOT NULL,
            published_at TEXT,
            created_at TEXT NOT NULL,
            is_market_moving INTEGER,
            confidence_score INTEGER,
            impact_grade TEXT,
            tickers TEXT,
            sectors TEXT,
            alerted INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    conn.commit()
    conn.close()

    # Open with SQLiteStorage
    storage = SQLiteStorage(str(db_file))

    # Record and retrieve event lineage to ensure auto-migration created the table seamlessly
    rec = EventLineageRecord(
        event_id="evt_migrated_1",
        title="Tesla initiates FSD v13 rollout",
        primary_ticker="TSLA",
        source_count=1,
        all_sources=["Electrek"],
        status="ALERT",
    )
    storage.record_event_lineage(rec)
    fetched = storage.get_event_lineage("evt_migrated_1")
    assert fetched is not None
    assert fetched.primary_ticker == "$TSLA"
    assert fetched.status == "ALERT"

    storage.close()







