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
    synchronous = conn.execute("PRAGMA synchronous;").fetchone()[0]
    conn.close()

    assert journal_mode.lower() == "wal"
    # synchronous 1 corresponds to NORMAL
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
    ]:
        assert expected_col in col_names

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




