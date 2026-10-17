import concurrent.futures
import datetime
import json
import sqlite3
import pytest

from src.storage.base import BaseStorage
from src.storage.sqlite import SQLiteStorage
from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade


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

