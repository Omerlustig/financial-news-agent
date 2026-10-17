import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from src.models import MarketImpactAnalysis, RawArticle
from src.storage.base import BaseStorage


class SQLiteStorage(BaseStorage):
    """SQLite implementation of article persistence and deduplication with WAL mode."""

    def __init__(self, db_path: Union[str, Path] = "data/agent.db") -> None:
        self.db_path = str(db_path)
        self._lock = threading.RLock()

        if self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            timeout=30.0,
        )
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema, WAL mode, and indexes."""
        with self._lock:
            # Enable WAL mode and pragmas for production robustness
            if self.db_path != ":memory:":
                self._conn.execute("PRAGMA journal_mode = WAL;")
            self._conn.execute("PRAGMA synchronous = NORMAL;")
            self._conn.execute("PRAGMA busy_timeout = 30000;")
            self._conn.execute("PRAGMA foreign_keys = ON;")

            # Schema creation with automatic transaction management
            with self._conn:
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS articles (
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
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_alerted ON articles(alerted);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_is_market_moving ON articles(is_market_moving);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);")

    def is_article_seen(self, content_hash: str) -> bool:
        """Check whether an article with the given content hash has already been stored."""
        if not content_hash:
            return False
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute("SELECT 1 FROM articles WHERE content_hash = ? LIMIT 1", (content_hash,))
            return cursor.fetchone() is not None

    def filter_unseen(self, articles: List[RawArticle]) -> List[RawArticle]:
        """Filter a list of articles, returning only those not yet recorded in storage.

        Deduplicates incoming articles within the batch, preserving the order of first occurrence.
        """
        if not articles:
            return []

        # Extract unique non-empty content hashes preserving order
        unique_hashes = list(dict.fromkeys(a.content_hash for a in articles if a.content_hash))
        if not unique_hashes:
            return []

        seen_hashes = set()
        chunk_size = 500  # Stays well within SQLite bind variable limits
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            for i in range(0, len(unique_hashes), chunk_size):
                chunk = unique_hashes[i:i + chunk_size]
                placeholders = ",".join("?" for _ in chunk)
                cursor.execute(
                    f"SELECT content_hash FROM articles WHERE content_hash IN ({placeholders})",
                    chunk,
                )
                for row in cursor.fetchall():
                    seen_hashes.add(row[0])

        unseen: List[RawArticle] = []
        batch_seen = set()
        for article in articles:
            h = article.content_hash
            if h not in seen_hashes and h not in batch_seen:
                batch_seen.add(h)
                unseen.append(article)

        return unseen

    def record_article(
        self,
        article: RawArticle,
        analysis: Optional[MarketImpactAnalysis] = None,
        alerted: bool = False,
    ) -> None:
        """Persist or update an article and its optional analysis and alert status."""
        published_at_str = article.published_at.isoformat() if article.published_at else None
        created_at_str = datetime.now(timezone.utc).isoformat()

        is_market_moving = int(analysis.is_market_moving) if analysis and analysis.is_market_moving is not None else None
        confidence_score = analysis.confidence_score if analysis and analysis.confidence_score is not None else None
        impact_grade = analysis.impact_grade.value if analysis and analysis.impact_grade is not None else None
        tickers = json.dumps(analysis.affected_tickers) if analysis and analysis.affected_tickers is not None else None
        sectors = json.dumps(analysis.affected_sectors) if analysis and analysis.affected_sectors is not None else None
        alerted_int = 1 if alerted else 0

        sql = """
        INSERT INTO articles (
            content_hash,
            url,
            title,
            source_name,
            published_at,
            created_at,
            is_market_moving,
            confidence_score,
            impact_grade,
            tickers,
            sectors,
            alerted
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(content_hash) DO UPDATE SET
            url = excluded.url,
            title = excluded.title,
            source_name = excluded.source_name,
            published_at = COALESCE(excluded.published_at, articles.published_at),
            is_market_moving = COALESCE(excluded.is_market_moving, articles.is_market_moving),
            confidence_score = COALESCE(excluded.confidence_score, articles.confidence_score),
            impact_grade = COALESCE(excluded.impact_grade, articles.impact_grade),
            tickers = COALESCE(excluded.tickers, articles.tickers),
            sectors = COALESCE(excluded.sectors, articles.sectors),
            alerted = CASE WHEN excluded.alerted = 1 THEN 1 ELSE articles.alerted END
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    sql,
                    (
                        article.content_hash,
                        article.url,
                        article.title,
                        article.source_name,
                        published_at_str,
                        created_at_str,
                        is_market_moving,
                        confidence_score,
                        impact_grade,
                        tickers,
                        sectors,
                        alerted_int,
                    ),
                )

    def get_stats(self) -> Dict[str, int]:
        """Return aggregate storage statistics: total_articles, alerted_count, market_moving_count."""
        sql = """
        SELECT
            COUNT(*) AS total_articles,
            COALESCE(SUM(CASE WHEN alerted = 1 THEN 1 ELSE 0 END), 0) AS alerted_count,
            COALESCE(SUM(CASE WHEN is_market_moving = 1 THEN 1 ELSE 0 END), 0) AS market_moving_count
        FROM articles
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql)
            row = cursor.fetchone()
            if row:
                return {
                    "total_articles": int(row[0] or 0),
                    "alerted_count": int(row[1] or 0),
                    "market_moving_count": int(row[2] or 0),
                }
            return {
                "total_articles": 0,
                "alerted_count": 0,
                "market_moving_count": 0,
            }

    def close(self) -> None:
        """Close SQLite database connection cleanly and reset reference for fail-fast behavior."""
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
