import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        """Initialize database schema, WAL mode, indexes, and migration safety checks."""
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
                        alerted INTEGER NOT NULL DEFAULT 0,
                        status TEXT,
                        directional_stance TEXT,
                        causal_mechanism TEXT,
                        beneficiaries TEXT,
                        negatively_exposed TEXT,
                        counterarguments TEXT,
                        invalidation_conditions TEXT,
                        passes_hard_gates INTEGER
                    );
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS feed_cache (
                        url TEXT PRIMARY KEY,
                        etag TEXT,
                        last_modified TEXT,
                        updated_at TIMESTAMP
                    );
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_metrics (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        content_hash TEXT,
                        model_name TEXT,
                        prompt_tokens INTEGER,
                        completion_tokens INTEGER,
                        total_tokens INTEGER,
                        cost_usd REAL,
                        created_at TIMESTAMP
                    );
                    """
                )
                # Migration safety check: ensure all newly specified columns exist in existing databases
                cursor = self._conn.execute("PRAGMA table_info(articles);")
                existing_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor.fetchall()}
                new_columns = [
                    ("status", "TEXT"),
                    ("directional_stance", "TEXT"),
                    ("causal_mechanism", "TEXT"),
                    ("beneficiaries", "TEXT"),
                    ("negatively_exposed", "TEXT"),
                    ("counterarguments", "TEXT"),
                    ("invalidation_conditions", "TEXT"),
                    ("passes_hard_gates", "INTEGER"),
                ]
                for col_name, col_type in new_columns:
                    if col_name not in existing_cols:
                        self._conn.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type};")

                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_alerted ON articles(alerted);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_is_market_moving ON articles(is_market_moving);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_created_at ON llm_metrics(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_content_hash ON llm_metrics(content_hash);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_model_name ON llm_metrics(model_name);")

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

        status = (
            analysis.status.value
            if analysis and hasattr(analysis.status, "value")
            else str(analysis.status)
            if analysis and analysis.status is not None
            else None
        )
        directional_stance = (
            analysis.directional_stance.value
            if analysis and hasattr(analysis.directional_stance, "value")
            else str(analysis.directional_stance)
            if analysis and analysis.directional_stance is not None
            else None
        )

        causal_mechanism = None
        beneficiaries = None
        negatively_exposed = None
        counterarguments = None
        invalidation_conditions = None
        if analysis and analysis.causal_chain is not None:
            causal_mechanism = analysis.causal_chain.economic_mechanism
            beneficiaries = (
                json.dumps(analysis.causal_chain.beneficiaries)
                if analysis.causal_chain.beneficiaries is not None
                else None
            )
            negatively_exposed = (
                json.dumps(analysis.causal_chain.negatively_exposed)
                if analysis.causal_chain.negatively_exposed is not None
                else None
            )
            counterarguments = analysis.causal_chain.counterarguments
            invalidation_conditions = analysis.causal_chain.invalidation_conditions

        passes_hard_gates = None
        if analysis and analysis.hard_gates is not None:
            passes_hard_gates = 1 if analysis.hard_gates.passes_all_gates else 0

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
            alerted,
            status,
            directional_stance,
            causal_mechanism,
            beneficiaries,
            negatively_exposed,
            counterarguments,
            invalidation_conditions,
            passes_hard_gates
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            alerted = CASE WHEN excluded.alerted = 1 THEN 1 ELSE articles.alerted END,
            status = COALESCE(excluded.status, articles.status),
            directional_stance = COALESCE(excluded.directional_stance, articles.directional_stance),
            causal_mechanism = COALESCE(excluded.causal_mechanism, articles.causal_mechanism),
            beneficiaries = COALESCE(excluded.beneficiaries, articles.beneficiaries),
            negatively_exposed = COALESCE(excluded.negatively_exposed, articles.negatively_exposed),
            counterarguments = COALESCE(excluded.counterarguments, articles.counterarguments),
            invalidation_conditions = COALESCE(excluded.invalidation_conditions, articles.invalidation_conditions),
            passes_hard_gates = COALESCE(excluded.passes_hard_gates, articles.passes_hard_gates)
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
                        status,
                        directional_stance,
                        causal_mechanism,
                        beneficiaries,
                        negatively_exposed,
                        counterarguments,
                        invalidation_conditions,
                        passes_hard_gates,
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

    def get_feed_cache(self, url: str) -> Tuple[Optional[str], Optional[str]]:
        """Retrieve cached ETag and Last-Modified headers for a feed URL."""
        if not url:
            return None, None
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute("SELECT etag, last_modified FROM feed_cache WHERE url = ?", (url,))
            row = cursor.fetchone()
            if row:
                etag = row["etag"] if isinstance(row, sqlite3.Row) else row[0]
                last_modified = row["last_modified"] if isinstance(row, sqlite3.Row) else row[1]
                return etag, last_modified
            return None, None

    def set_feed_cache(self, url: str, etag: Optional[str], last_modified: Optional[str]) -> None:
        """Store or update cached ETag and Last-Modified headers for a feed URL."""
        if not url:
            return
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO feed_cache (url, etag, last_modified, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(url) DO UPDATE SET
            etag = excluded.etag,
            last_modified = excluded.last_modified,
            updated_at = excluded.updated_at
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(sql, (url, etag, last_modified, now_str))

    def record_llm_metric(
        self,
        content_hash: str,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record token usage and financial cost telemetry for an LLM invocation."""
        now_str = datetime.now(timezone.utc).isoformat()
        sql = """
        INSERT INTO llm_metrics (
            content_hash,
            model_name,
            prompt_tokens,
            completion_tokens,
            total_tokens,
            cost_usd,
            created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    sql,
                    (
                        content_hash,
                        model_name,
                        int(prompt_tokens),
                        int(completion_tokens),
                        int(total_tokens),
                        float(cost_usd),
                        now_str,
                    ),
                )

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Return aggregate LLM usage telemetry and expenditure metrics."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()

            # Aggregate totals across all models and invocations
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_calls,
                    COALESCE(SUM(prompt_tokens), 0) AS total_prompt_tokens,
                    COALESCE(SUM(completion_tokens), 0) AS total_completion_tokens,
                    COALESCE(SUM(total_tokens), 0) AS total_tokens,
                    COALESCE(SUM(cost_usd), 0.0) AS total_cost_usd
                FROM llm_metrics;
                """
            )
            agg_row = cursor.fetchone()
            total_calls = int(agg_row[0] or 0)
            total_prompt_tokens = int(agg_row[1] or 0)
            total_completion_tokens = int(agg_row[2] or 0)
            total_tokens = int(agg_row[3] or 0)
            total_cost_usd = float(agg_row[4] or 0.0)

            # Breakdown by model
            cursor.execute(
                """
                SELECT model_name, COUNT(*) AS count
                FROM llm_metrics
                GROUP BY model_name
                ORDER BY count DESC;
                """
            )
            calls_by_model: Dict[str, int] = {}
            for row in cursor.fetchall():
                model = row[0] or ""
                calls_by_model[model] = int(row[1] or 0)

            # Monthly spend for current UTC month
            current_month_prefix = datetime.now(timezone.utc).strftime("%Y-%m")
            cursor.execute(
                """
                SELECT COALESCE(SUM(cost_usd), 0.0) AS month_cost
                FROM llm_metrics
                WHERE created_at LIKE ?;
                """,
                (f"{current_month_prefix}%",),
            )
            month_row = cursor.fetchone()
            spend_this_month = float(month_row[0] or 0.0)

            return {
                "total_calls": total_calls,
                "total_prompt_tokens": total_prompt_tokens,
                "total_completion_tokens": total_completion_tokens,
                "total_tokens": total_tokens,
                "total_cost_usd": total_cost_usd,
                "calls_by_model": calls_by_model,
                "spend_this_month": spend_this_month,
            }

    def close(self) -> None:
        """Close SQLite database connection cleanly and reset reference for fail-fast behavior."""
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
