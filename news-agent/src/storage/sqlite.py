import json
import re
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.models import (
    EventLineageRecord,
    MarketImpactAnalysis,
    RawArticle,
    RecommendationRecord,
)
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
                        passes_hard_gates INTEGER,
                        corroborating_sources TEXT,
                        corroboration_count INTEGER DEFAULT 1,
                        sec_filing_item TEXT,
                        surge_level TEXT,
                        sentiment_score REAL,
                        sentiment_label TEXT,
                        buzz_ratio REAL,
                        trend_verdict TEXT
                    );
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS ticker_sentiment_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        ticker TEXT NOT NULL,
                        sentiment_score REAL NOT NULL,
                        sentiment_label TEXT NOT NULL,
                        article_hash TEXT NOT NULL,
                        published_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    );
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS sector_sentiment_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        sector TEXT NOT NULL,
                        sentiment_score REAL NOT NULL,
                        sentiment_label TEXT NOT NULL,
                        article_hash TEXT NOT NULL,
                        published_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS event_lineage (
                        event_id TEXT PRIMARY KEY,
                        title TEXT NOT NULL,
                        primary_ticker TEXT,
                        first_seen TIMESTAMP NOT NULL,
                        last_updated TIMESTAMP NOT NULL,
                        source_count INTEGER DEFAULT 1,
                        all_sources TEXT,
                        event_quality_score REAL DEFAULT 0.0,
                        opportunity_score REAL DEFAULT 0.0,
                        status TEXT DEFAULT 'WATCHLIST',
                        directional_stance TEXT DEFAULT 'NEUTRAL',
                        initial_price REAL,
                        subsequent_return_1d REAL,
                        subsequent_return_7d REAL
                    );
                    """
                )
                self._conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS recommendation_tracking (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        article_hash TEXT NOT NULL,
                        event_id TEXT,
                        recommended_at TIMESTAMP NOT NULL,
                        ticker TEXT NOT NULL,
                        action TEXT NOT NULL DEFAULT 'BUY',
                        initial_price REAL,
                        confidence_score INTEGER DEFAULT 0,
                        opportunity_score REAL DEFAULT 0.0,
                        headline TEXT NOT NULL,
                        source TEXT,
                        latest_price REAL,
                        return_pct REAL,
                        status TEXT DEFAULT 'OPEN'
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
                    ("corroborating_sources", "TEXT"),
                    ("corroboration_count", "INTEGER"),
                    ("sec_filing_item", "TEXT"),
                    ("surge_level", "TEXT"),
                    ("sentiment_score", "REAL"),
                    ("sentiment_label", "TEXT"),
                    ("buzz_ratio", "REAL"),
                    ("trend_verdict", "TEXT"),
                ]
                for col_name, col_type in new_columns:
                    if col_name not in existing_cols:
                        self._conn.execute(f"ALTER TABLE articles ADD COLUMN {col_name} {col_type};")

                # Migration safety check for event_lineage columns
                cursor_el = self._conn.execute("PRAGMA table_info(event_lineage);")
                existing_el_cols = {row["name"] if isinstance(row, sqlite3.Row) else row[1] for row in cursor_el.fetchall()}
                if "directional_stance" not in existing_el_cols:
                    self._conn.execute("ALTER TABLE event_lineage ADD COLUMN directional_stance TEXT DEFAULT 'NEUTRAL';")

                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_alerted ON articles(alerted);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_is_market_moving ON articles(is_market_moving);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_created_at ON articles(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_status ON articles(status);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_articles_surge_level ON articles(surge_level);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiment_ticker_created_at ON ticker_sentiment_history(ticker, created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_ticker_sentiment_created_at ON ticker_sentiment_history(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sector_sentiment_sector_created_at ON sector_sentiment_history(sector, created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_sector_sentiment_created_at ON sector_sentiment_history(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_created_at ON llm_metrics(created_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_content_hash ON llm_metrics(content_hash);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_metrics_model_name ON llm_metrics(model_name);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_lineage_primary_ticker ON event_lineage(primary_ticker);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_lineage_first_seen ON event_lineage(first_seen);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_lineage_last_updated ON event_lineage(last_updated);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_event_lineage_status ON event_lineage(status);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_tracking_ticker ON recommendation_tracking(ticker);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_tracking_recommended_at ON recommendation_tracking(recommended_at);")
                self._conn.execute("CREATE INDEX IF NOT EXISTS idx_rec_tracking_status ON recommendation_tracking(status);")
                self._conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_rec_tracking_unique_article_ticker ON recommendation_tracking(article_hash, ticker);")

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

        corroborating_sources = (
            json.dumps(analysis.corroborating_sources)
            if analysis and analysis.corroborating_sources
            else None
        )
        corroboration_count = (
            analysis.corroboration_count
            if analysis and analysis.corroboration_count is not None
            else 1
        )
        sec_filing_item = (
            analysis.sec_filing_item.value
            if analysis and hasattr(analysis.sec_filing_item, "value")
            else str(analysis.sec_filing_item)
            if analysis and analysis.sec_filing_item is not None
            else None
        )
        surge_level = (
            analysis.surge_level.value
            if analysis and hasattr(analysis.surge_level, "value")
            else str(analysis.surge_level)
            if analysis and analysis.surge_level is not None
            else None
        )
        sentiment_score = None
        sentiment_label = None
        buzz_ratio = None
        trend_verdict = None

        if analysis and analysis.sentiment is not None:
            sentiment_score = float(analysis.sentiment.score)
            sentiment_label = (
                analysis.sentiment.label.value
                if hasattr(analysis.sentiment.label, "value")
                else str(analysis.sentiment.label)
            )

        if analysis and analysis.trend is not None:
            buzz_ratio = float(analysis.trend.buzz_ratio)
            trend_verdict = (
                analysis.trend.trend_verdict.value
                if hasattr(analysis.trend.trend_verdict, "value")
                else str(analysis.trend.trend_verdict)
            )

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
            passes_hard_gates,
            corroborating_sources,
            corroboration_count,
            sec_filing_item,
            surge_level,
            sentiment_score,
            sentiment_label,
            buzz_ratio,
            trend_verdict
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
            passes_hard_gates = COALESCE(excluded.passes_hard_gates, articles.passes_hard_gates),
            corroborating_sources = COALESCE(excluded.corroborating_sources, articles.corroborating_sources),
            corroboration_count = COALESCE(excluded.corroboration_count, articles.corroboration_count),
            sec_filing_item = COALESCE(excluded.sec_filing_item, articles.sec_filing_item),
            surge_level = COALESCE(excluded.surge_level, articles.surge_level),
            sentiment_score = COALESCE(excluded.sentiment_score, articles.sentiment_score),
            sentiment_label = COALESCE(excluded.sentiment_label, articles.sentiment_label),
            buzz_ratio = COALESCE(excluded.buzz_ratio, articles.buzz_ratio),
            trend_verdict = COALESCE(excluded.trend_verdict, articles.trend_verdict)
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
                        corroborating_sources,
                        corroboration_count,
                        sec_filing_item,
                        surge_level,
                        sentiment_score,
                        sentiment_label,
                        buzz_ratio,
                        trend_verdict,
                    ),
                )
                if analysis and analysis.affected_tickers and analysis.sentiment is not None:
                    for ticker in analysis.affected_tickers:
                        cleaned_ticker = str(ticker).strip().upper()
                        if cleaned_ticker:
                            self._conn.execute(
                                """
                                INSERT INTO ticker_sentiment_history (
                                    ticker, sentiment_score, sentiment_label, article_hash, published_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    cleaned_ticker,
                                    sentiment_score,
                                    sentiment_label,
                                    article.content_hash,
                                    published_at_str,
                                    created_at_str,
                                ),
                            )
                if analysis and analysis.affected_sectors and analysis.sentiment is not None:
                    for sector in analysis.affected_sectors:
                        cleaned_sector = str(sector).strip()
                        if cleaned_sector:
                            self._conn.execute(
                                """
                                INSERT INTO sector_sentiment_history (
                                    sector, sentiment_score, sentiment_label, article_hash, published_at, created_at
                                ) VALUES (?, ?, ?, ?, ?, ?)
                                """,
                                (
                                    cleaned_sector,
                                    sentiment_score,
                                    sentiment_label,
                                    article.content_hash,
                                    published_at_str,
                                    created_at_str,
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

    def get_recent_catalysts(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve recorded articles from the past `hours` where alerted=1 or status='OPPORTUNITY_CANDIDATE'."""
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()

        sql = """
        SELECT
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
            passes_hard_gates,
            corroborating_sources,
            corroboration_count,
            sec_filing_item,
            surge_level
        FROM articles
        WHERE (alerted = 1 OR status = 'OPPORTUNITY_CANDIDATE')
          AND (created_at >= ? OR (published_at IS NOT NULL AND published_at >= ?))
        ORDER BY confidence_score DESC, created_at DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (cutoff_iso, cutoff_iso))
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                if isinstance(r, sqlite3.Row):
                    results.append(dict(r))
                else:
                    col_names = [col[0] for col in cursor.description]
                    results.append(dict(zip(col_names, r)))
            return results

    def record_ticker_sentiment(
        self,
        ticker: str,
        sentiment_score: float,
        sentiment_label: str,
        article_hash: str,
        published_at: Optional[datetime] = None,
    ) -> None:
        """Record a sentiment history data point for a specific ticker."""
        if not ticker:
            return
        published_at_str = published_at.isoformat() if published_at else None
        created_at_str = datetime.now(timezone.utc).isoformat()
        cleaned_ticker = str(ticker).strip().upper()
        label_str = sentiment_label.value if hasattr(sentiment_label, "value") else str(sentiment_label)

        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO ticker_sentiment_history (
                        ticker, sentiment_score, sentiment_label, article_hash, published_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cleaned_ticker,
                        float(sentiment_score),
                        label_str,
                        article_hash,
                        published_at_str,
                        created_at_str,
                    ),
                )

    def get_ticker_sentiment_window(self, ticker: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve sentiment history records for a ticker within the past `hours`."""
        if not ticker:
            return []
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()
        cleaned_ticker = str(ticker).strip().upper()

        candidates = [cleaned_ticker]
        if cleaned_ticker.startswith("$"):
            candidates.append(cleaned_ticker[1:])
        else:
            candidates.append(f"${cleaned_ticker}")

        sql = """
        SELECT
            id,
            ticker,
            sentiment_score,
            sentiment_label,
            article_hash,
            published_at,
            created_at
        FROM ticker_sentiment_history
        WHERE (ticker = ? OR ticker = ?)
          AND COALESCE(published_at, created_at) >= ?
        ORDER BY COALESCE(published_at, created_at) DESC, id DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (candidates[0], candidates[1], cutoff_iso))

            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                if isinstance(r, sqlite3.Row):
                    results.append(dict(r))
                else:
                    col_names = [col[0] for col in cursor.description]
                    results.append(dict(zip(col_names, r)))
            return results

    def get_all_tickers_trend_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return aggregate sentiment trend summaries across all tickers in the past `hours`."""
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()

        sql = """
        SELECT
            ticker,
            AVG(sentiment_score) AS avg_sentiment,
            COUNT(*) AS total_articles
        FROM ticker_sentiment_history
        WHERE (created_at >= ? OR (published_at IS NOT NULL AND published_at >= ?))
        GROUP BY ticker
        ORDER BY total_articles DESC, avg_sentiment DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (cutoff_iso, cutoff_iso))
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                row_dict = dict(r) if isinstance(r, sqlite3.Row) else {
                    "ticker": r[0],
                    "avg_sentiment": r[1],
                    "total_articles": r[2],
                }
                avg_val = float(row_dict["avg_sentiment"]) if row_dict["avg_sentiment"] is not None else 0.0

                if avg_val >= 0.5:
                    label = "VERY_BULLISH"
                elif avg_val >= 0.15:
                    label = "BULLISH"
                elif avg_val <= -0.5:
                    label = "VERY_BEARISH"
                elif avg_val <= -0.15:
                    label = "BEARISH"
                else:
                    label = "NEUTRAL"

                results.append({
                    "ticker": row_dict["ticker"],
                    "avg_sentiment": round(avg_val, 4),
                    "total_articles": int(row_dict["total_articles"]),
                    "sentiment_label": label,
                })
            return results

    def record_sector_sentiment(
        self,
        sector: str,
        sentiment_score: float,
        sentiment_label: str,
        article_hash: str,
        published_at: Optional[datetime] = None,
    ) -> None:
        """Record a sentiment history data point for a specific sector."""
        if not sector:
            return
        published_at_str = published_at.isoformat() if published_at else None
        created_at_str = datetime.now(timezone.utc).isoformat()
        cleaned_sector = str(sector).strip()
        label_str = sentiment_label.value if hasattr(sentiment_label, "value") else str(sentiment_label)

        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    """
                    INSERT INTO sector_sentiment_history (
                        sector, sentiment_score, sentiment_label, article_hash, published_at, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        cleaned_sector,
                        float(sentiment_score),
                        label_str,
                        article_hash,
                        published_at_str,
                        created_at_str,
                    ),
                )

    def get_sector_sentiment_window(self, sector: str, hours: int = 24) -> List[Dict[str, Any]]:
        """Retrieve sentiment history records for a sector within the past `hours`."""
        if not sector:
            return []
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()
        cleaned_sector = str(sector).strip()

        sql = """
        SELECT
            id,
            sector,
            sentiment_score,
            sentiment_label,
            article_hash,
            published_at,
            created_at
        FROM sector_sentiment_history
        WHERE LOWER(sector) = LOWER(?)
          AND COALESCE(published_at, created_at) >= ?
        ORDER BY COALESCE(published_at, created_at) DESC, id DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (cleaned_sector, cutoff_iso))

            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                if isinstance(r, sqlite3.Row):
                    results.append(dict(r))
                else:
                    col_names = [col[0] for col in cursor.description]
                    results.append(dict(zip(col_names, r)))
            return results

    def get_all_sectors_trend_summary(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Return aggregate sentiment trend summaries across all sectors in the past `hours`."""
        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=hours)
        cutoff_iso = cutoff_dt.isoformat()

        sql = """
        SELECT
            sector,
            AVG(sentiment_score) AS avg_sentiment,
            COUNT(*) AS total_articles
        FROM sector_sentiment_history
        WHERE (created_at >= ? OR (published_at IS NOT NULL AND published_at >= ?))
        GROUP BY sector
        ORDER BY total_articles DESC, avg_sentiment DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (cutoff_iso, cutoff_iso))
            rows = cursor.fetchall()
            results: List[Dict[str, Any]] = []
            for r in rows:
                row_dict = dict(r) if isinstance(r, sqlite3.Row) else {
                    "sector": r[0],
                    "avg_sentiment": r[1],
                    "total_articles": r[2],
                }
                avg_val = float(row_dict["avg_sentiment"]) if row_dict["avg_sentiment"] is not None else 0.0

                if avg_val >= 0.5:
                    label = "VERY_BULLISH"
                elif avg_val >= 0.15:
                    label = "BULLISH"
                elif avg_val <= -0.5:
                    label = "VERY_BEARISH"
                elif avg_val <= -0.15:
                    label = "BEARISH"
                else:
                    label = "NEUTRAL"

                results.append({
                    "sector": row_dict["sector"],
                    "avg_sentiment": round(avg_val, 4),
                    "total_articles": int(row_dict["total_articles"]),
                    "sentiment_label": label,
                })
            return results

    def _row_to_event_lineage(self, row: Any) -> EventLineageRecord:
        """Convert a database row into an EventLineageRecord instance."""
        if isinstance(row, sqlite3.Row):
            d = dict(row)
        else:
            d = {
                "event_id": row[0],
                "title": row[1],
                "primary_ticker": row[2],
                "first_seen": row[3],
                "last_updated": row[4],
                "source_count": row[5],
                "all_sources": row[6],
                "event_quality_score": row[7],
                "opportunity_score": row[8],
                "status": row[9],
                "directional_stance": row[10] if len(row) > 13 else "NEUTRAL",
                "initial_price": row[11] if len(row) > 13 else row[10],
                "subsequent_return_1d": row[12] if len(row) > 13 else row[11],
                "subsequent_return_7d": row[13] if len(row) > 13 else row[12],
            }
        if "directional_stance" not in d or d["directional_stance"] is None:
            d["directional_stance"] = "NEUTRAL"
        all_sources = d.get("all_sources")
        if isinstance(all_sources, str):
            try:
                all_sources = json.loads(all_sources)
            except Exception:
                all_sources = [s.strip() for s in all_sources.split(",") if s.strip()]
        d["all_sources"] = all_sources or []
        return EventLineageRecord(**d)

    def record_event_lineage(self, event: EventLineageRecord) -> None:
        """Persist or update an event lineage tracking record."""
        if not event or not event.event_id:
            return

        first_seen_str = (
            event.first_seen.isoformat()
            if isinstance(event.first_seen, datetime)
            else str(event.first_seen)
        )
        last_updated_str = (
            event.last_updated.isoformat()
            if isinstance(event.last_updated, datetime)
            else str(event.last_updated)
        )
        sources_json = json.dumps(event.all_sources) if event.all_sources else "[]"
        cleaned_ticker = event.primary_ticker.strip().upper() if event.primary_ticker else None
        stance_str = event.directional_stance.strip().upper() if event.directional_stance else "NEUTRAL"

        sql = """
        INSERT INTO event_lineage (
            event_id,
            title,
            primary_ticker,
            first_seen,
            last_updated,
            source_count,
            all_sources,
            event_quality_score,
            opportunity_score,
            status,
            directional_stance,
            initial_price,
            subsequent_return_1d,
            subsequent_return_7d
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(event_id) DO UPDATE SET
            title = excluded.title,
            primary_ticker = COALESCE(excluded.primary_ticker, event_lineage.primary_ticker),
            last_updated = excluded.last_updated,
            source_count = excluded.source_count,
            all_sources = excluded.all_sources,
            event_quality_score = excluded.event_quality_score,
            opportunity_score = excluded.opportunity_score,
            status = excluded.status,
            directional_stance = COALESCE(excluded.directional_stance, event_lineage.directional_stance),
            initial_price = COALESCE(excluded.initial_price, event_lineage.initial_price),
            subsequent_return_1d = COALESCE(excluded.subsequent_return_1d, event_lineage.subsequent_return_1d),
            subsequent_return_7d = COALESCE(excluded.subsequent_return_7d, event_lineage.subsequent_return_7d);
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    sql,
                    (
                        event.event_id,
                        event.title,
                        cleaned_ticker,
                        first_seen_str,
                        last_updated_str,
                        int(event.source_count),
                        sources_json,
                        float(event.event_quality_score),
                        float(event.opportunity_score),
                        str(event.status),
                        stance_str,
                        float(event.initial_price) if event.initial_price is not None else None,
                        float(event.subsequent_return_1d) if event.subsequent_return_1d is not None else None,
                        float(event.subsequent_return_7d) if event.subsequent_return_7d is not None else None,
                    ),
                )

    def get_event_lineage(self, event_id: str) -> Optional[EventLineageRecord]:
        """Retrieve an event lineage record by unique event ID."""
        if not event_id:
            return None
        sql = """
        SELECT
            event_id,
            title,
            primary_ticker,
            first_seen,
            last_updated,
            source_count,
            all_sources,
            event_quality_score,
            opportunity_score,
            status,
            directional_stance,
            initial_price,
            subsequent_return_1d,
            subsequent_return_7d
        FROM event_lineage
        WHERE event_id = ?;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (event_id,))
            row = cursor.fetchone()
            if not row:
                return None
            return self._row_to_event_lineage(row)

    def find_existing_event(
        self, ticker: str, title_tokens: List[str], window_hours: int = 48
    ) -> Optional[EventLineageRecord]:
        """Search for an existing event lineage within window_hours matching ticker and >= 2 title tokens."""
        if not ticker or not title_tokens:
            return None

        cleaned_ticker = str(ticker).strip().upper()
        candidates = [cleaned_ticker]
        if cleaned_ticker.startswith("$"):
            candidates.append(cleaned_ticker[1:])
        else:
            candidates.append(f"${cleaned_ticker}")

        input_token_set = set()
        for t in title_tokens:
            words = re.findall(r"\b\w+\b", str(t).lower())
            input_token_set.update(words)

        if len(input_token_set) < 2:
            return None

        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=window_hours)

        sql = """
        SELECT
            event_id,
            title,
            primary_ticker,
            first_seen,
            last_updated,
            source_count,
            all_sources,
            event_quality_score,
            opportunity_score,
            status,
            directional_stance,
            initial_price,
            subsequent_return_1d,
            subsequent_return_7d
        FROM event_lineage
        WHERE (primary_ticker = ? OR primary_ticker = ?)
        ORDER BY last_updated DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (candidates[0], candidates[1]))
            rows = cursor.fetchall()

        for row in rows:
            record = self._row_to_event_lineage(row)

            rec_dt = record.last_updated or record.first_seen
            if rec_dt:
                rec_utc = rec_dt if rec_dt.tzinfo else rec_dt.replace(tzinfo=timezone.utc)
                if rec_utc < cutoff_dt:
                    continue

            event_tokens = set(re.findall(r"\b\w+\b", record.title.lower()))
            overlap = input_token_set.intersection(event_tokens)
            if len(overlap) >= 2:
                return record

        return None

    def update_event_return(self, event_id: str, horizon: str, return_pct: float) -> None:
        """Update forward realized return for a tracked event across specified horizon (e.g. 1d, 7d)."""
        if not event_id or not horizon:
            return

        norm_h = str(horizon).lower().strip().replace("-", "_")
        if "1d" in norm_h or "1_day" in norm_h:
            col = "subsequent_return_1d"
        elif "7d" in norm_h or "7_day" in norm_h:
            col = "subsequent_return_7d"
        else:
            raise ValueError(f"Unsupported return horizon: {horizon}. Expected '1d' or '7d'.")

        now_str = datetime.now(timezone.utc).isoformat()
        sql = f"""
        UPDATE event_lineage
        SET {col} = ?, last_updated = ?
        WHERE event_id = ?;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(sql, (float(return_pct), now_str, event_id))

    def get_pending_return_events(
        self, horizon: str = "1d", min_age_hours: int = 24
    ) -> List[EventLineageRecord]:
        """Retrieve tracked events eligible for forward return evaluation across specified horizon."""
        norm_h = str(horizon).lower().strip().replace("-", "_")
        if "1d" in norm_h or "1_day" in norm_h:
            return_col = "subsequent_return_1d"
        elif "7d" in norm_h or "7_day" in norm_h:
            return_col = "subsequent_return_7d"
        else:
            raise ValueError(f"Unsupported return horizon: {horizon}. Expected '1d' or '7d'.")

        cutoff_dt = datetime.now(timezone.utc) - timedelta(hours=min_age_hours)
        cutoff_iso = cutoff_dt.isoformat()

        sql = f"""
        SELECT
            event_id,
            title,
            primary_ticker,
            first_seen,
            last_updated,
            source_count,
            all_sources,
            event_quality_score,
            opportunity_score,
            status,
            directional_stance,
            initial_price,
            subsequent_return_1d,
            subsequent_return_7d
        FROM event_lineage
        WHERE initial_price IS NOT NULL
          AND initial_price > 0
          AND {return_col} IS NULL
          AND first_seen <= ?
        ORDER BY first_seen ASC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql, (cutoff_iso,))
            rows = cursor.fetchall()
            return [self._row_to_event_lineage(r) for r in rows]

    def get_resolved_lineage_records(self) -> List[EventLineageRecord]:
        """Retrieve all tracked events that have realized forward returns (1d or 7d)."""
        sql = """
        SELECT
            event_id,
            title,
            primary_ticker,
            first_seen,
            last_updated,
            source_count,
            all_sources,
            event_quality_score,
            opportunity_score,
            status,
            directional_stance,
            initial_price,
            subsequent_return_1d,
            subsequent_return_7d
        FROM event_lineage
        WHERE subsequent_return_1d IS NOT NULL OR subsequent_return_7d IS NOT NULL
        ORDER BY last_updated DESC;
        """
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            cursor.execute(sql)
            rows = cursor.fetchall()
            return [self._row_to_event_lineage(r) for r in rows]

    def list_event_lineage(self, ticker: Optional[str] = None, limit: int = 20) -> List[EventLineageRecord]:
        """List recent event lineage tracking records, optionally filtered by primary ticker."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            cursor = self._conn.cursor()
            if ticker:
                clean_ticker = ticker.strip().upper()
                if not clean_ticker.startswith("$"):
                    clean_ticker = f"${clean_ticker}"
                sql = """
                SELECT event_id, title, primary_ticker, first_seen, last_updated,
                       source_count, all_sources, event_quality_score, opportunity_score,
                       status, directional_stance, initial_price, subsequent_return_1d, subsequent_return_7d
                FROM event_lineage
                WHERE primary_ticker = ?
                ORDER BY last_updated DESC
                LIMIT ?;
                """
                cursor.execute(sql, (clean_ticker, limit))
            else:
                sql = """
                SELECT event_id, title, primary_ticker, first_seen, last_updated,
                       source_count, all_sources, event_quality_score, opportunity_score,
                       status, directional_stance, initial_price, subsequent_return_1d, subsequent_return_7d
                FROM event_lineage
                ORDER BY last_updated DESC
                LIMIT ?;
                """
                cursor.execute(sql, (limit,))
            rows = cursor.fetchall()
            return [self._row_to_event_lineage(r) for r in rows]

    def _row_to_recommendation(self, row: Any) -> RecommendationRecord:
        d = dict(row)
        return RecommendationRecord(
            id=d.get("id"),
            article_hash=d.get("article_hash", ""),
            event_id=d.get("event_id"),
            recommended_at=d.get("recommended_at"),
            ticker=d.get("ticker", ""),
            action=d.get("action", "BUY"),
            initial_price=d.get("initial_price"),
            confidence_score=d.get("confidence_score", 0),
            opportunity_score=d.get("opportunity_score", 0.0),
            headline=d.get("headline", ""),
            source=d.get("source", "Unknown"),
            latest_price=d.get("latest_price"),
            return_pct=d.get("return_pct"),
            status=d.get("status", "OPEN"),
        )

    def record_recommendation(self, record: RecommendationRecord) -> int:
        """Persist an auditable recommendation record upon Telegram notification dispatch."""
        if not record or not record.ticker:
            return 0
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                rec_at_str = (
                    record.recommended_at.isoformat()
                    if record.recommended_at
                    else datetime.now(timezone.utc).isoformat()
                )
                cursor = self._conn.execute(
                    """
                    INSERT INTO recommendation_tracking (
                        article_hash, event_id, recommended_at, ticker, action,
                        initial_price, confidence_score, opportunity_score,
                        headline, source, latest_price, return_pct, status
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(article_hash, ticker) DO UPDATE SET
                        initial_price = COALESCE(excluded.initial_price, recommendation_tracking.initial_price),
                        latest_price = COALESCE(excluded.latest_price, recommendation_tracking.latest_price),
                        return_pct = COALESCE(excluded.return_pct, recommendation_tracking.return_pct);
                    """,
                    (
                        record.article_hash,
                        record.event_id,
                        rec_at_str,
                        record.ticker,
                        record.action,
                        record.initial_price,
                        record.confidence_score,
                        record.opportunity_score,
                        record.headline,
                        record.source,
                        record.latest_price,
                        record.return_pct,
                        record.status,
                    ),
                )
                if cursor.lastrowid:
                    return int(cursor.lastrowid)
                # If updated on conflict, fetch existing id
                cur2 = self._conn.execute(
                    "SELECT id FROM recommendation_tracking WHERE article_hash = ? AND ticker = ?;",
                    (record.article_hash, record.ticker),
                )
                r2 = cur2.fetchone()
                return int(r2[0]) if r2 else 0

    def get_recommendations(
        self, limit: int = 50, ticker: Optional[str] = None, status: Optional[str] = None
    ) -> List[RecommendationRecord]:
        """Retrieve recorded recommendations, optionally filtered by ticker or status."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            query = "SELECT * FROM recommendation_tracking WHERE 1=1"
            params: List[Any] = []
            if ticker:
                cl_t = ticker.strip().upper()
                if not cl_t.startswith("$"):
                    cl_t = f"${cl_t}"
                query += " AND ticker = ?"
                params.append(cl_t)
            if status:
                query += " AND status = ?"
                params.append(status.strip().upper())
            query += " ORDER BY recommended_at DESC LIMIT ?;"
            params.append(limit)
            cursor = self._conn.execute(query, tuple(params))
            return [self._row_to_recommendation(r) for r in cursor.fetchall()]

    def update_recommendation_price(self, rec_id: int, current_price: float, return_pct: float) -> None:
        """Update realized latest price and return for a tracked recommendation."""
        with self._lock:
            if self._conn is None:
                raise RuntimeError("Database connection is closed.")
            with self._conn:
                self._conn.execute(
                    """
                    UPDATE recommendation_tracking
                    SET latest_price = ?, return_pct = ?
                    WHERE id = ?;
                    """,
                    (current_price, return_pct, rec_id),
                )

    def close(self) -> None:

        """Close SQLite database connection cleanly and reset reference for fail-fast behavior."""
        with self._lock:
            if hasattr(self, "_conn") and self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None
