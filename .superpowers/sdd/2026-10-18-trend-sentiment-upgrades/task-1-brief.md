# Task 1 Brief: Data Contracts, Storage Schema, and Watchlist Configuration

## Context
Upgrading the Market News Intelligence Agent to support advanced trend understanding, second-derivative sentiment acceleration, sector sentiment aggregation, and user preference watchlist profiles inspired by `nickhawn/news-agent`, `The-Swarm-Corporation/NewsAgent`, `Sujith8257/Flash_News_AI`, and `finaldie/auto-news`.

## Scope & Deliverables
All code and modifications MUST be in `news-agent/`.

1. **Model Upgrades (`news-agent/src/models.py`)**:
   - In `TrendVerdict(str, Enum)`: Add `BULLISH_REVERSAL = "BULLISH_REVERSAL"`, `BEARISH_BREAKDOWN = "BEARISH_BREAKDOWN"`. Preserve backward compatibility with existing members (`STRONG_BULLISH`, `MODERATE_BULLISH`, `NEUTRAL_CONSOLIDATION`, `MODERATE_BEARISH`, `PANIC_CAPITULATION`).
   - In `TrendMetrics(BaseModel)`:
     * Add `sentiment_acceleration: float = Field(default=0.0, description="Second derivative of sentiment (rate of change of velocity)")`
     * Add `is_divergent: bool = Field(default=False, description="True if buzz volume diverges significantly from sentiment direction")`
     * Add `divergence_type: Optional[str] = Field(default=None, description="Type of divergence: 'BEARISH_PANIC_LIQUIDATION', 'BULLISH_ACCUMULATION_DIVERGENCE', or None")`
   - Define `SectorTrendMetrics(BaseModel)`:
     * `sector: str`
     * `avg_sentiment: float = 0.0`
     * `sentiment_velocity: float = 0.0`
     * `sentiment_acceleration: float = 0.0`
     * `buzz_ratio: float = 1.0`
     * `article_count: int = 1`
     * `sentiment_label: SentimentLabel = SentimentLabel.NEUTRAL`
     * `trend_verdict: TrendVerdict = TrendVerdict.NEUTRAL_CONSOLIDATION`
   - In `MarketImpactAnalysis`:
     * Add `is_watchlist_match: bool = Field(default=False, description="True if catalyst matches user watchlist tickers or priority sectors")`
     * Add `matched_watchlist_items: List[str] = Field(default_factory=list, description="List of matched tickers or sectors from user watchlist")`
     * Add `sector_trends: List[SectorTrendMetrics] = Field(default_factory=list, description="Sector sentiment momentum metrics for affected sectors")`

2. **Configuration Upgrades (`news-agent/src/config.py` & `news-agent/config/settings.yaml`)**:
   - Define `WatchlistSettings(BaseModel)`:
     * `enabled: bool = Field(default=True)`
     * `tickers: List[str] = Field(default_factory=list, description="Watched tickers e.g. ['$NVDA', '$AAPL', '$TSLA']")`
     * `sectors: List[str] = Field(default_factory=list, description="Watched priority sectors e.g. ['Aviation', 'Defense', 'Semiconductors']")`
     * `score_boost: int = Field(default=1, description="Confidence score boost applied to matched watchlist items (1-2)")`
   - Attach `watchlist: WatchlistSettings = Field(default_factory=WatchlistSettings)` to `Settings` (or `FilterSettings`).
   - Update `news-agent/config/settings.yaml` with a clear `watchlist` section.

3. **Storage Engine Upgrades (`news-agent/src/storage/sqlite.py` & `news-agent/src/storage/base.py`)**:
   - In `news-agent/src/storage/base.py`:
     * Add abstract methods or default implementations:
       - `record_sector_sentiment(self, sector: str, sentiment_score: float, sentiment_label: str, article_hash: str, published_at: Optional[datetime] = None) -> None`
       - `get_sector_sentiment_window(self, sector: str, hours: int = 24) -> List[Dict[str, Any]]`
       - `get_all_sectors_trend_summary(self, hours: int = 24) -> List[Dict[str, Any]]`
   - In `news-agent/src/storage/sqlite.py`:
     * Create table `sector_sentiment_history`:
       ```sql
       CREATE TABLE IF NOT EXISTS sector_sentiment_history (
           id INTEGER PRIMARY KEY AUTOINCREMENT,
           sector TEXT NOT NULL,
           sentiment_score REAL NOT NULL,
           sentiment_label TEXT NOT NULL,
           article_hash TEXT NOT NULL,
           published_at TIMESTAMP,
           created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
       );
       ```
     * In `record_article()`:
       When recording an article, also record into `sector_sentiment_history` for each sector in `analysis.affected_sectors`.
     * Implement `record_sector_sentiment()`, `get_sector_sentiment_window()`, and `get_all_sectors_trend_summary()`.
     * Add migration check in `_init_db` ensuring `sector_sentiment_history` exists.

4. **Tests (`news-agent/tests/test_storage.py`, `news-agent/tests/test_config.py`)**:
   - Add unit tests for `sector_sentiment_history` persistence and window retrieval.
   - Add unit tests for `SectorTrendMetrics`, updated `TrendMetrics`, and `WatchlistSettings`.

5. **Verification**:
   - Run `python -m pytest news-agent/tests/ -v` and ensure 100% pass rate.
