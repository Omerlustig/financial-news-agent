# Task 4 Brief: Clustering Enhancements, Watchlist Priority Routing, and Heatmap/Scanner CLI

## Context
Upgrading the ingestion & pipeline clustering and user interface:
1. **Advanced Jaccard Token & Entity Clustering (`Flash_News_AI` & `auto-news`)**:
   - Improve `EventCorroborator` with multi-token phrase overlap, ticker recognition, and company entity matching.
   - Add noise filtering: detect and suppress zero-value syndicated noise releases.
2. **Watchlist Priority Routing (`pipeline.py`)**:
   - Check if incoming article or analysis matches user's `settings.watchlist.tickers` or `settings.watchlist.sectors`.
   - If match:
     * Set `analysis.is_watchlist_match = True`
     * Populate `analysis.matched_watchlist_items`
     * Apply confidence score boost (`settings.watchlist.score_boost`) if score < 10.
     * Ensure alert is prioritized.
3. **Heatmap & Scanner CLI Commands (`cli.py`)**:
   - `python -m src.main heatmap`:
     Renders a Rich terminal table showing Sector Trends (sector, average sentiment, velocity, acceleration, buzz ratio, verdict, article count) and Ticker Trends.
   - `python -m src.main scan`:
     Runs a cycle and displays both the scan summary, actionable bottom line, and the sector/ticker heatmap!
4. **Telegram Alert Badge**:
   - In `news-agent/src/notification/telegram.py`:
     If `analysis.is_watchlist_match`: display `⭐ [WATCHLIST FOCUS: {items}]` badge in alert header.
     If `analysis.trend` has `sentiment_acceleration`: format acceleration indicator (e.g. `⚡ Acceleration: +0.45 (BULLISH REVERSAL)`).

## Scope & Deliverables
All code and modifications MUST be in `news-agent/`.

1. **Update `news-agent/src/analysis/corroborator.py`**:
   - Improve tokenization and Jaccard entity clustering.
2. **Update `news-agent/src/pipeline.py`**:
   - Integrate watchlist matching and score boost.
   - Call `trend_engine.enrich_analysis_with_sector_trends`.
3. **Update `news-agent/src/notification/telegram.py`**:
   - Render watchlist badge and acceleration details.
4. **Update `news-agent/src/cli.py`**:
   - Implement `heatmap` and `scan` commands.
5. **Tests (`news-agent/tests/test_corroborator.py`, `news-agent/tests/test_pipeline.py`, `news-agent/tests/test_cli.py`)**:
   - Unit tests for watchlist boosting, heatmap rendering, and Jaccard clustering.
6. **Verification**:
   - Run `python -m pytest news-agent/tests/ -v` and ensure 100% test pass rate.
