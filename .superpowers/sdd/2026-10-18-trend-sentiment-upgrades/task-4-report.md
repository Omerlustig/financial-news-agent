# Task 4 Implementation Report: Clustering Enhancements, Watchlist Priority Routing, and Heatmap/Scanner CLI

**Date**: 2026-10-18  
**Status**: DONE  
**Scope**: `news-agent/`

---

## 1. Overview & Objectives
Task 4 upgraded the ingestion, clustering, prioritization, alerting, and terminal interfaces of `news-agent/`:
1. **Advanced Jaccard Token & Entity Clustering**:
   - Enhanced `EventCorroborator` with multi-token phrase extraction, capitalized multi-word entity detection, company-to-ticker mapping, and domain action synonym clustering.
   - Cross-newsroom headline deduplication ensures distinct newsrooms reporting the same real-world catalyst (e.g. Delta capacity cuts due to fuel costs) cluster into a single multi-source event.
   - Built zero-value syndicated noise filtering to suppress boilerplate PR wire notices and law firm solicitations.
2. **Watchlist Priority Routing & Score Boosting**:
   - Integrated user watchlist matching against catalyst affected tickers and sectors in `Pipeline`.
   - Matching watchlist items receive automatic priority confidence score boosts (+1 to +2), status elevation, and automatic promotion of grade to `HIGH` (`is_market_moving = True`) for score >= 7.
   - Wired `WatchlistSettings` throughout `Pipeline`, `MarketImpactAnalyzer`, and `EventCorroborator`.
3. **Telegram Alert Upgrades**:
   - Added priority header badge: `⭐ [WATCHLIST FOCUS: {items}]` in Telegram alert messages.
   - Added `⚡ Acceleration: {val} ({verdict})` indicator in trend velocity section when acceleration is present.
4. **Interactive CLI Commands (`heatmap` and `scan`)**:
   - Implemented `python -m src.cli heatmap [--hours 24]`: queries sector and ticker trend summaries and renders Rich tables color-coded by sentiment polarity (green for bullish, red for bearish, yellow for neutral).
   - Enhanced `run-once` and added `scan` command alias: executes cycle, displays execution metrics, outputs Actionable Investment Bottom Line card, and displays the Sector News Sentiment table if active sector news exists.

---

## 2. Modified & Created Components

### 2.1 `news-agent/src/analysis/corroborator.py`
- **Action Synonym & Domain Clusters (`SYNONYM_MAP`)**: Maps variations of cuts, hikes, drops, surges, aviation capacity, and fuel/oil to canonical cluster tokens (`token_cut`, `token_hike`, `token_drop`, `token_surge`, `token_aviation_capacity`, `token_fuel`, `token_cost`).
- **Company-to-Ticker Normalization (`COMPANY_TICKER_MAP`)**: Maps natural language company names ("Delta", "Apple", "Nvidia", etc.) directly to normalized tickers (`$DAL`, `$AAPL`, `$NVDA`).
- **Capitalized Multi-Word Entity Extraction**: Extracts multi-token entities (e.g. `Delta Air Lines`, `Jet Fuel`, `White House`, `Amazon Web Services`).
- **Ticker Extraction**: Extracts explicit `$TICKER` symbols.
- **Enhanced Similarity & Clustering (`compute_similarity`)**: Blends Jaccard overlap on enriched informative tokens with SequenceMatcher title similarity, plus entity and ticker overlap boosts.
- **Noise Filtering (`is_noise_article`)**: Flags and suppresses syndicated class action announcements, law firm solicitations, and routine webcast/conference reminders.

### 2.2 `news-agent/src/pipeline.py`
- **Watchlist Integration**:
  - `Pipeline.__init__` accepts optional `watchlist: WatchlistSettings` and passes it to `EventCorroborator` and `MarketImpactAnalyzer`.
  - In `process_articles()`: checks if analyzed tickers or sectors match configured watchlist.
  - If match: marks `analysis.is_watchlist_match = True`, sets `analysis.matched_watchlist_items`, boosts `analysis.confidence_score` by `score_boost`, and elevates `impact_grade = ImpactGrade.HIGH` and `is_market_moving = True` when score >= 7.
  - Suppresses syndicated noise articles from alerting (`EventStatus.NO_ACTION`, `is_market_moving = False`).

### 2.3 `news-agent/src/notification/telegram.py`
- **Watchlist Alert Badge**: Displays `⭐ [WATCHLIST FOCUS: {items}]` in the header of the notification message.
- **Trend Sentiment Acceleration**: Formats second derivative `⚡ <b>Acceleration:</b> {acceleration:+.2f} ({verdict})` when non-zero.

### 2.4 `news-agent/src/cli.py`
- **`heatmap` Command**:
  - Displays color-coded `Sector News Sentiment` and `Ticker News Sentiment` Rich tables across a configurable lookback window (`--hours`, default 24).
- **`scan` Command & Enhanced `run-once`**:
  - Executes full ingestion, analysis, deduplication, and notification cycle.
  - Displays cycle summary table and actionable investment bottom line cards.
  - Automatically queries and displays the `Sector News Sentiment Summary` table if active sector news is present.

### 2.5 Tests (`news-agent/tests/`)
- **`news-agent/tests/test_corroborator.py`**:
  - `test_cross_newsroom_headline_clustering_delta`: verifies that Delta fuel cut headlines from different newsrooms cluster together cleanly.
  - `test_noise_article_detection`: verifies identification and suppression of PR wire noise releases.
- **`news-agent/tests/test_pipeline.py`**:
  - `test_pipeline_watchlist_ticker_match_boosts_score_and_elevates_grade`: verifies ticker matching, score boost (+1), and grade elevation.
  - `test_pipeline_watchlist_sector_match`: verifies sector matching, score boost (+2), and alert trigger.
  - `test_pipeline_watchlist_disabled_no_boost`: verifies no boosting when watchlist is disabled.
- **`news-agent/tests/test_notification.py`**:
  - `test_format_telegram_message_watchlist_badge`: verifies header badge formatting.
  - `test_format_telegram_message_with_sentiment_acceleration`: verifies acceleration indicator formatting.
- **`news-agent/tests/test_cli.py`**:
  - `test_heatmap_command_empty`: verifies empty database message.
  - `test_heatmap_command_populated`: verifies rendering of sector and ticker heatmap tables with color-coded labels.
  - `test_scan_command_alias`: verifies execution of scan cycle and bottom line card.
  - `test_run_once_displays_sector_news_when_available`: verifies sector news summary table output when news exists.

---

## 3. Test Verification Results

All 303 unit and integration tests across the test suite passed with 100% success rate:

```
news-agent/tests/test_analysis.py: 52 passed
news-agent/tests/test_cli.py: 4 passed
news-agent/tests/test_config.py: 22 passed
news-agent/tests/test_corroborator.py: 8 passed
news-agent/tests/test_deploy.py: 7 passed
news-agent/tests/test_digest.py: 6 passed
news-agent/tests/test_extractor.py: 16 passed
news-agent/tests/test_ingestion.py: 25 passed
news-agent/tests/test_notification.py: 31 passed
news-agent/tests/test_pipeline.py: 37 passed
news-agent/tests/test_scanner.py: 3 passed
news-agent/tests/test_sentiment.py: 23 passed
news-agent/tests/test_storage.py: 27 passed
news-agent/tests/test_surge.py: 9 passed
news-agent/tests/test_taxonomy.py: 21 passed
news-agent/tests/test_trend.py: 32 passed

============================= 303 passed in 9.56s ==============================
```
