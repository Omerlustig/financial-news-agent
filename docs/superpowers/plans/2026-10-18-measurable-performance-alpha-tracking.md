# Measurable Performance & Publisher Credibility Alpha Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement automated forward return tracking, directional alpha metrics, win/loss hit rates, and publisher credibility scorecards to empirically verify recommendations and rate newsrooms.

**Architecture:** Extend `EventLineageRecord` with directional stance tracking; implement `PerformanceTrackerEngine` in `src/analysis/performance.py` to calculate forward returns (1d, 7d) and aggregate publisher performance scorecards; extend SQLite storage with queries for pending resolution events; and expose an interactive CLI command `performance`.

**Tech Stack:** Python 3.12, SQLite3 (WAL mode), Pydantic v2, Typer CLI, Rich formatting, Pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-measurable-performance-alpha-tracking.md`

## Global Constraints
- All code and tests must live in `news-agent/`.
- Git author identity must strictly be `Omerlustig <omerlustig@gmail.com>`.
- 100% test pass rate with 0 regressions.
- Absolute paths in file edits, no `cd` shell commands.

## Review Focus
1. Ticker formatting with and without `$` symbols ($NVDA vs NVDA) handled gracefully.
2. Division-by-zero protection when initial price is zero or no catalysts resolved.
3. Bearish stance invert: a falling stock price must count as positive directional alpha (+Alpha).
4. Multi-source attribution: an event corroborated by multiple publishers attributes performance metrics to each contributing newsroom.
5. In-flight events: catalysts less than 24h old must not be treated as 1d losses; they remain pending.

---

### Task 1: Data Contracts & Storage Extensions

**Files:**
- Modify: `news-agent/src/models.py`
- Modify: `news-agent/src/storage/base.py`
- Modify: `news-agent/src/storage/sqlite.py`
- Test: `news-agent/tests/test_storage.py`

- [ ] **Step 1: Write failing tests in `news-agent/tests/test_storage.py`**
  - Test `EventLineageRecord` with `directional_stance` field.
  - Test `storage.get_pending_return_events(horizon="1d", min_age_hours=24)`.
  - Test `storage.get_resolved_lineage_records()`.
- [ ] **Step 2: Add models to `news-agent/src/models.py`**
  - Add `CredibilityTier(str, Enum)`.
  - Add `SourceCredibilityRecord(BaseModel)`.
  - Add `PerformanceMetricsSummary(BaseModel)`.
  - Add `directional_stance: Optional[str] = "NEUTRAL"` to `EventLineageRecord`.
- [ ] **Step 3: Update `news-agent/src/storage/sqlite.py`**
  - Add `directional_stance` column to `event_lineage` table schema with migration check.
  - Implement `get_pending_return_events` and `get_resolved_lineage_records`.
- [ ] **Step 4: Run tests and ensure all pass**
  - `pytest news-agent/tests/test_storage.py -v`
- [ ] **Step 5: Commit**
  - `git commit -m "feat(storage): add directional stance to event lineage and pending return queries"`

---

### Task 2: Performance Tracker Engine & Alpha Math

**Files:**
- Create: `news-agent/src/analysis/performance.py`
- Modify: `news-agent/src/analysis/__init__.py`
- Test: `news-agent/tests/test_performance.py`

- [ ] **Step 1: Write failing unit tests in `news-agent/tests/test_performance.py`**
  - Test directional return math:
    * Bullish stance + price gain -> positive alpha, WIN.
    * Bullish stance + price drop -> negative alpha, LOSS.
    * Bearish stance + price drop -> positive alpha, WIN.
    * Bearish stance + price gain -> negative alpha, LOSS.
  - Test hit rate calculation with friction threshold ($\pm 0.20\%$).
  - Test publisher credibility scorecard aggregation and tier assignment.
  - Test updating forward returns via mock price fetcher.
- [ ] **Step 2: Implement `PerformanceTrackerEngine` in `news-agent/src/analysis/performance.py`**
  - Implement `compute_directional_alpha(stance, initial_price, current_price)`.
  - Implement `classify_outcome(stance, return_pct)`.
  - Implement `update_forward_returns(storage, price_fetcher, min_age_hours_1d=24, min_age_hours_7d=168)`.
  - Implement `calculate_performance_metrics(storage, min_opportunity=6.0, ticker=None, source=None)`.
- [ ] **Step 3: Export in `news-agent/src/analysis/__init__.py`**
- [ ] **Step 4: Run tests and verify 100% pass rate**
  - `pytest news-agent/tests/test_performance.py -v`
- [ ] **Step 5: Commit**
  - `git commit -m "feat(analysis): implement directional alpha math, hit rate, and publisher credibility engine"`

---

### Task 3: Pipeline Lineage Stance Persistence & Live Price Integration

**Files:**
- Modify: `news-agent/src/pipeline.py`
- Test: `news-agent/tests/test_pipeline.py`

- [ ] **Step 1: Write test in `news-agent/tests/test_pipeline.py`**
  - Verify that `EventLineageRecord` stored in SQLite contains `directional_stance` from `analysis.directional_stance`.
- [ ] **Step 2: Update `news-agent/src/pipeline.py`**
  - Pass `directional_stance=analysis.directional_stance.value if hasattr(analysis.directional_stance, "value") else str(analysis.directional_stance)` into `EventLineageRecord`.
- [ ] **Step 3: Run pipeline tests**
  - `pytest news-agent/tests/test_pipeline.py -v`
- [ ] **Step 4: Commit**
  - `git commit -m "feat(pipeline): persist directional stance in event lineage record"`

---

### Task 4: CLI Command `performance` & Dashboard

**Files:**
- Modify: `news-agent/src/cli.py`
- Test: `news-agent/tests/test_cli.py`

- [ ] **Step 1: Write CLI tests in `news-agent/tests/test_cli.py`**
  - Test `performance` command with empty database.
  - Test `performance` command with populated return records (renders Global Metrics & Publisher Scorecard).
  - Test `performance --update` flag.
- [ ] **Step 2: Implement `performance` command in `news-agent/src/cli.py`**
  - Options: `--update/--no-update`, `--ticker`, `--source`, `--min-opportunity`.
  - Renders:
    1. Rich summary panel for Global Hit Rate and Alpha (1-day & 7-day).
    2. Rich table for Publisher Credibility Scorecard (Source, Events, Avg Quality, 1d Hit Rate, Avg Alpha, Tier).
    3. Rich table for Catalyst Realized Return Ledger.
- [ ] **Step 3: Run CLI tests and full test suite**
  - `pytest news-agent/tests/test_cli.py -v`
  - `pytest news-agent/tests/ -v`
- [ ] **Step 4: Commit**
  - `git commit -m "feat(cli): add performance command for measurable alpha tracking and publisher credibility"`

---

### Task 5: QA Verification, Git Push & Cloud Deployment

**Files:**
- Run full test suite: `pytest news-agent/tests/ -v`
- Push to GitHub: `git push origin main`
- Deploy to Google Cloud Functions: `firebase deploy --only functions`
- Summary report to user.
