# Market News Analysis & Telegram Alert Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an automated periodic news agent in `news-agent/` that ingests financial headlines, analyzes them with institutional stock market reasoning and confidence scoring (1–10 / Low–Critical), prevents duplicates with SQLite, and sends instant push alerts via Telegram.

**Architecture:** A modular pipeline orchestrated under `news-agent/` with clean boundaries: pluggable Ingestion (RSS/Atom), SQLite Deduplication/Storage (`WAL` mode), an LLM Market Impact Analyzer using LiteLLM with structured Pydantic schema, a formatted Telegram Notifier, and a dual-mode CLI supporting both continuous background daemon loops and one-shot cron/serverless executions.

**Tech Stack:** Python 3.11+, Pydantic v2, LiteLLM, feedparser, requests, python-telegram-bot / direct Telegram Bot API, typer/click, SQLite3, pytest, pytest-asyncio, Docker.

**Spec:** [docs/superpowers/specs/2026-10-17-market-news-agent-design.md](file:///Users/omer.lustig/IdeaProjects/omer/docs/superpowers/specs/2026-10-17-market-news-agent-design.md)

## Global Constraints

- All source code, configs, tests, and deployment files must reside under `news-agent/`.
- Zero duplicate alerts: Deduplication must be backed by SQLite using SHA-256 content hashes (`url + title`).
- Scoring rubric must be strictly calibrated from 1 to 10 with matching grades (`LOW`, `AVERAGE`, `HIGH`, `CRITICAL`).
- Ingestion must be resilient: failure of one feed or website must not crash or halt the pipeline.
- Dual execution modes: `run-once` for cron/serverless and `run --daemon --interval <min>` for persistent execution.

## Review Focus

- Malformed or unreachable RSS feed: Returns graceful empty list with warning log, other feeds succeed.
- LLM output parsing failure or rate limit: Retries with backoff; un-analyzed articles remain queued for next cycle without losing state.
- Special characters in Telegram Markdown: Markdown formatting properly escapes or handles symbols (`_`, `*`, `[`, `]`, `(`, `)`) to prevent Telegram API 400 Bad Request errors.
- SQLite concurrent access or crash: SQLite WAL mode enabled to ensure database integrity during unexpected termination.
- Zero-catalyst news cycle: If all headlines are routine noise (score < threshold), agent silently logs completion and dispatches zero Telegram notifications.

---

### Task 1: Project Setup, Data Models, and Configuration Subsystem

**Role Perspectives**: Expert Software Architect & Developer

**Files:**
- Create: `news-agent/requirements.txt`
- Create: `news-agent/requirements-dev.txt`
- Create: `news-agent/.env.example`
- Create: `news-agent/config/settings.yaml`
- Create: `news-agent/config/sources.yaml`
- Create: `news-agent/src/__init__.py`
- Create: `news-agent/src/models.py`
- Create: `news-agent/src/config.py`
- Create: `news-agent/tests/__init__.py`
- Create: `news-agent/tests/test_config.py`

**Interfaces:**
- Consumes: None (Root foundation)
- Produces:
  - `RawArticle(title: str, url: str, source_name: str, published_at: Optional[datetime], summary: str, content: Optional[str])` with `.content_hash -> str`
  - `ImpactGrade(str, Enum)` with values `LOW`, `AVERAGE`, `HIGH`, `CRITICAL`
  - `MarketImpactAnalysis(is_market_moving: bool, confidence_score: int, impact_grade: ImpactGrade, affected_tickers: List[str], affected_sectors: List[str], summary: str, catalyst_thesis: str)`
  - `load_settings(config_path: Optional[str]) -> Settings`
  - `load_sources(sources_path: Optional[str]) -> List[SourceConfig]`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_config.py`**

```python
from news-agent.src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from news-agent.src.config import load_settings, load_sources

def test_raw_article_hash():
    a1 = RawArticle(title="Chip export curbs", url="https://news.com/1", source_name="Reuters", published_at=None, summary="Test")
    a2 = RawArticle(title="Chip export curbs", url="https://news.com/1", source_name="Reuters", published_at=None, summary="Different summary")
    assert a1.content_hash == a2.content_hash

def test_load_default_sources(tmp_path):
    sources_yaml = tmp_path / "sources.yaml"
    sources_yaml.write_text("""
sources:
  - name: "Reuters Business"
    type: "rss"
    url: "https://feeds.reuters.com/reuters/businessNews"
    enabled: true
""")
    sources = load_sources(str(sources_yaml))
    assert len(sources) == 1
    assert sources[0].name == "Reuters Business"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_config.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement dependencies, models, and configuration loaders**

1. Create `news-agent/requirements.txt` with `pydantic>=2.0`, `pyyaml>=6.0`, `litellm>=1.30.0`, `feedparser>=6.0.10`, `requests>=2.31.0`, `typer>=0.9.0`, `python-dotenv>=1.0.0`.
2. Create `news-agent/requirements-dev.txt` with `pytest>=8.0.0`, `pytest-asyncio>=0.23.0`.
3. Implement `RawArticle`, `ImpactGrade`, and `MarketImpactAnalysis` in `news-agent/src/models.py`.
4. Implement `load_settings` and `load_sources` in `news-agent/src/config.py` using Pydantic and PyYAML.
5. Create default `config/settings.yaml` and `config/sources.yaml` with top financial news feeds.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/
git commit -m "feat(core): setup project models, config loader and sources"
```

---

### Task 2: News Ingestion Engine (RSS/Atom & Scraper Normalization)

**Role Perspectives**: Expert Developer & QA Engineer

**Files:**
- Create: `news-agent/src/ingestion/__init__.py`
- Create: `news-agent/src/ingestion/base.py`
- Create: `news-agent/src/ingestion/rss.py`
- Create: `news-agent/tests/test_ingestion.py`

**Interfaces:**
- Consumes: `RawArticle`, `SourceConfig` from Task 1
- Produces:
  - `BaseFetcher.fetch(self) -> List[RawArticle]`
  - `RSSFetcher(source_config: SourceConfig).fetch() -> List[RawArticle]`
  - `fetch_all_sources(sources: List[SourceConfig]) -> List[RawArticle]`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_ingestion.py`**

```python
from unittest.mock import patch
from news-agent.src.config import SourceConfig
from news-agent.src.ingestion.rss import RSSFetcher, fetch_all_sources

MOCK_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Financial Feed</title>
    <item>
      <title>Fed Signals Pause on Rates</title>
      <link>https://example.com/fed-pause</link>
      <description>The Federal Reserve announced...</description>
      <pubDate>Sat, 17 Oct 2026 12:00:00 GMT</pubDate>
    </item>
  </channel>
</rss>"""

def test_rss_fetcher_parses_feed():
    src = SourceConfig(name="Mock Feed", type="rss", url="https://example.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        # Mock feedparser response
        mock_entry = {
            "title": "Fed Signals Pause on Rates",
            "link": "https://example.com/fed-pause",
            "summary": "The Federal Reserve announced...",
            "published": "Sat, 17 Oct 2026 12:00:00 GMT"
        }
        mock_parse.return_value = {"entries": [mock_entry], "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()
        assert len(articles) == 1
        assert articles[0].title == "Fed Signals Pause on Rates"
        assert articles[0].source_name == "Mock Feed"

def test_rss_fetcher_handles_feed_failure_gracefully():
    src = SourceConfig(name="Broken Feed", type="rss", url="https://broken.com/rss", enabled=True)
    with patch("feedparser.parse", side_effect=Exception("Connection timeout")):
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()
        assert articles == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_ingestion.py -v`
Expected: FAIL (ImportError / ModuleNotFoundError)

- [ ] **Step 3: Implement `BaseFetcher` and `RSSFetcher`**

1. Define `BaseFetcher` abstract class with `fetch() -> List[RawArticle]`.
2. Implement `RSSFetcher` in `news-agent/src/ingestion/rss.py` using `feedparser`, normalizing publication dates to UTC datetime, stripping HTML tags from summaries, and wrapping network errors in try/except blocks.
3. Implement `fetch_all_sources(sources: List[SourceConfig]) -> List[RawArticle]` iterating through enabled sources with feed-level error containment.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_ingestion.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/ingestion/ news-agent/tests/test_ingestion.py
git commit -m "feat(ingestion): implement robust RSS/Atom news fetcher with error isolation"
```

---

### Task 3: SQLite Storage & Deduplication Subsystem

**Role Perspectives**: Expert Software Architect & QA Engineer

**Files:**
- Create: `news-agent/src/storage/__init__.py`
- Create: `news-agent/src/storage/base.py`
- Create: `news-agent/src/storage/sqlite.py`
- Create: `news-agent/tests/test_storage.py`

**Interfaces:**
- Consumes: `RawArticle`, `MarketImpactAnalysis` from Task 1
- Produces:
  - `BaseStorage.is_article_seen(content_hash: str) -> bool`
  - `BaseStorage.filter_unseen(articles: List[RawArticle]) -> List[RawArticle]`
  - `BaseStorage.record_article(article: RawArticle, analysis: Optional[MarketImpactAnalysis], alerted: bool) -> None`
  - `BaseStorage.get_stats() -> dict`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_storage.py`**

```python
from news-agent.src.storage.sqlite import SQLiteStorage
from news-agent.src.models import RawArticle, MarketImpactAnalysis, ImpactGrade

def test_deduplication_and_unseen_filter(tmp_path):
    db_path = str(tmp_path / "test.db")
    storage = SQLiteStorage(db_path)
    
    a1 = RawArticle(title="Article 1", url="https://news.com/1", source_name="Reuters", published_at=None, summary="Test")
    a2 = RawArticle(title="Article 2", url="https://news.com/2", source_name="CNBC", published_at=None, summary="Test")
    
    assert storage.is_article_seen(a1.content_hash) is False
    
    analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=3,
        impact_grade=ImpactGrade.LOW,
        affected_tickers=[],
        affected_sectors=[],
        summary="Low impact",
        catalyst_thesis="Routine"
    )
    storage.record_article(a1, analysis, alerted=False)
    
    assert storage.is_article_seen(a1.content_hash) is True
    unseen = storage.filter_unseen([a1, a2])
    assert len(unseen) == 1
    assert unseen[0].title == "Article 2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_storage.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `SQLiteStorage` with WAL mode and schema migration**

1. Create `news-agent/src/storage/sqlite.py` initializing schema:
   * `articles` table: `content_hash (PK)`, `url`, `title`, `source_name`, `published_at`, `created_at`, `is_market_moving`, `confidence_score`, `impact_grade`, `tickers`, `sectors`, `alerted`.
2. Enable SQLite WAL mode (`PRAGMA journal_mode=WAL;`).
3. Implement `filter_unseen` using batch queries (`SELECT content_hash FROM articles WHERE content_hash IN (...)`).
4. Implement `record_article` and `get_stats`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_storage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/storage/ news-agent/tests/test_storage.py
git commit -m "feat(storage): implement SQLite deduplication engine with WAL mode"
```

---

### Task 4: Financial Intelligence & Market Impact Analyzer

**Role Perspectives**: Expert Financial Advisor, Stock Analyst & Developer

**Files:**
- Create: `news-agent/src/analysis/__init__.py`
- Create: `news-agent/src/analysis/prompts.py`
- Create: `news-agent/src/analysis/analyzer.py`
- Create: `news-agent/tests/test_analysis.py`

**Interfaces:**
- Consumes: `RawArticle`, `MarketImpactAnalysis` from Task 1, LLM settings from `config.py`
- Produces:
  - `build_analyst_prompt(article: RawArticle) -> Tuple[str, str]`
  - `MarketImpactAnalyzer.analyze(article: RawArticle) -> MarketImpactAnalysis`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_analysis.py`**

```python
from unittest.mock import patch
from news-agent.src.models import RawArticle, ImpactGrade
from news-agent.src.analysis.analyzer import MarketImpactAnalyzer

MOCK_LLM_JSON = """{
  "is_market_moving": true,
  "confidence_score": 9,
  "impact_grade": "CRITICAL",
  "affected_tickers": ["$NVDA", "$AMD"],
  "affected_sectors": ["Semiconductors"],
  "summary": "U.S. widens restrictions on advanced AI chip exports.",
  "catalyst_thesis": "Immediate high-margin revenue impairment in foreign markets."
}"""

def test_analyzer_parses_llm_response():
    analyzer = MarketImpactAnalyzer(model="gemini/gemini-1.5-flash")
    article = RawArticle(
        title="U.S. expands export bans on AI chips",
        url="https://reuters.com/1",
        source_name="Reuters",
        published_at=None,
        summary="New restrictions announced by Commerce Dept."
    )
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": MOCK_LLM_JSON}}]
        }
        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert result.confidence_score == 9
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert "$NVDA" in result.affected_tickers
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_analysis.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Financial Prompt Engineering & `MarketImpactAnalyzer`**

1. In `news-agent/src/analysis/prompts.py`, write institutional financial analyst prompt:
   * Rule-sets distinguishing 95% market noise from 5% genuine catalysts (unexpected M&A, regulatory shock, sudden rate pivots, surprise earnings with guidance cuts).
   * 1st-order and 2nd-order impact criteria.
   * Strict calibration rubric for 1-10 confidence scoring and `LOW`/`AVERAGE`/`HIGH`/`CRITICAL` grading.
   * Enforced JSON schema format.
2. In `news-agent/src/analysis/analyzer.py`, call `litellm.completion` with JSON response format, retry backoff (3 attempts), and fallback handling for malformed JSON.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_analysis.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/analysis/ news-agent/tests/test_analysis.py
git commit -m "feat(analysis): implement financial analyst LLM impact scoring and catalyst engine"
```

---

### Task 5: Telegram Notifier Subsystem

**Role Perspectives**: Expert Developer & QA Engineer

**Files:**
- Create: `news-agent/src/notification/__init__.py`
- Create: `news-agent/src/notification/base.py`
- Create: `news-agent/src/notification/telegram.py`
- Create: `news-agent/tests/test_notification.py`

**Interfaces:**
- Consumes: `RawArticle`, `MarketImpactAnalysis` from Task 1
- Produces:
  - `BaseNotifier.send_alert(article: RawArticle, analysis: MarketImpactAnalysis) -> bool`
  - `TelegramNotifier(bot_token: str, chat_id: str).send_alert(article, analysis) -> bool`
  - `format_telegram_message(article: RawArticle, analysis: MarketImpactAnalysis) -> str`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_notification.py`**

```python
from unittest.mock import patch
from news-agent.src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from news-agent.src.notification.telegram import TelegramNotifier, format_telegram_message

def test_telegram_message_formatting():
    article = RawArticle(title="Fed cuts rates by 50bps", url="https://cnbc.com/rates", source_name="CNBC", published_at=None, summary="Fed surprise move")
    analysis = MarketImpactAnalysis(
        is_market_moving=True,
        confidence_score=9,
        impact_grade=ImpactGrade.CRITICAL,
        affected_tickers=["$SPY", "$QQQ"],
        affected_sectors=["Financials", "Real Estate"],
        summary="The Federal Reserve implemented an emergency 50bps rate cut.",
        catalyst_thesis="Lower borrowing costs will immediately re-rate equity multiples."
    )
    msg = format_telegram_message(article, analysis)
    assert "[CRITICAL | Score: 9/10]" in msg
    assert "$SPY" in msg
    assert "https://cnbc.com/rates" in msg

def test_telegram_send_alert_success():
    notifier = TelegramNotifier(bot_token="fake_token", chat_id="12345")
    with patch("requests.post") as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}
        success = notifier.send_message("Test message")
        assert success is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_notification.py -v`
Expected: FAIL

- [ ] **Step 3: Implement Telegram message formatting and HTTP dispatcher**

1. Create `news-agent/src/notification/telegram.py`.
2. Implement clean markdown formatting highlighting Grade Badge (`CRITICAL`, `HIGH`, etc.), Score, Impacted Tickers, Sectors, Summary, Analyst Thesis, and URL.
3. Implement `send_message` with requests POST to `https://api.telegram.org/bot<token>/sendMessage`, handling network errors, HTTP 429 throttling, and logging failures.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_notification.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/notification/ news-agent/tests/test_notification.py
git commit -m "feat(notification): implement Telegram alert formatter and push dispatcher"
```

---

### Task 6: Pipeline Orchestration & Dual-Mode CLI

**Role Perspectives**: Expert Software Architect, Developer & QA Engineer

**Files:**
- Create: `news-agent/src/pipeline.py`
- Create: `news-agent/src/cli.py`
- Create: `news-agent/src/main.py`
- Create: `news-agent/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Ingestion (Task 2), Storage (Task 3), Analyzer (Task 4), Notifier (Task 5)
- Produces:
  - `Pipeline.run_cycle() -> CycleSummary`
  - CLI commands: `run`, `run-once`, `test-telegram`, `test-source`, `stats`

- [ ] **Step 1: Write the failing tests in `news-agent/tests/test_pipeline.py`**

```python
from unittest.mock import MagicMock
from news-agent.src.pipeline import Pipeline
from news-agent.src.models import RawArticle, MarketImpactAnalysis, ImpactGrade

def test_pipeline_filters_below_threshold():
    mock_storage = MagicMock()
    mock_analyzer = MagicMock()
    mock_notifier = MagicMock()
    
    # Configure article and low-impact analysis
    art = RawArticle(title="Routine CEO Interview", url="https://test.com/1", source_name="Test", published_at=None, summary="Routine")
    mock_storage.filter_unseen.return_value = [art]
    mock_analyzer.analyze.return_value = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=4,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=[],
        affected_sectors=[],
        summary="Routine interview",
        catalyst_thesis="No catalyst"
    )
    
    pipeline = Pipeline(storage=mock_storage, analyzer=mock_analyzer, notifier=mock_notifier, min_score=6)
    summary = pipeline.process_articles([art])
    
    assert summary["seen"] == 1
    assert summary["alerted"] == 0
    mock_notifier.send_alert.assert_not_called()
    mock_storage.record_article.assert_called_once_with(art, mock_analyzer.analyze.return_value, alerted=False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest news-agent/tests/test_pipeline.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `Pipeline` and Typer CLI**

1. Implement `Pipeline` class in `news-agent/src/pipeline.py`:
   * `run_cycle()`: Fetches sources, filters unseen via SQLite, analyzes candidates with LLM, filters by `min_confidence_score` and `min_impact_grade`, dispatches Telegram alerts, and records results in SQLite.
2. Implement `cli.py` using `typer`:
   * `run-once`: Executes a single `pipeline.run_cycle()` and prints summary statistics.
   * `run --daemon --interval <minutes>`: Continuous `while True` loop with `time.sleep()`, handling `SIGINT`/`SIGTERM` gracefully.
   * `test-telegram`: Sends an instant test alert to Telegram.
   * `test-source --name <source_name>`: Fetches and displays parsed headlines from a single feed.
   * `stats`: Displays total seen articles, analyzed items, and triggered alerts.
3. Implement `main.py` as entry point: `if __name__ == "__main__": app()`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest news-agent/tests/test_pipeline.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/pipeline.py news-agent/src/cli.py news-agent/src/main.py news-agent/tests/test_pipeline.py
git commit -m "feat(pipeline): implement orchestration engine and dual-mode CLI"
```

---

### Task 7: Containerization, Deployment Scripts & Documentation

**Role Perspectives**: Expert Software Architect & Developer

**Files:**
- Create: `news-agent/Dockerfile`
- Create: `news-agent/docker-compose.yml`
- Create: `news-agent/news-agent.service` (Systemd unit for EC2)
- Create: `news-agent/README.md`

**Interfaces:**
- Consumes: Entire `news-agent` system
- Produces: Ready-to-deploy Docker image, compose file with volume persistence, systemd unit, and complete setup guide.

- [ ] **Step 1: Create multi-stage `Dockerfile` and `docker-compose.yml`**

1. Python 3.11-slim base image, non-root user, SQLite persistent volume mount (`/app/data`).
2. Compose file with auto-restart policy and `.env` loading.

- [ ] **Step 2: Create EC2 systemd service template (`news-agent.service`)**

Systemd unit configured with `Restart=always`, `RestartSec=30`, and environment file loading.

- [ ] **Step 3: Create comprehensive `README.md`**

Full guide for:
1. Quickstart local run.
2. Telegram Bot creation & obtaining Chat ID.
3. LLM API configuration (Gemini / OpenAI / Claude).
4. Running as a background daemon vs cron / AWS Lambda.
5. Customizing `sources.yaml` and scoring thresholds.

- [ ] **Step 4: Run full test suite across the entire project**

Run: `pytest news-agent/tests/ -v`
Expected: 100% tests PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/Dockerfile news-agent/docker-compose.yml news-agent/news-agent.service news-agent/README.md
git commit -m "docs(deploy): add Docker container, systemd service and operations guide"
```
