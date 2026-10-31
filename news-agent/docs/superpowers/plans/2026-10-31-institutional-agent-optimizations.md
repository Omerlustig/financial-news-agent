# Institutional Financial News Agent Optimizations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement 6 institutional optimization engines inspired by leading open-source quant and agent architectures into `news-agent`: Ingestion Noise/Ticker Pre-filtering, Hallucination-Proof Integer-Index Triage, Quantitative Market Data Grounding, Adversarial Dialectic Debate, Citation & Factual Grounding Verification, and Signal Lifecycle/Ticker Quota Throttling.

**Architecture:** Each optimization is built as a modular, decoupled engine with isolated interfaces in `src/ingestion/` and `src/analysis/`, configured via `src/config.py` with feature flags, and coordinated in `src/pipeline.py` preserving 100% backward compatibility with existing tests.

**Tech Stack:** Python 3.11+, Pydantic v2, pytest, LiteLLM / Gemini / OpenAI, requests / urllib3.

**Spec:** `docs/superpowers/specs/2026-10-31-institutional-agent-optimizations-design.md`

## Global Constraints
- Python 3.11+ compatibility.
- Zero network dependency in unit tests (all network, external APIs, and LLM calls must be fully mocked or inject mock drivers).
- Every engine must be individually toggleable via `src/config.py` settings.
- All 431 existing pytest tests must continue to pass with zero regressions.
- TDD required: write failing tests first, verify failure, implement minimal code, verify passing, and commit per task.

## Review Focus
1. Non-tradeable false acronyms (e.g. `CEO`, `AI`, `US`, `FED`) must be strictly stripped from detected tickers.
2. Triage LLM returning out-of-range integer indices (e.g. index 99 when batch size is 5) must be safely discarded with zero runtime exceptions.
3. Market data network timeouts or rate limits must fail gracefully to neutral context without blocking news analysis.
4. Dialectic debate must only trigger for high-conviction events (`score >= 8` or `HIGH`/`CRITICAL`) to avoid token bloat on routine news.
5. Citation verifier must handle punctuation, currency symbols, and percentage normalization without false positive hallucination warnings.

---

### Task 1: Ingestion Noise Pre-Filter & Ticker Blacklist

**Files:**
- Create: `src/ingestion/prefilter.py`
- Test: `tests/test_prefilter.py`

**Interfaces:**
- Consumes: `RawArticle` from `src.models`.
- Produces: `NewsPreFilter` class with `is_noise(article: RawArticle) -> bool` and `filter_tickers(tickers: List[str]) -> List[str]`.

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_prefilter.py` covering:
- Dropping sponsored / clickbait articles ("3 stocks to buy now", "Zacks investment ideas", "partner content").
- Allowing genuine market catalyst articles ("Nvidia unveils new Rubin GPU architecture").
- Filtering out blacklisted ticker acronyms (`CEO`, `AI`, `US`, `FED`, `GDP`, `SEC`, `FDA`, `IPO`, `ETF`, `EV`, `CAGR`, `EPS`).
- Keeping legitimate stock tickers (`NVDA`, `AAPL`, `MSFT`, `LLY`).

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_prefilter.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.ingestion.prefilter'`

- [ ] **Step 3: Implement minimal code in `src/ingestion/prefilter.py`**
Implement `NewsPreFilter` and `PreFilterConfig` with regex matching and set filtering.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_prefilter.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/ingestion/prefilter.py tests/test_prefilter.py && git commit -m "feat(ingestion): add noise pre-filter and ticker blacklist engine"`

---

### Task 2: Hallucination-Proof Integer-Index Triage Engine

**Files:**
- Create: `src/analysis/integer_triage.py`
- Test: `tests/test_integer_triage.py`

**Interfaces:**
- Consumes: List of `RawArticle`, optional LLM client or completion function.
- Produces: `IntegerIndexTriageEngine` with `triage_batch(articles: List[RawArticle]) -> List[RawArticle]`.

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_integer_triage.py`:
- Batching articles into numbered prompt format: `[0] Ticker: Title\n[1] ...`.
- Parsing LLM integer array response (e.g. `"[0, 2]"` or JSON `{"selected": [0, 2]}`).
- Discarding out-of-range indices (e.g. index `99` on a 3-article batch).
- Handling malformed/non-JSON LLM response by falling back gracefully to heuristic triage.
- Returning the filtered subset of `RawArticle` objects matching selected indices.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_integer_triage.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.integer_triage'`

- [ ] **Step 3: Implement minimal code in `src/analysis/integer_triage.py`**
Implement `IntegerIndexTriageEngine` with robust integer extraction regex fallback, range checking, and configurable batch sizing.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_integer_triage.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/analysis/integer_triage.py tests/test_integer_triage.py && git commit -m "feat(analysis): add integer-index hallucination-proof triage engine"`

---

### Task 3: TradingView Quantitative Market Data Engine

**Files:**
- Create: `src/analysis/market_data.py`
- Test: `tests/test_market_data.py`

**Interfaces:**
- Consumes: Ticker symbols (e.g. `AAPL`, `NVDA`).
- Produces: `TradingViewMarketDataEngine` with `get_ticker_context(ticker: str) -> Optional[MarketContext]` and `format_prompt_context(tickers: List[str]) -> str`.

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_market_data.py`:
- Mocking screener API response for a ticker (`close`, `change`, `volume`, `avg_vol_10d`, `RSI`, `SMA50`, `SMA200`, `short_percent`).
- Computing Relative Volume (RVOL) = `volume / avg_vol`.
- Formatting the human-readable market context string for LLM injection.
- Handling network timeouts / non-existent tickers by returning `None` or neutral prompt without raising uncaught exceptions.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_market_data.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.market_data'`

- [ ] **Step 3: Implement minimal code in `src/analysis/market_data.py`**
Implement `MarketContext` dataclass and `TradingViewMarketDataEngine` using REST query / mock adapter with timeout protection and caching.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_market_data.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/analysis/market_data.py tests/test_market_data.py && git commit -m "feat(analysis): add TradingView quantitative market data engine"`

---

### Task 4: Adversarial Dialectic Debate Engine

**Files:**
- Create: `src/analysis/debate.py`
- Test: `tests/test_debate.py`

**Interfaces:**
- Consumes: `MarketImpactAnalysis`, `RawArticle`, optional `MarketContext`.
- Produces: `DialecticDebateEngine` with `conduct_debate(analysis: MarketImpactAnalysis, article: RawArticle, context: Optional[str] = None) -> DebateVerdict`.

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_debate.py`:
- Trigger condition check: should only trigger when `confidence >= 8` or `grade >= HIGH`.
- Executing Bull Analyst thesis generation vs. Risk Skeptic (Bear) counter-thesis generation.
- Generating `DebateVerdict` with:
  - `bull_thesis`: key catalysts and upside mechanism.
  - `bear_counter_thesis`: priced-in risks, valuation/regulatory traps.
  - `net_conviction_score`: adjusted 1–10 score.
  - `invalidation_triggers`: explicit conditions that bust the thesis.
- Fallback handling if LLM call fails (preserves original analysis without crashing).

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_debate.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.debate'`

- [ ] **Step 3: Implement minimal code in `src/analysis/debate.py`**
Implement `DebateVerdict` and `DialecticDebateEngine` with structured prompting, score synthesis, and invalidation trigger extraction.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_debate.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/analysis/debate.py tests/test_debate.py && git commit -m "feat(analysis): add dialectic bull vs bear debate engine"`

---

### Task 5: Citation & Evidence Grounding Verifier

**Files:**
- Create: `src/analysis/citation.py`
- Test: `tests/test_citation.py`

**Interfaces:**
- Consumes: `catalyst_thesis: str`, `article_body: str`, optional repair callback.
- Produces: `CitationGroundingEngine` with `verify_grounding(thesis: str, article_body: str) -> CitationResult`.

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_citation.py`:
- Extracting numerical figures (e.g. `"$500 million"`, `"45%"`, `"$1.20 per share"`).
- Exact and normalized matching against the full article text.
- Detecting ungrounded/fabricated statistics (e.g. article mentions "$200M" but thesis claims "$900M").
- Invoking `auto_repair` callback when ungrounded claims are flagged.
- Producing `CitationResult` with `is_grounded: bool`, `unsupported_claims: List[str]`, and `evidence_snippets: List[str]`.

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_citation.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.citation'`

- [ ] **Step 3: Implement minimal code in `src/analysis/citation.py`**
Implement `CitationGroundingEngine` using numerical/entity regex extractors and fuzzy sentence window matching.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_citation.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/analysis/citation.py tests/test_citation.py && git commit -m "feat(analysis): add factual citation and evidence verifier"`

---

### Task 6: Signal Lifecycle Evolution & Ticker Fair-Share Quota

**Files:**
- Create: `src/analysis/lifecycle.py`
- Test: `tests/test_lifecycle_quota.py`

**Interfaces:**
- Consumes: `ticker: str`, `impact_grade: ImpactGrade`, `db_conn` or in-memory tracker.
- Produces: `SignalLifecycleEngine` and `TickerQuotaTracker` with:
  - `check_quota(ticker: str, grade: ImpactGrade) -> Tuple[bool, str]`
  - `record_alert(ticker: str, grade: ImpactGrade)`
  - `evaluate_signal_state(current_analysis: MarketImpactAnalysis, prior_analyses: List[MarketImpactAnalysis]) -> SignalEvolutionState`

- [ ] **Step 1: Write the failing tests**
Write tests in `tests/test_lifecycle_quota.py`:
- Allowing first and second alert for a ticker within the 6-hour rolling window.
- Blocking 3rd non-critical alert with reason `"TICKER_QUOTA_EXCEEDED"`.
- Bypassing quota limit when impact grade is `ImpactGrade.CRITICAL` (critical catalysts must never be suppressed).
- Transitioning signal states: `NEW` $\rightarrow$ `CORROBORATED` $\rightarrow$ `STRENGTHENED` (same stance reinforced) vs `FALSIFIED` (stance reversed).

- [ ] **Step 2: Run test to verify it fails**
Run: `pytest tests/test_lifecycle_quota.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.lifecycle'`

- [ ] **Step 3: Implement minimal code in `src/analysis/lifecycle.py`**
Implement `SignalEvolutionState` enum, `TickerQuotaTracker`, and `SignalLifecycleEngine`.

- [ ] **Step 4: Run test to verify it passes**
Run: `pytest tests/test_lifecycle_quota.py -v`
Expected: PASS

- [ ] **Step 5: Commit**
`git add src/analysis/lifecycle.py tests/test_lifecycle_quota.py && git commit -m "feat(analysis): add signal lifecycle tracker and ticker quota limiter"`

---

### Task 7: Config Settings, Pipeline Integration & Full Verification

**Files:**
- Modify: `src/config.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_optimizations.py`

**Interfaces:**
- Consumes: All 6 new engines.
- Produces: Updated `Pipeline` coordinating pre-filtering, integer triage, market data injection, dialectic debate, citation checking, and quota gating.

- [ ] **Step 1: Update `src/config.py` with feature flags**
Add `PreFilterSettings`, `MarketDataSettings`, `DebateSettings`, `CitationSettings`, and `QuotaSettings` models under `Settings`.

- [ ] **Step 2: Write pipeline integration tests in `tests/test_pipeline_optimizations.py`**
Verify that `Pipeline.process_cycle()` seamlessly connects:
- Pre-filtering dropping clickbait before deduplication.
- Integer-index triage filtering candidates.
- Market data injected into analysis context.
- Debate engine refining high-conviction catalysts.
- Quota limiter throttling repetitive ticker alerts.

- [ ] **Step 3: Wire new engines into `src/pipeline.py`**
Inject engines into `Pipeline.__init__` with backward-compatible defaults, updating the execution flow.

- [ ] **Step 4: Run integration tests**
Run: `pytest tests/test_pipeline_optimizations.py -v`
Expected: PASS

- [ ] **Step 5: Run full regression test suite**
Run: `pytest -v`
Expected: ALL 431+ tests pass with zero regressions.

- [ ] **Step 6: Commit**
`git add src/config.py src/pipeline.py tests/test_pipeline_optimizations.py && git commit -m "feat(pipeline): integrate institutional optimization engines and update settings"`
