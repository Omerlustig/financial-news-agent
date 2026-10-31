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

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/ingestion/prefilter.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 2: Hallucination-Proof Integer-Index Triage Engine

**Files:**
- Create: `src/analysis/integer_triage.py`
- Test: `tests/test_integer_triage.py`

**Interfaces:**
- Consumes: List of `RawArticle`, optional LLM client or completion function.
- Produces: `IntegerIndexTriageEngine` with `triage_batch(articles: List[RawArticle]) -> List[RawArticle]`.

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/analysis/integer_triage.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 3: TradingView Quantitative Market Data Engine

**Files:**
- Create: `src/analysis/market_data.py`
- Test: `tests/test_market_data.py`

**Interfaces:**
- Consumes: Ticker symbols (e.g. `AAPL`, `NVDA`).
- Produces: `TradingViewMarketDataEngine` with `get_ticker_context(ticker: str) -> Optional[MarketContext]` and `format_prompt_context(tickers: List[str]) -> str`.

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/analysis/market_data.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 4: Adversarial Dialectic Debate Engine

**Files:**
- Create: `src/analysis/debate.py`
- Test: `tests/test_debate.py`

**Interfaces:**
- Consumes: `MarketImpactAnalysis`, `RawArticle`, optional `MarketContext`.
- Produces: `DialecticDebateEngine` with `conduct_debate(analysis: MarketImpactAnalysis, article: RawArticle, context: Optional[str] = None) -> DebateVerdict`.

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/analysis/debate.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 5: Citation & Evidence Grounding Verifier

**Files:**
- Create: `src/analysis/citation.py`
- Test: `tests/test_citation.py`

**Interfaces:**
- Consumes: `catalyst_thesis: str`, `article_body: str`, optional repair callback.
- Produces: `CitationGroundingEngine` with `verify_grounding(thesis: str, article_body: str) -> CitationResult`.

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/analysis/citation.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 6: Signal Lifecycle Evolution & Ticker Fair-Share Quota

**Files:**
- Create: `src/analysis/lifecycle.py`
- Test: `tests/test_lifecycle_quota.py`

**Interfaces:**
- Consumes: `ticker: str`, `impact_grade: ImpactGrade`, `db_conn` or in-memory tracker.
- Produces: `SignalLifecycleEngine` and `TickerQuotaTracker` with:
  - `check_quota(ticker: str, impact_grade: ImpactGrade) -> Tuple[bool, str]`
  - `record_alert(ticker: str, impact_grade: ImpactGrade)`
  - `evaluate_signal_state(current_analysis: MarketImpactAnalysis, prior_analyses: List[MarketImpactAnalysis]) -> SignalEvolutionState`

- [x] **Step 1: Write the failing tests**
- [x] **Step 2: Run test to verify it fails**
- [x] **Step 3: Implement minimal code in `src/analysis/lifecycle.py`**
- [x] **Step 4: Run test to verify it passes**
- [x] **Step 5: Commit**

---

### Task 7: Config Settings, Pipeline Integration & Full Verification

**Files:**
- Modify: `src/config.py`
- Modify: `src/pipeline.py`
- Test: `tests/test_pipeline_optimizations.py`

**Interfaces:**
- Consumes: All 6 new engines.
- Produces: Updated `Pipeline` coordinating pre-filtering, integer triage, market data injection, dialectic debate, citation checking, and quota gating.

- [x] **Step 1: Update `src/config.py` with feature flags**
- [x] **Step 2: Write pipeline integration tests in `tests/test_pipeline_optimizations.py`**
- [x] **Step 3: Wire new engines into `src/pipeline.py`**
- [x] **Step 4: Run integration tests**
- [x] **Step 5: Run full regression test suite** (454 passed!)
- [x] **Step 6: Commit**
