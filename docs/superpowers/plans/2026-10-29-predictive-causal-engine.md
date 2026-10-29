# Predictive Causal Intelligence & Second-Order Opportunity Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transform the news agent into an anticipatory market intelligence system that predicts unpriced stock and sector opportunities from real-world shocks (geopolitics, flight disruptions, defense, regulatory actions) across US and TASE markets, while strictly suppressing backward-looking post-rally alerts.

**Architecture:** A three-stage pipeline: (1) Stage 1 Economic Relevance Pre-filter on breaking feeds to eliminate noise with zero token cost; (2) Stage 2 Deep Causal Transmission Prompting in the LLM analyzer producing structured causal chains (`trigger_event` -> `transmission_mechanism` -> `predicted_beneficiaries`); (3) Stage 3 Two-Layer Decision Gate that blocks `POST_FACTO_RECAP` moves and awards alpha scoring bonuses to anticipatory opportunities before Telegram delivery.

**Tech Stack:** Python 3.10+, Pydantic v2, Pytest, PyYAML, feedparser, requests.

**Spec:** [docs/superpowers/specs/2026-10-29-predictive-causal-engine-design.md](file:///Users/omer.lustig/IdeaProjects/omer/docs/superpowers/specs/2026-10-29-predictive-causal-engine-design.md)

## Global Constraints

- Python 3.10+ typing (`from __future__ import annotations`, `Optional`, `List`, `Dict`, Pydantic models).
- Ticker format: tradeable US and TASE equity tickers prefixed with `$` (e.g. `$ELAL`, `$ESLT`, `$LMT`, `$XLE`, `$NVDA`).
- Absolute zero regressions on the existing 420 unit and integration tests (`PYTHONPATH=news-agent pytest news-agent/tests/`).
- Telegram messages must retain `"disable_web_page_preview": True` and clean word-boundary truncation.

## Review Focus

1. Article with post-earnings jump (e.g. "Oracle shares surged 11% on cloud guidance") must be tagged `POST_FACTO_RECAP` and blocked from generating a single-stock BUY alert unless an unpriced second-order partner is surfaced.
2. Israeli TASE tickers (e.g. `$ELAL`, `$ESLT`) must pass ticker validation without being dropped by the ticker whitelist/regex.
3. Breaking news triage filter must discard non-economic general news (local crime, sports, celebrities) while allowing aviation, defense, and sanctions news to pass.
4. Telegram alert formatter must cleanly render both predictive cards (with trigger, transmission, exchange badge, and horizon) and legacy analysis cards without throwing KeyErrors or missing field exceptions.
5. Missing or `None` `causal_transmission` in legacy articles must fall back gracefully throughout the pipeline without regressions.

---

### Task 1: Causal Models, Enums & TASE Ticker Support

**Files:**
- Modify: `news-agent/src/models/analysis.py`
- Modify: `news-agent/src/analysis/prompts.py:31-64`
- Create/Modify: `news-agent/tests/test_causal_engine.py`

**Interfaces:**
- Consumes: Pydantic `BaseModel`, `Field`, `Enum`.
- Produces: `CatalystStage`, `TimeHorizon`, `MarketExchange`, `CausalTransmissionAnalysis`, updated `MarketImpactAnalysis.causal_transmission`.

- [ ] **Step 1: Write the failing test for Causal Models and TASE Ticker Filtering**

In `news-agent/tests/test_causal_engine.py`:
```python
from src.models.analysis import (
    CatalystStage,
    TimeHorizon,
    MarketExchange,
    CausalTransmissionAnalysis,
    MarketImpactAnalysis,
)
from src.analysis.prompts import filter_valid_tickers

def test_causal_transmission_model_serialization():
    causal = CausalTransmissionAnalysis(
        catalyst_stage=CatalystStage.PREDICTIVE_ANTICIPATORY,
        trigger_event="Foreign airlines suspend Tel Aviv routes",
        transmission_mechanism="Removes 40% seat capacity; grants monopoly pricing power to flag carrier",
        time_horizon=TimeHorizon.TACTICAL_1_3_DAYS,
        market_exchange=MarketExchange.TASE,
        predicted_beneficiaries=["$ELAL"],
        unpriced_edge_thesis="Anticipatory window before traffic report discounts yield surge",
    )
    data = causal.model_dump()
    assert data["catalyst_stage"] == "PREDICTIVE_ANTICIPATORY"
    assert data["market_exchange"] == "TASE"
    assert "$ELAL" in data["predicted_beneficiaries"]

def test_tase_ticker_filtering_support():
    tickers = ["$ELAL", "$ESLT", "$LUMI", "ICL", "$NVDA", "$USA"]
    valid = filter_valid_tickers(tickers)
    assert "$ELAL" in valid
    assert "$ESLT" in valid
    assert "$LUMI" in valid
    assert "$ICL" in valid
    assert "$NVDA" in valid
    assert "$USA" not in valid  # Sovereign entity dropped
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py -v`
Expected: FAIL with `ImportError: cannot import name 'CatalystStage'`

- [ ] **Step 3: Implement `CatalystStage`, `TimeHorizon`, `MarketExchange`, `CausalTransmissionAnalysis` and update `MarketImpactAnalysis`**

In `news-agent/src/models/analysis.py`:
Add the enum classes and model as defined in the spec.
Update `MarketImpactAnalysis` with `causal_transmission: Optional[CausalTransmissionAnalysis] = None`.

In `news-agent/src/analysis/prompts.py`:
Ensure `filter_valid_tickers` correctly handles TASE symbols like `$ELAL`, `$ESLT`, `$LUMI`, `$POLI`, `$ICL`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/models/analysis.py news-agent/src/analysis/prompts.py news-agent/tests/test_causal_engine.py
git commit -m "feat(models): add causal transmission models, enums, and TASE ticker support"
```

---

### Task 2: Economic Relevance Pre-Filter for Breaking Feeds

**Files:**
- Create: `news-agent/src/analysis/triage.py`
- Modify: `news-agent/tests/test_causal_engine.py`

**Interfaces:**
- Consumes: `src.models.RawArticle`
- Produces: `is_economically_relevant(article: RawArticle) -> bool`

- [ ] **Step 1: Write the failing test for Economic Relevance Pre-Filter**

In `news-agent/tests/test_causal_engine.py`:
```python
from datetime import datetime, timezone
from src.models import RawArticle
from src.analysis.triage import is_economically_relevant

def test_triage_filter_allows_aviation_and_defense_shocks():
    article = RawArticle(
        title="Flight diverted after security alert over airspace",
        body="Foreign carriers are suspending flights to Tel Aviv amid regional tensions.",
        url="https://ynetnews.com/article/1",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is True

def test_triage_filter_drops_celebrity_and_sports_noise():
    article = RawArticle(
        title="Celebrity wedding draws thousands of fans in Tel Aviv",
        body="Actor married singer in private ceremony with star guests.",
        url="https://ynetnews.com/article/2",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News",
    )
    assert is_economically_relevant(article) is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py::test_triage_filter_allows_aviation_and_defense_shocks -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.analysis.triage'`

- [ ] **Step 3: Implement `is_economically_relevant(article: RawArticle) -> bool`**

In `news-agent/src/analysis/triage.py`:
Implement multilingual keyword matching (English & Hebrew) targeting:
- Aviation, shipping, defense, military strikes, sanctions, tariffs, energy/crude/oil, central bank/shekel/dollar, and labor/port strikes.
- Always returns `True` for dedicated financial sources (Reuters, Bloomberg, Calcalist, TheMarker, Seeking Alpha).
- Applies the filter strictly to general breaking/political feeds (Ynet Breaking, Google News World/Politics).

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/analysis/triage.py news-agent/tests/test_causal_engine.py
git commit -m "feat(analysis): add economic relevance pre-filter for breaking feeds"
```

---

### Task 3: Anticipatory & Second-Order Causal Prompting

**Files:**
- Modify: `news-agent/src/analysis/prompts.py`
- Modify: `news-agent/src/analysis/analyzer.py`
- Modify: `news-agent/tests/test_causal_engine.py`

**Interfaces:**
- Consumes: Raw text, `SYSTEM_ANALYST_PROMPT`, JSON response schema.
- Produces: Populated `causal_transmission` inside `MarketImpactAnalysis`.

- [ ] **Step 1: Write the failing test for Causal Analysis Extraction**

In `news-agent/tests/test_causal_engine.py`:
```python
from unittest.mock import MagicMock
from src.analysis.analyzer import MarketImpactAnalyzer
from src.models import RawArticle

def test_analyzer_parses_causal_transmission():
    mock_llm = MagicMock()
    mock_llm.generate.return_value = '''{
        "status": "OPPORTUNITY_CANDIDATE",
        "confidence_score": 9,
        "impact_grade": "CRITICAL",
        "affected_tickers": ["$ELAL"],
        "beneficiaries": ["$ELAL"],
        "negatively_exposed": [],
        "affected_sectors": ["Airlines", "Commercial Aviation"],
        "recommendation": "Prioritize tactical long exposure on $ELAL before flight suspension capacity squeeze is priced in",
        "causal_transmission": {
            "catalyst_stage": "PREDICTIVE_ANTICIPATORY",
            "trigger_event": "Foreign airlines halt flights to Israel",
            "transmission_mechanism": "Eliminates international seat capacity; yields surge",
            "time_horizon": "TACTICAL_1_3_DAYS",
            "market_exchange": "TASE",
            "predicted_beneficiaries": ["$ELAL"],
            "unpriced_edge_thesis": "Traffic surge not yet reflected in consensus earnings"
        }
    }'''
    analyzer = MarketImpactAnalyzer(llm_client=mock_llm)
    article = RawArticle(
        title="Foreign carriers cancel Tel Aviv flights",
        body="Major airlines suspend operations following airspace warning.",
        url="https://example.com/flights",
        published_at=datetime.now(timezone.utc),
        source_name="Test Source",
    )
    analysis = analyzer.analyze(article)
    assert analysis.causal_transmission is not None
    assert analysis.causal_transmission.catalyst_stage == CatalystStage.PREDICTIVE_ANTICIPATORY
    assert analysis.causal_transmission.market_exchange == MarketExchange.TASE
    assert "$ELAL" in analysis.causal_transmission.predicted_beneficiaries
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py::test_analyzer_parses_causal_transmission -v`
Expected: FAIL (either `causal_transmission` not parsed or LLM prompt lacks instructions).

- [ ] **Step 3: Update `SYSTEM_ANALYST_PROMPT` and `analyzer.py`**

In `news-agent/src/analysis/prompts.py`:
- Add `causal_transmission` to JSON response schema format in the prompt.
- Add Section: `PREDICTIVE CAUSAL TRANSMISSION & SECOND-ORDER ALPHA`:
  - Rule 1: Post-Facto Recap Rule (If headline/body says "shares rallied 10%+", tag `POST_FACTO_RECAP` and never recommend the jumped stock as new BUY).
  - Rule 2: Anticipatory Real-World Shocks (Aviation, defense, commodity chokepoint -> tag `PREDICTIVE_ANTICIPATORY`).
  - Rule 3: Second-Order Ripple Beneficiaries (Tag `SECOND_ORDER_UNPRICED`).
  - Rule 4: TASE Ticker Guidance (`$ELAL`, `$ESLT`, `$LUMI`, `$ICL`).
In `news-agent/src/analysis/analyzer.py`:
- Parse `causal_transmission` dict from JSON into `CausalTransmissionAnalysis` and assign to `analysis.causal_transmission`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/analysis/prompts.py news-agent/src/analysis/analyzer.py news-agent/tests/test_causal_engine.py
git commit -m "feat(analysis): incorporate predictive causal transmission playbook into prompt and analyzer"
```

---

### Task 4: Two-Layer Decision Gate Hard Gate & Alpha Scoring Boost

**Files:**
- Modify: `news-agent/src/analysis/decision_gate.py`
- Modify: `news-agent/tests/test_causal_engine.py`

**Interfaces:**
- Consumes: `MarketImpactAnalysis` with `causal_transmission`.
- Produces: `TwoLayerGateDecision` with `POST_FACTO_RECAP` suppression and alpha scoring bonuses.

- [ ] **Step 1: Write the failing test for Decision Gate Causal Rules**

In `news-agent/tests/test_causal_engine.py`:
```python
from src.analysis.decision_gate import TwoLayerGateEngine

def test_decision_gate_blocks_post_facto_recap():
    engine = TwoLayerGateEngine()
    analysis = MarketImpactAnalysis(
        status="OPPORTUNITY_CANDIDATE",
        confidence_score=9.0,
        impact_grade="HIGH",
        affected_tickers=["$ORCL"],
        beneficiaries=["$ORCL"],
        causal_transmission=CausalTransmissionAnalysis(
            catalyst_stage=CatalystStage.POST_FACTO_RECAP,
            trigger_event="Oracle shares rallied 11% after revenue projection",
            transmission_mechanism="Capex capped",
            time_horizon=TimeHorizon.SWING_2_4_WEEKS,
            market_exchange=MarketExchange.US,
            predicted_beneficiaries=["$ORCL"],
            unpriced_edge_thesis="Move already completed",
        )
    )
    decision = engine.evaluate_gates(analysis)
    assert decision.passes_opportunity_gate is False
    assert "POST_FACTO" in decision.decision_verdict

def test_decision_gate_boosts_anticipatory_catalyst():
    engine = TwoLayerGateEngine()
    analysis = MarketImpactAnalysis(
        status="OPPORTUNITY_CANDIDATE",
        confidence_score=7.0,
        impact_grade="HIGH",
        affected_tickers=["$ELAL"],
        beneficiaries=["$ELAL"],
        causal_transmission=CausalTransmissionAnalysis(
            catalyst_stage=CatalystStage.PREDICTIVE_ANTICIPATORY,
            trigger_event="Airlines cancel flights",
            transmission_mechanism="Domestic monopoly",
            time_horizon=TimeHorizon.TACTICAL_1_3_DAYS,
            market_exchange=MarketExchange.TASE,
            predicted_beneficiaries=["$ELAL"],
            unpriced_edge_thesis="Unpriced traffic surge",
        )
    )
    decision = engine.evaluate_gates(analysis)
    assert decision.passes_opportunity_gate is True
    assert decision.opportunity_score >= 8.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py::test_decision_gate_blocks_post_facto_recap -v`
Expected: FAIL

- [ ] **Step 3: Implement Causal Gate Rules in `TwoLayerGateEngine`**

In `news-agent/src/analysis/decision_gate.py`:
- In `evaluate_opportunity_score`:
  - If `analysis.causal_transmission.catalyst_stage == CatalystStage.PREDICTIVE_ANTICIPATORY`: add `+1.5` boost.
  - If `analysis.causal_transmission.catalyst_stage == CatalystStage.SECOND_ORDER_UNPRICED`: add `+1.0` boost.
- In `evaluate_gates`:
  - If `analysis.causal_transmission.catalyst_stage == CatalystStage.POST_FACTO_RECAP`:
    - Force `passes_opportunity_gate = False`.
    - Set `decision_verdict = "REJECTED_POST_FACTO_ALREADY_PRICED"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_causal_engine.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/analysis/decision_gate.py news-agent/tests/test_causal_engine.py
git commit -m "feat(gate): suppress post-facto moves and boost anticipatory opportunities"
```

---

### Task 5: Predictive Telegram Alert Formatting

**Files:**
- Modify: `news-agent/src/notification/telegram.py`
- Modify: `news-agent/tests/test_notification.py`

**Interfaces:**
- Consumes: `MarketImpactAnalysis` with `causal_transmission`.
- Produces: Formatted Telegram HTML string displaying exchange, horizon, trigger, transmission, edge.

- [ ] **Step 1: Write the failing test for Predictive Telegram Formatting**

In `news-agent/tests/test_notification.py`:
```python
def test_format_predictive_opportunity_item():
    analysis = MarketImpactAnalysis(
        status="OPPORTUNITY_CANDIDATE",
        confidence_score=9.0,
        impact_grade="CRITICAL",
        affected_tickers=["$ELAL"],
        beneficiaries=["$ELAL"],
        affected_sectors=["Commercial Aviation"],
        recommendation="Buy $ELAL before Q3 report",
        causal_transmission=CausalTransmissionAnalysis(
            catalyst_stage=CatalystStage.PREDICTIVE_ANTICIPATORY,
            trigger_event="Foreign airlines cancel Israel flights following security alert",
            transmission_mechanism="Seat capacity falls 40%; domestic carrier captures near-monopoly pricing power",
            time_horizon=TimeHorizon.TACTICAL_1_3_DAYS,
            market_exchange=MarketExchange.TASE,
            predicted_beneficiaries=["$ELAL"],
            unpriced_edge_thesis="Unpriced load factor surge before quarterly traffic filings",
        )
    )
    article = RawArticle(
        title="Airlines halt Tel Aviv routes",
        body="...",
        url="https://ynetnews.com/1",
        published_at=datetime.now(timezone.utc),
        source_name="Ynet Breaking News"
    )
    item = format_clean_opportunity_item(1, analysis, article)
    assert "PREDICTIVE" in item or "TASE" in item
    assert "🇮🇱 TASE" in item
    assert "Tactical (1-3 Days)" in item
    assert "Trigger:" in item
    assert "Transmission:" in item
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_notification.py::test_format_predictive_opportunity_item -v`
Expected: FAIL

- [ ] **Step 3: Update `format_clean_opportunity_item` in `telegram.py`**

In `news-agent/src/notification/telegram.py`:
- Check `if analysis.causal_transmission is not None:`
  - Add exchange badge: `🇮🇱 TASE` if `TASE` else `🇺🇸 US`.
  - Add horizon badge: `⏱️ Tactical (1-3 Days)` if `TACTICAL_1_3_DAYS` else `⏱️ Swing (2-4 Weeks)`.
  - Render:
    - `• 🎯 Trigger: {truncate_at_word(causal.trigger_event, 140)}`
    - `• ⚙️ Transmission: {truncate_at_word(causal.transmission_mechanism, 160)}`
    - `• 💡 Unpriced Edge: {truncate_at_word(causal.unpriced_edge_thesis, 140)}`
    - `• 🛡️ Invalidation: {truncate_at_word(counterarg, 120)}`
  - Fallback cleanly to standard format if `causal_transmission is None`.

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_notification.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add news-agent/src/notification/telegram.py news-agent/tests/test_notification.py
git commit -m "feat(telegram): format predictive opportunities with exchange, horizon, and causal bullets"
```

---

### Task 6: Targeted Geopolitical Feeds & Pipeline Integration

**Files:**
- Modify: `news-agent/config/sources.yaml`
- Modify: `news-agent/src/pipeline.py`
- Modify: `news-agent/tests/test_pipeline.py`

**Interfaces:**
- Consumes: Configured feeds in `sources.yaml`.
- Produces: Integrated end-to-end ingestion and triage flow.

- [ ] **Step 1: Write integration test for Pipeline Triage and Predictive Flow**

In `news-agent/tests/test_pipeline.py`:
Add test verifying that when an article comes from a breaking feed, `is_economically_relevant` is checked before calling LLM analysis.

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/test_pipeline.py -k test_pipeline_triage -v`
Expected: FAIL

- [ ] **Step 3: Integrate Triage Pre-Filter into `MarketNewsPipeline` and add feeds to `sources.yaml`**

In `news-agent/src/pipeline.py`:
- Before running heavy LLM `analyzer.analyze(article)`:
  - If source is tagged as breaking/general or non-financial, check `is_economically_relevant(article)`.
  - If `False`, skip LLM analysis and log debug message (saving token costs).
In `news-agent/config/sources.yaml`:
- Add `Reuters World Wire`, `Google News World`, `Ynet Breaking News`, `Ynet News English Breaking`, `Times of Israel News` with `category: breaking` or `type: rss`.

- [ ] **Step 4: Run full test suite to verify 100% pass rate**

Run: `PYTHONPATH=news-agent pytest news-agent/tests/ -v`
Expected: 425+ tests PASS with 0 failures and 0 warnings.

- [ ] **Step 5: Commit**

```bash
git add news-agent/config/sources.yaml news-agent/src/pipeline.py news-agent/tests/test_pipeline.py
git commit -m "feat(pipeline): integrate economic triage pre-filter and breaking feeds into pipeline"
```

---
