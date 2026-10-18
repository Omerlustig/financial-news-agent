# Institutional Intelligence Optimization — Implementation Plan

**Spec**: `docs/superpowers/specs/2026-10-18-institutional-intelligence-optimization.md`  
**Date**: 2026-10-18  

---

### Task 1: Data Models & Storage Schema Foundation (`software-architect`)
- **Files**:
  - `news-agent/src/models.py`
  - `news-agent/src/storage/base.py`
  - `news-agent/src/storage/sqlite.py`
  - `news-agent/tests/test_storage.py`
- **Deliverables**:
  - Pydantic models: `StatementType`, `ProvenanceLevel`, `NoveltyState`, `FourDimensionalSentiment`, `AdversarialReview`, `TwoLayerGateDecision`, `EventLineageRecord`.
  - Attach new fields to `MarketImpactAnalysis`.
  - SQLite schema upgrade: `event_lineage` table with CRUD methods.
  - Unit tests verifying model serialization and storage persistence.

---

### Task 2: Fact Verification & Novelty Engines (`financial-stock-expert`)
- **Files**:
  - `news-agent/src/analysis/verification.py`
  - `news-agent/src/analysis/novelty.py`
  - `news-agent/tests/test_verification.py`
  - `news-agent/tests/test_novelty.py`
- **Deliverables**:
  - `FactVerificationEngine`: extracts atomic claims, classifies statements (Verified Fact, Attributed Claim, Analyst Opinion, Anonymous Rumor), checks primary source matching, counts independent newsrooms.
  - `NoveltyEngine`: checks expectation gap against guidance/consensus, classifies novelty state (`GENUINE_SURPRISE`, `IN_LINE_CONFIRMATION`, `PRICED_IN_OR_RECYCLED`).
  - Unit tests for fact verification and expectation gap detection.

---

### Task 3: 4D Sentiment & Adversarial Challenger (`financial-stock-expert`)
- **Files**:
  - `news-agent/src/analysis/sentiment_4d.py`
  - `news-agent/src/analysis/challenger.py`
  - `news-agent/src/analysis/prompts.py`
  - `news-agent/src/analysis/analyzer.py`
  - `news-agent/tests/test_sentiment_4d.py`
  - `news-agent/tests/test_challenger.py`
- **Deliverables**:
  - `FourDimensionalSentimentAnalyzer`: calculates linguistic sentiment, fundamental cash-flow impact, expectation surprise, and directional price implication.
  - `AdversarialChallenger`: executes self-argumentation, identifies strongest bearish counter-thesis, priced-in risks, macro vs alpha confounding, and explicit falsification conditions.
  - Prompt updates incorporating these deterministic analysis stages.
  - Unit tests.

---

### Task 4: Quantitative Confirmation & Two-Layer Decision Gates (`quant-researcher`)
- **Files**:
  - `news-agent/src/analysis/technical.py`
  - `news-agent/src/analysis/decision_gate.py`
  - `news-agent/tests/test_decision_gate.py`
  - `news-agent/tests/test_technical.py`
- **Deliverables**:
  - `TradingViewEngine` enhancement: quantitative RSI thresholds, SMA-50/200 position, overbought/oversold risk flags.
  - `TwoLayerGateEngine`:
    - Evaluates `event_quality_score` (0-10) based on factual evidence, primary sources, corroboration.
    - Evaluates `opportunity_score` (0-10) based on surprise delta, cash flow alteration, technical alignment, risk/reward asymmetry.
    - Applies alert filter: both scores must meet threshold (default >= 8) to trigger external alerts.
  - Unit tests.

---

### Task 5: Pipeline Integration, Telegram Alert Upgrade & CLI Commands (`software-developer`)
- **Files**:
  - `news-agent/src/pipeline.py`
  - `news-agent/src/notification/telegram.py`
  - `news-agent/src/cli.py`
  - `news-agent/tests/test_pipeline.py`
  - `news-agent/tests/test_notification.py`
- **Deliverables**:
  - Pipeline orchestrator wired with all 6 stages.
  - Telegram alert layout upgraded with:
    - `Event Quality: X/10` and `Opportunity Score: Y/10`
    - 4D Sentiment breakdown
    - Adversarial Counter-Thesis & Invalidation Threshold
    - TradingView technical confluence badge
  - CLI `history` command to inspect persistent event lineage.
  - Integration tests.

---

### Task 6: QA Verification, Git Commit & Deployment (`qa-engineer`)
- **Deliverables**:
  - Full test suite run (`pytest news-agent/tests/ -v`).
  - Verify 0 regressions and 100% test pass rate.
  - Git commit with author `Omerlustig <omerlustig@gmail.com>`.
  - Push to `origin main`.
  - Deploy to Firebase Functions (`firebase deploy --only functions`).
