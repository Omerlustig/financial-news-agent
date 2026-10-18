# Task 3 Report: Financial Prompt Logic, Watchlist Alignment, and Reversal Reasoning

**Status:** DONE  
**Date:** 2026-10-18  
**Implementer:** Expert Financial Advisor & Stock Analyst  

---

## 1. Executive Summary

Task 3 of the trend and sentiment upgrades has been fully implemented in `news-agent/`:
- **Inflection/Reversal vs Trend Continuation**: Upgraded `SYSTEM_ANALYST_PROMPT` and `build_analyst_prompt` evaluation instructions to explicitly instruct the LLM to identify whether a catalyst represents an inflection/reversal (e.g., positive surprise after prolonged downtrend, cyclical trough, turnaround, sudden margin squeeze ending an expansion) versus a trend continuation (e.g., persistent secular momentum, recurring beat in line with guidance).
- **Sectoral Spillovers & Asymmetric Impact**: Prompt instructions now explicitly demand evaluating 2nd-order sectoral spillovers and asymmetric consequences across sub-sectors, supply chain tiers, and competing business models (e.g., how an airline fuel surcharge differentially affects air freight logistics vs commercial passenger carriers, or how chip export controls severely curtail lithography equipment makers vs domestic foundries vs downstream cloud hyperscalers).
- **Watchlist Context & Transmission Channel Depth**: Implemented optional watchlist tickers and priority sectors injection into the prompt context (`WATCHLIST CONTEXT (PRIORITY SURVEILLANCE)`) and enhanced `MarketImpactAnalyzer` to automatically correlate affected tickers/sectors with active user watchlists, setting `is_watchlist_match` and `matched_watchlist_items`.
- **Institutional Schema Adherence & Fallback Resilience**: 
  - `MarketImpactAnalyzer._parse_and_validate()` validates the complete `MarketImpactAnalysis` model without dropping fields, handles missing/partial inner fields in `causal_chain` (`event_facts`, `economic_mechanism`, `beneficiaries`, `negatively_exposed`, `counterarguments`, `invalidation_conditions`), and synthesizes an actionable `recommendation` if omitted.
  - `_build_fallback()` generates fully compliant models with valid defaults for all fields (`is_watchlist_match=False`, `matched_watchlist_items=[]`, `sector_trends=[]`, `surge_level=SurgeLevel.NORMAL`, institutional fallback recommendation, calibrated sentiment, compliant `CausalChain` and `HardGates`).
- **Comprehensive Testing**: Added 7 new unit tests in `news-agent/tests/test_analysis.py` and `news-agent/tests/test_sentiment.py`. Full test suite executed with 292 passing tests (100% pass rate).

---

## 2. Key Code Modifications

### 2.1 `news-agent/src/analysis/prompts.py`
- **`SYSTEM_ANALYST_PROMPT`**:
  - Added deterministic instructions for **INFLECTION / REVERSAL VS TREND CONTINUATION** under skill 3 (`analyze_financial_impact`).
  - Added instructions for **SECTORAL SPILLOVERS & ASYMMETRIC IMPACT** under skill 3.
  - Added instructions for **WATCHLIST & TRANSMISSION CHANNEL DEPTH** under skill 3.
  - Upgraded **BALANCED DEBATE & ACTIONABLE RECOMMENDATION LOGIC** to explicitly require `affected_sectors`, `beneficiaries`, `negatively_exposed`, `counterarguments`, `invalidation_conditions`, and `recommendation`.
  - Updated output JSON specification to incorporate `recommendation`, `is_watchlist_match`, and `matched_watchlist_items`.
- **`build_analyst_prompt()`**:
  - Added optional arguments `watchlist_tickers: Optional[List[str]] = None` and `watchlist_sectors: Optional[List[str]] = None`.
  - Formats and injects `WATCHLIST CONTEXT (PRIORITY SURVEILLANCE)` into `user_prompt` when present.
  - Updated `EVALUATION INSTRUCTIONS` to explicitly guide the LLM on:
    * Inflection vs Continuation
    * Sectoral Spillovers (freight vs passenger, equipment makers vs foundries)
    * Watchlist Focus
    * Balanced Debate & Actionable Recommendation

### 2.2 `news-agent/src/analysis/analyzer.py`
- **`MarketImpactAnalyzer.__init__()`**:
  - Added optional parameter `watchlist_settings: Optional[WatchlistSettings] = None`.
- **`analyze()`**:
  - Added optional parameters `watchlist_tickers: Optional[List[str]] = None` and `watchlist_sectors: Optional[List[str]] = None`, falling back to `self.watchlist_settings` if configured.
  - Propagates watchlist context to `build_analyst_prompt()` and `_parse_and_validate()`.
- **`_parse_and_validate()`**:
  - Ensures robust parsing of `causal_chain` (sanitizing tickers, auto-populating missing facts or economic mechanism from summary/thesis, ensuring string defaults for `counterarguments` and `invalidation_conditions`).
  - Implements watchlist matching against configured/passed watched tickers and sectors, setting `is_watchlist_match` and populating `matched_watchlist_items`.
  - Ensures `recommendation` is retained or synthesized with institutional directional phrasing if omitted.
  - Guarantees `MarketImpactAnalysis` model validation without dropping fields.
- **`_build_fallback()`**:
  - Guarantees valid defaults for all newly added and existing fields (`recommendation`, `is_watchlist_match=False`, `matched_watchlist_items=[]`, `sector_trends=[]`, `surge_level=SurgeLevel.NORMAL`, `sec_filing_item`, etc.), producing zero validation errors on serialization and deserialization.

### 2.3 `news-agent/tests/test_analysis.py`
- `test_prompt_construction_inflection_reversal_and_sector_spillovers`: Validates that system and user prompts contain inflection/reversal and sector spillover evaluation instructions.
- `test_prompt_construction_with_watchlist_context`: Validates watchlist context injection into user prompt.
- `test_analyzer_parses_updated_institutional_schema_fields`: Validates end-to-end parsing of `recommendation`, `causal_chain`, `is_watchlist_match`, and `matched_watchlist_items`.
- `test_analyzer_auto_matches_watchlist_when_configured`: Validates automated watchlist matching when `WatchlistSettings` are provided.
- `test_analyzer_fallback_complete_model_defaults_zero_validation_errors`: Validates fallback analysis model completeness and round-trip serialization with zero validation errors.
- `test_analyzer_causal_chain_resilience_missing_optional_inner_fields`: Validates graceful recovery and default filling when LLM omits inner causal chain fields.

### 2.4 `news-agent/tests/test_sentiment.py`
- `test_prompt_combines_sentiment_and_inflection_reversal_guidance`: Validates coexistence of financial sentiment and inflection/reversal guidelines.
- `test_analyzer_calibrates_sentiment_and_preserves_recommendation_and_watchlist`: Validates sentiment engine calibration while retaining recommendation and watchlist metadata.

---

## 3. Test Verification Results

Full test suite execution:
```
python -m pytest news-agent/tests/ -v
============================= 292 passed in 9.50s ==============================
```
- Total tests: 292
- Passed: 292
- Failed: 0
- Pass rate: 100%
