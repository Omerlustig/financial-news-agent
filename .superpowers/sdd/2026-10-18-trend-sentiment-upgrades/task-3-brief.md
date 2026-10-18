# Task 3 Brief: Financial Prompt Logic, Watchlist Alignment, and Reversal Reasoning

## Context
Upgrading the LLM prompt engine and financial reasoning to incorporate:
1. Sector-level macroeconomic context and correlation.
2. Watchlist focus: when evaluating news for priority tickers or sectors, provide deep transmission channel analysis.
3. Market stance nuances: incorporate inflection awareness (reversal vs continuation, accumulation vs distribution).

## Scope & Deliverables
All code and modifications MUST be in `news-agent/`.

1. **Update `news-agent/src/analysis/prompts.py`**:
   - Enhance institutional system and catalyst evaluation prompts to instruct the LLM:
     * To identify whether the catalyst represents an **inflection/reversal** (e.g. positive surprise after prolonged downtrend) or **trend continuation**.
     * To explicitly evaluate sectoral spillovers (e.g. how airline fuel surcharge affects freight vs passenger carriers, or how chip export controls affect equipment makers vs foundries).
     * Output clean JSON compliant with updated `MarketImpactAnalysis`.

2. **Update `news-agent/src/analysis/analyzer.py`**:
   - Ensure `MarketImpactAnalyzer` handles the updated schema gracefully, including `is_watchlist_match` and `matched_watchlist_items`.
   - Ensure fallback analysis provides valid defaults for newly added fields.

3. **Tests (`news-agent/tests/test_analysis.py`, `news-agent/tests/test_sentiment.py`)**:
   - Add unit tests verifying prompt instructions include sector spillover and inflection guidelines.
   - Verify fallback analyzer generates compliant models with zero validation errors.

4. **Verification**:
   - Run `python -m pytest news-agent/tests/ -v` and ensure 100% test pass rate.
