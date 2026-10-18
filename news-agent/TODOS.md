# Market News Agent - Optimization & Roadmap TODOs

## Strategic Roadmap & Enhancements

- [ ] **Quantitative confirmation** — combine news with actual financial and market data:
  - Intraday price change and volume relative to historical averages (volume spikes / RVOL).
  - Relative performance vs. sector ETF / market benchmark (SPY/QQQ/TA-125).
  - Pre-publication price momentum and trend divergence.
  - Valuation context & consensus earnings expectations to evaluate if a catalyst is already priced in.
  - Assess whether catalyst permanently alters discounted future cash flows or is merely transient headline noise.

- [ ] **Measurable performance** — test whether the agent's recommendations predict returns, not just whether its analysis sounds convincing:
  - Track recommendation timestamp alongside initial stock price.
  - Measure subsequent alpha/excess return over predefined forward horizons (1-day, 3-day, 7-day, 30-day).
  - Systematic hit rate, Sharpe/Sortino ratios, and false positive rates per source newsroom.
  - Source credibility performance scoring across US and Israeli publishers.

## Core Architectural Upgrades (Completed)
- [x] 1. Fact Verification & Provenance Engine: Separate facts, reported claims, and speculation; verify material events against official primary sources (SEC 8-K/10-Q, MAYA, Fed, official press releases) and require independent corroboration.
- [x] 2. Novelty & Expectation Gap Analysis: Contrast breaking news against pre-existing consensus, prior guidance, and pre-news price action (distinguish good news vs unexpectedly good news vs priced-in news).
- [x] 3. Four-Dimensional Sentiment & Impact Decomposition: Disentangle raw linguistic sentiment, fundamental business impact, market expectation surprise, and directional price implication.
- [x] 4. Self-Challenger & Thesis Invalidation Subagent: Adversarial review arguing against the initial thesis, detecting priced-in risk, sector co-movement, and defining explicit invalidation criteria.
- [x] 5. Two-Layer Gate Decision Engine: Decouple News Event Quality score from Actionable Investment Opportunity score.
- [x] 6. Persistent Structured Event Memory & Lineage: Track event IDs, lineage across multiple days, historical catalyst returns, source reliability metrics, and deduplication across follow-up stories.
