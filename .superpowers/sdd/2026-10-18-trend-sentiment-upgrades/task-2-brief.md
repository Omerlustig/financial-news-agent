# Task 2 Brief: Quantitative Sentiment Acceleration, Divergence, and Sector Trend Analysis

## Context
Deepening quantitative market trend and sentiment understanding.
We are introducing:
1. **Sentiment Acceleration (Second Derivative: $\Delta^2 S / \Delta t^2$)**:
   - Velocity is $V = S_{24h} - S_{prior}$
   - Acceleration is $A = V_{current} - V_{prior}$ (or change in rate of sentiment change over sub-windows). If sentiment was falling at -0.3 and is now rising at +0.2, acceleration is $+0.5$ (sharp bullish inflection/reversal).
2. **Volume-Sentiment Divergence Detector**:
   - `BEARISH_PANIC_LIQUIDATION`: High buzz ($\ge 2.0$) with negative sentiment ($\le -0.3$) and negative velocity.
   - `BULLISH_ACCUMULATION_DIVERGENCE`: Normal/low buzz ($\le 1.2$) but strong positive velocity ($> +0.2$) and positive sentiment ($> +0.3$) - quiet accumulation.
3. **Sector-Level Trend Calculation**:
   - `calculate_sector_trend(storage: BaseStorage, sector: str, current_sentiment: Optional[float] = None) -> SectorTrendMetrics`:
     Queries `storage.get_sector_sentiment_window(sector, hours=24)` and 7d baseline to calculate sector buzz ratio, sentiment velocity, acceleration, and verdict.
   - `enrich_analysis_with_sector_trends(storage: BaseStorage, analysis: MarketImpactAnalysis) -> MarketImpactAnalysis`:
     For each sector in `analysis.affected_sectors`, populates `analysis.sector_trends`.

## Scope & Deliverables
All code and modifications MUST be in `news-agent/`.

1. **Update `news-agent/src/analysis/trend.py`**:
   - Update `TrendEngine`:
     * Implement acceleration calculation in `calculate_ticker_trend`:
       Compute velocity in current window vs prior window to derive `sentiment_acceleration = round(velocity_current - velocity_prior, 3)`.
     * Implement divergence detector:
       Identify `BEARISH_PANIC_LIQUIDATION` and `BULLISH_ACCUMULATION_DIVERGENCE`. Populate `is_divergent` and `divergence_type`.
     * Update `trend_verdict` logic:
       - If `sentiment_acceleration >= 0.25` and `avg_sentiment_24h > 0`: `TrendVerdict.BULLISH_REVERSAL`
       - If `sentiment_acceleration <= -0.25` and `avg_sentiment_24h < 0`: `TrendVerdict.BEARISH_BREAKDOWN`
     * Implement `calculate_sector_trend(storage, sector, current_sentiment=None) -> SectorTrendMetrics`.
     * Implement `enrich_analysis_with_sector_trends(storage, analysis) -> MarketImpactAnalysis`.
     * Update `enrich_analysis_with_trend()` to also enrich sector trends.

2. **Tests (`news-agent/tests/test_trend.py`)**:
   - Test sentiment acceleration with inflection fixtures.
   - Test volume-sentiment divergence detection.
   - Test sector trend calculation with mock and SQLite storage.
   - Test `BULLISH_REVERSAL` and `BEARISH_BREAKDOWN` verdicts.

3. **Verification**:
   - Run `python -m pytest news-agent/tests/ -v` and ensure 100% test pass rate.
