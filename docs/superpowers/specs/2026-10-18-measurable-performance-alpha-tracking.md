# Specification: Measurable Performance & Publisher Credibility Alpha Tracking

## 1. Problem Statement
Institutional investment operations require accountability: an alerting agent must not merely sound convincing; it must demonstrate whether its recommendations predict forward market returns, measure directional accuracy, and evaluate which publishers provide genuinely tradeable signals versus noise.

Following the completed Core Architectural Upgrades, `event_lineage` records:
- `event_id`, `title`, `primary_ticker`, `first_seen`, `source_count`, `all_sources`
- `event_quality_score`, `opportunity_score`, `status`
- `initial_price`, `subsequent_return_1d`, `subsequent_return_7d`

This specification defines the measurable performance evaluation subsystem, automated price update engine, hit rate calculator, publisher credibility scorecard, and CLI dashboard.

---

## 2. Core Mathematical & Directional Models

### 2.1 Directional Forward Return & Alpha Formula
For each evaluated event candidate with an `initial_price` and directional stance:
- Absolute Price Return:
  $$\text{Return}_{\text{raw}} = \frac{P_{t} - P_{0}}{P_{0}} \times 100$$
- Directional Alpha:
  - If Stance is `BULLISH`:
    $$\text{Alpha} = \text{Return}_{\text{raw}}$$
  - If Stance is `BEARISH`:
    $$\text{Alpha} = -\text{Return}_{\text{raw}}$$
  - If Stance is `NEUTRAL` or `VOLATILE`:
    $$\text{Alpha} = |\text{Return}_{\text{raw}}|$$

### 2.2 Hit Rate Metric
A recommendation is marked as a **WIN** (hit) if:
- Stance is `BULLISH` and $\text{Return}_{\text{raw}} > +0.20\%$ (accounting for friction)
- Stance is `BEARISH` and $\text{Return}_{\text{raw}} < -0.20\%$
- Stance is `VOLATILE` and $|\text{Return}_{\text{raw}}| > 1.50\%$

A recommendation is marked as a **LOSS** (miss) if:
- Stance is `BULLISH` and $\text{Return}_{\text{raw}} < -0.20\%$
- Stance is `BEARISH` and $\text{Return}_{\text{raw}} > +0.20\%$

$$\text{Hit Rate} = \frac{\text{Wins}}{\text{Wins} + \text{Losses}} \times 100$$

### 2.3 Publisher Credibility Scoring & Tiers
For each publisher newsroom (e.g., Reuters, Bloomberg, Financial Times, Globes, Calcalist, SEC EDGAR):
- `total_catalysts`: Total events where publisher was primary or corroborating source
- `avg_quality_score`: Average Layer 1 Event Quality Score (0-10)
- `hit_rate_pct`: Historical directional hit rate (%)
- `avg_alpha_1d`: Average 1-day forward alpha (%)
- `avg_alpha_7d`: Average 7-day forward alpha (%)

**Credibility Tier Rubric**:
- `TIER_A_ALPHA` (Institutional Edge): Hit Rate $\ge 65\%$ and Average 1d Alpha $\ge +0.80\%$
- `TIER_B_RELIABLE` (High Reliability): Hit Rate $55\% - 64\%$
- `TIER_C_NEUTRAL` (Market In-Line): Hit Rate $45\% - 54\%$
- `TIER_D_UNRELIABLE` (Noise / Falsified Rumors): Hit Rate $< 45\%$ or negative average alpha.

---

## 3. Data Contracts (`src/models.py`)

### 3.1 `CredibilityTier(str, Enum)`
```python
class CredibilityTier(str, Enum):
    TIER_A_ALPHA = "TIER_A_ALPHA"
    TIER_B_RELIABLE = "TIER_B_RELIABLE"
    TIER_C_NEUTRAL = "TIER_C_NEUTRAL"
    TIER_D_UNRELIABLE = "TIER_D_UNRELIABLE"
```

### 3.2 `SourceCredibilityRecord(BaseModel)`
```python
class SourceCredibilityRecord(BaseModel):
    source_name: str
    total_events: int
    avg_quality_score: float
    hit_rate_pct: float
    avg_alpha_1d_pct: float
    avg_alpha_7d_pct: float
    credibility_tier: CredibilityTier
```

### 3.3 `PerformanceMetricsSummary(BaseModel)`
```python
class PerformanceMetricsSummary(BaseModel):
    total_catalysts: int
    resolved_1d: int
    resolved_7d: int
    wins_1d: int
    losses_1d: int
    hit_rate_1d_pct: float
    avg_return_1d_pct: float
    avg_alpha_1d_pct: float
    wins_7d: int
    losses_7d: int
    hit_rate_7d_pct: float
    avg_return_7d_pct: float
    avg_alpha_7d_pct: float
    source_scorecard: List[SourceCredibilityRecord]
```

---

## 4. Subsystem Components

1. **`PerformanceTrackerEngine` (`src/analysis/performance.py`)**:
   - `update_forward_returns(storage, price_fetcher)`: Identifies events needing 1d/7d price quotes, queries market price, computes returns, and updates SQLite.
   - `calculate_metrics(storage, min_opportunity, ticker_filter, source_filter)`: Aggregates return ledger and computes global metrics + publisher scorecard.
2. **Storage Extensions (`src/storage/sqlite.py` & `src/storage/base.py`)**:
   - `get_pending_return_events(horizon: str, min_age_hours: int)`: Queries un-resolved events.
   - `get_completed_lineage_records()`: Queries all records with realized returns.
3. **CLI Dashboard (`src/cli.py`)**:
   - `python3 -m src.main performance [--update] [--ticker TICKER] [--source SOURCE]`:
     Renders Rich tables for:
     * Global Alpha & Hit Rate Metrics
     * Publisher Credibility Scorecard
     * Realized Performance Ledger

---

## 5. Verification & Testing Requirements
- Unit tests for return and directional alpha math (Bullish gain = win, Bearish drop = win).
- Unit tests for Publisher Credibility Tier grading.
- Storage tests for updating forward returns and querying pending events.
- CLI tests for `performance` command output rendering.
- 100% test pass rate with 0 regressions.
