# Predictive Causal Intelligence & Second-Order Opportunity Engine

**Date:** 2026-10-29  
**Status:** Approved for Implementation Planning  
**Target:** `news-agent/`  

---

## 1. Problem Statement & Motivation

Currently, the news agent frequently generates opportunity alerts based on backward-looking, post-facto market movements. For instance:
> *"Catalyst: Oracle shares rallied 11% after projecting fiscal 2030 revenue of $225B and capping additional fundraising, easing capex concerns"*

By the time this alert reaches Telegram, the stock has already surged 11%. There is no trading edge or unpriced alpha left for an investor to capture. 

Furthermore, real-world developments in **geopolitics, regulatory decisions, macro shocks, and breaking political/defense events** often drive asymmetric price reactions in equities and sectors *before* the financial press discounts them. For example:
- **Trigger Event:** Foreign airlines suspend flights to Tel Aviv following regional security incidents or airspace closures.
- **Economic Transmission Mechanism:** Instant removal of 40% of seat capacity grants domestic flag carrier **El Al ($ELAL)** near-monopoly pricing power, surging passenger load factors, and record passenger yields.
- **Actionable Opportunity:** **BUY $ELAL (TASE)** with a tactical (1–3 day) horizon, alerted *before* quarterly passenger traffic figures or earnings reports make the upside consensus.

The objective of this design is to upgrade the intelligence pipeline to:
1. **Actively read between the lines & decode subcontext** from world events, corporate releases, and political developments.
2. **Trace causal transmission chains**: `Real-World Trigger` $\rightarrow$ `Economic Transmission Mechanism` $\rightarrow$ `Unpriced Asymmetric Beneficiaries`.
3. **Strictly suppress post-facto price recaps**: Prevent already-surged stocks from triggering single-stock BUY recommendations.
4. **Identify 2nd-order beneficiaries**: If Stock A surges, seek unpriced suppliers, peers, or sector ETFs that have not yet moved.
5. **Support dual markets (US + TASE)**: Accurately identify both US equities/ETFs (`$LMT`, `$XLE`, `$NVDA`) and Israeli Tel Aviv Stock Exchange equities (`$ELAL`, `$ESLT`, `$ICL`, `$LUMI`).

---

## 2. Goals & Non-Goals

### Goals
- **Anticipatory Alpha**: Alert on opportunities *before* price moves are fully realized, capturing the anticipatory window.
- **Causal Chain Transparency**: Every alert must explicitly articulate the causal transmission chain so the investor understands the economic mechanics in 3–5 seconds.
- **Post-Move Suppression**: Block single-stock BUY alerts where the primary catalyst is that the stock already surged 10%+ on news (`POST_FACTO_RECAP`).
- **Dual-Market Ticker Support**: Support and format Israeli TASE tickers (`$ELAL`, `$ESLT`, `$LUMI`, `$TEVA`) alongside US equities.
- **Multi-Horizon Classification**: Tag recommendations as `Tactical (1-3 Days)` (e.g. flight suspension, commodity shock) vs `Swing/Thematic (2-4 Weeks)` (e.g. tariff implementation, earnings forecast inflection).
- **Economic Relevance Triage**: Ingest breaking geopolitical/defense feeds while silently discarding non-economic political noise without consuming LLM analysis tokens.

### Non-Goals
- Real-time order execution or direct brokerage trade execution.
- Guaranteed price targets or personal financial advice (system produces institutional hypotheses).
- Building an exhaustive static database graph of all corporate relationships (the engine uses dynamic LLM subcontext reasoning).

---

## 3. High-Level Architecture & End-to-End Data Flow

```
[RSS & Breaking Feeds]
 (Financial + Geopolitical + Israeli Breaking News)
                     │
                     ▼
  [Stage 1: Economic Relevance Triage Filter]
  • High-speed heuristic check on non-financial breaking feeds
  • Silently drops non-economic noise (campaign speeches, celebrity, sports)
  • Forwards economically relevant shocks (airspace, tariffs, defense, energy)
                     │
                     ▼
  [Stage 2: Deep Causal Reasoning & Subcontext Engine (LLM)]
  • Subcontext Decoder: Decodes underlying motives and operational consequences
  • Causal Transmission: Traces Trigger -> Transmission Mechanism -> Beneficiaries
  • Pricing Status Audit:
      - PREDICTIVE_ANTICIPATORY (Event happened, stock unpriced -> HIGH ALPHA)
      - SECOND_ORDER_UNPRICED (Stock A surged, but Stock B is unpriced ripple winner)
      - POST_FACTO_RECAP (Stock already rallied 10%+ -> BLOCKED)
  • Market Exchange: Tagged as US vs TASE
  • Time Horizon: Tagged as Tactical (1-3d) vs Swing (2-4w)
                     │
                     ▼
  [Stage 3: Two-Layer Gate & Hard Filters]
  • Blocks POST_FACTO_RECAP from passes_opportunity_gate
  • Awards +1.5 bonus to PREDICTIVE_ANTICIPATORY catalysts
  • Awards +1.0 bonus to SECOND_ORDER_UNPRICED catalysts
                     │
                     ▼
  [Stage 4: Telegram Push Notification]
  • High-impact, zero-noise card formatting
  • Displays Exchange, Horizon, Trigger, Transmission, and Unpriced Edge
```

---

## 4. Detailed Component Design

### 4.1 Data Models & Schemas (`src/models/analysis.py`)

New structured enums and sub-models:

```python
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field

class CatalystStage(str, Enum):
    PREDICTIVE_ANTICIPATORY = "PREDICTIVE_ANTICIPATORY"  # Real-world event; stock has not moved yet (Maximum Alpha)
    SECOND_ORDER_UNPRICED   = "SECOND_ORDER_UNPRICED"    # Stock A rallied, but supplier/rival Stock B is unpriced
    POST_FACTO_RECAP        = "POST_FACTO_RECAP"         # Stock already surged 10%+; move already priced in

class TimeHorizon(str, Enum):
    TACTICAL_1_3_DAYS = "TACTICAL_1_3_DAYS"  # Fast reaction (airspace chokepoint, defense flare-up, cargo delay)
    SWING_2_4_WEEKS   = "SWING_2_4_WEEKS"    # Fundamental repricing (policy/tariffs/capex cycle/earnings revision)

class MarketExchange(str, Enum):
    US     = "US"      # NYSE, NASDAQ
    TASE   = "TASE"    # Tel Aviv Stock Exchange
    GLOBAL = "GLOBAL"  # Multi-market / ETFs

class CausalTransmissionAnalysis(BaseModel):
    catalyst_stage: CatalystStage = Field(
        description="Stage of catalyst: PREDICTIVE_ANTICIPATORY, SECOND_ORDER_UNPRICED, or POST_FACTO_RECAP"
    )
    trigger_event: str = Field(
        description="Factual real-world trigger (e.g. 'Foreign airlines suspend flights to Tel Aviv')"
    )
    transmission_mechanism: str = Field(
        description="Step-by-step economic transmission (e.g. '40% seat capacity removed -> pricing power to $ELAL')"
    )
    time_horizon: TimeHorizon = Field(
        description="Expected transmission horizon: TACTICAL_1_3_DAYS or SWING_2_4_WEEKS"
    )
    market_exchange: MarketExchange = Field(
        description="Primary exchange of the tradeable opportunity: US or TASE"
    )
    predicted_beneficiaries: List[str] = Field(
        default_factory=list,
        description="Tradeable tickers positioned to rise (e.g. ['$ELAL', '$ESLT'])"
    )
    unpriced_edge_thesis: str = Field(
        description="Why this opportunity has not yet been fully priced in by the broader market"
    )
```

`MarketImpactAnalysis` is updated to include:
```python
causal_transmission: Optional[CausalTransmissionAnalysis] = None
```

### 4.2 Economic Relevance Pre-Filter (`src/analysis/triage.py`)

For non-financial breaking/political RSS feeds, a high-throughput, low-latency filter determines if an article qualifies for LLM analysis.

* **Relevance Keywords (English & Hebrew)**:
  - Transport & Logistics: airlines, flights, airspace, carriers, shipping, ports, strait, cargo, נמלים, טיסות, תעופה
  - Defense & Conflict: strikes, missile, defense contract, drone, military spending, sanctions, ביטחון, תקיפה, טילים, רחפנים
  - Commodities & Energy: crude oil, natural gas, refining, pipeline, OPEC, נפט, גז
  - Trade & Policy: tariffs, export ban, embargo, subsidies, מכסים, סנקציות, יצוא
  - Currencies & Rates: central bank, interest rate, shekel, dollar, inflation, בנק ישראל, ריבית, שקל, אינפלציה
* If the article originates from a general breaking news feed and contains none of these economic transmission markers, it is dropped at `Stage 1` (0 LLM token cost).

### 4.3 Second-Order Prompt & Playbook (`src/analysis/prompts.py`)

The `SYSTEM_ANALYST_PROMPT` is enhanced with the **Anticipatory & Causal Transmission Playbook**:

1. **Strict Post-Facto Rule**:
   - When text reports *"Stock X rallied 10% after earnings"* or *"shares surged post-market"*, the primary jumped company is tagged `POST_FACTO_RECAP`.
   - The model is strictly prohibited from recommending the surged stock as a new BUY.
   - It must either discover an unpriced supplier/competitor (`SECOND_ORDER_UNPRICED`) or declare status `INFORMATIONAL`.

2. **Reading Between the Lines & Hidden Motives**:
   - Decode corporate and political subtext:
     - Geopolitical flight suspensions $\rightarrow$ Windfall pricing power for domestic carrier `$ELAL`.
     - Regional defense escalations $\rightarrow$ Order backlog acceleration for `$ESLT` (Elbit Systems) and US defense primes (`$LMT`, `$RTX`).
     - Major chip capex announcements $\rightarrow$ Equipment toolmakers (`$ASML`, `$AMAT`, `$CAMT`).
     - Foreign export restrictions $\rightarrow$ Protected domestic substitutes.

3. **TASE & US Ticker Support**:
   - Whitelist and recognize Tel Aviv Stock Exchange tickers: `$ELAL`, `$ESLT`, `$ICL`, `$LUMI`, `$POLI`, `$DSCT`, `$BEZQ`, `$AZRG`, `$CAMT`, `$OPK`, `$ENOG`, `$MNDY`, `$CHKP`, `$NICE`, `$TEVA`.
   - Distinguish exchange origin cleanly (`US` vs `TASE`).

### 4.4 Decision Gate Rules (`src/analysis/decision_gate.py`)

In `TwoLayerGateEngine.evaluate_gates`:
```python
if analysis.causal_transmission is not None:
    # 1. Hard Block on Post-Facto Recaps:
    if analysis.causal_transmission.catalyst_stage == CatalystStage.POST_FACTO_RECAP:
        passes_opportunity_gate = False
        decision_verdict = "REJECTED_POST_FACTO_ALREADY_PRICED"

    # 2. Anticipatory Alpha Scoring Bonus:
    if analysis.causal_transmission.catalyst_stage == CatalystStage.PREDICTIVE_ANTICIPATORY:
        opportunity_score = min(10.0, opportunity_score + 1.5)
    elif analysis.causal_transmission.catalyst_stage == CatalystStage.SECOND_ORDER_UNPRICED:
        opportunity_score = min(10.0, opportunity_score + 1.0)
```

### 4.5 Telegram Alert Presentation (`src/notification/telegram.py`)

When an opportunity contains a `causal_transmission` object, the notification formats as an anticipatory predictive card:

```text
🚨 PREDICTIVE OPPORTUNITY (Score: 9/10)
🟢 BUY $ELAL | 🇮🇱 TASE | ⏱️ Tactical (1-3 Days)

• 🎯 Trigger: Foreign carriers suspend flights following regional security alert
• ⚙️ Transmission: Eliminates 40% of seat competition, handing domestic carrier monopoly pricing power and surging passenger yields
• 💡 Unpriced Edge: Anticipatory window before quarterly passenger load filings discount the margin expansion
• 🛡️ Invalidation: Rapid security resolution and immediate foreign carrier route resumption
🔗 Reuters World Wire | TheMarker
```

### 4.6 Feed Additions (`config/sources.yaml`)

Add targeted high-alpha breaking feeds:
* **Global Geopolitical Feeds**:
  - `Reuters World Wire`: `https://news.google.com/rss/search?q=site:reuters.com/world+when:1d&hl=en-US&gl=US&ceid=US:en`
  - `Google News World`: `https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-US&gl=US&ceid=US:en`
* **Israel Breaking Feeds (מבזקים וביטחון)**:
  - `Ynet Breaking News (מבזקי חדשות)`: `https://www.ynet.co.il/Integration/StoryRss1854.xml`
  - `Ynet News English Breaking`: `https://www.ynetnews.com/Integration/StoryRss3082.xml`
  - `Times of Israel News`: `https://www.timesofisrael.com/feed/`

---

## 5. Error Handling & Edge Cases

1. **Missing or Incomplete Causal Chain**: If an incoming article is a standard financial 8-K filing or financial release that doesn't fit the geopolitical 2nd-order template, the system gracefully falls back to the standard `MarketImpactAnalysis` pipeline without errors.
2. **Ambiguous Horizon**: Defaults to `TACTICAL_1_3_DAYS` if transmission is immediate or event-driven; `SWING_2_4_WEEKS` if earnings/policy-driven.
3. **Invalid Ticker Output**: Ticker validator automatically maps known Israeli dual-listed tickers or drops non-tradeable tokens.
4. **Rate Limiting & Cooldown**: Inherits the newly built domain cooldown tracker and retry-after logic to prevent 429 throttling on new feeds.

---

## 6. Testing & Verification Plan

1. **Unit Tests (`news-agent/tests/test_causal_engine.py`)**:
   - `test_post_facto_recap_is_blocked_at_gate`: Verifies that a stock surging 11% is flagged `POST_FACTO_RECAP` and fails the opportunity gate.
   - `test_predictive_anticipatory_boosts_score`: Verifies that an event like flight disruption receives `PREDICTIVE_ANTICIPATORY` status and an opportunity score boost.
   - `test_tase_ticker_support`: Verifies that `$ELAL` on TASE is preserved as a valid tradeable ticker and properly tagged.
   - `test_triage_prefilter_drops_noise`: Verifies that non-economic political/celebrity articles are dropped at triage without LLM invocation.
2. **Notification Tests (`news-agent/tests/test_notification.py`)**:
   - Verify that the Telegram card outputs the exchange badge (`🇮🇱 TASE` / `🇺🇸 US`), horizon badge (`⏱️ Tactical (1-3 Days)`), trigger bullet, and transmission bullet.
3. **Full Regression Suite**:
   - Run `PYTHONPATH=news-agent pytest news-agent/tests/` to guarantee all existing 420+ tests continue to pass.
