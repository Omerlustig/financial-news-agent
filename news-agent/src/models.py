import hashlib
from datetime import datetime
from enum import Enum
from typing import Any, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ImpactGrade(str, Enum):
    """Categorical severity grade for financial news catalyst impact."""
    LOW = "LOW"
    AVERAGE = "AVERAGE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper()
            if normalized == "MEDIUM":
                return cls.AVERAGE
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class EventStatus(str, Enum):
    """Institutional workflow status determining publication and notification actions."""
    OPPORTUNITY_CANDIDATE = "OPPORTUNITY_CANDIDATE"
    WATCHLIST = "WATCHLIST"
    INFORMATIONAL = "INFORMATIONAL"
    UNVERIFIED = "UNVERIFIED"
    NO_ACTION = "NO_ACTION"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class DirectionalStance(str, Enum):
    """Directional market stance inferred from catalyst economic mechanism."""
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    VOLATILE = "VOLATILE"
    NEUTRAL = "NEUTRAL"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class TimeHorizon(str, Enum):
    """Expected catalyst transmission and propagation horizon."""
    INTRADAY = "INTRADAY"
    SHORT_TERM = "SHORT_TERM"      # 1-7 days
    MEDIUM_TERM = "MEDIUM_TERM"    # 2-8 weeks
    LONG_TERM = "LONG_TERM"        # months+

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_").replace("-", "_")
            if "SHORT" in normalized or "1_7_DAY" in normalized:
                return cls.SHORT_TERM
            if "MEDIUM" in normalized or "2_8_WEEK" in normalized:
                return cls.MEDIUM_TERM
            if "LONG" in normalized or "MONTH" in normalized:
                return cls.LONG_TERM
            if "INTRADAY" in normalized:
                return cls.INTRADAY
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class SECFilingItem(str, Enum):
    """SEC 8-K item classifications for regulatory filings."""
    ITEM_1_01_MATERIAL_AGREEMENT = "Item 1.01: Entry into Material Definitive Agreement (M&A / Partnership)"
    ITEM_1_02_TERMINATION_AGREEMENT = "Item 1.02: Termination of a Material Agreement"
    ITEM_2_02_EARNINGS_OPERATIONS = "Item 2.02: Results of Operations and Financial Condition"
    ITEM_4_02_ACCOUNTING_RESTATEMENT = "Item 4.02: Non-Reliance on Previously Issued Financial Statements (Restatement)"
    ITEM_5_02_EXECUTIVE_CHANGE = "Item 5.02: Departure / Election of Key Officers or Directors"
    ITEM_GENERAL = "General Corporate Disclosure"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value.upper() == normalized or member.name.upper() == normalized:
                    return member
        return super()._missing_(value)


class SurgeLevel(str, Enum):
    """Corroboration surge velocity levels."""
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    SURGE_BREAKING = "SURGE_BREAKING"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper()
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class SentimentLabel(str, Enum):
    """Categorical directional sentiment polarity."""
    VERY_BEARISH = "VERY_BEARISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"
    BULLISH = "BULLISH"
    VERY_BULLISH = "VERY_BULLISH"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class TrendVerdict(str, Enum):
    """Multi-horizon momentum and market sentiment trend verdict."""
    STRONG_BULLISH = "STRONG_BULLISH"
    MODERATE_BULLISH = "MODERATE_BULLISH"
    BULLISH_REVERSAL = "BULLISH_REVERSAL"
    NEUTRAL_CONSOLIDATION = "NEUTRAL_CONSOLIDATION"
    MODERATE_BEARISH = "MODERATE_BEARISH"
    BEARISH_BREAKDOWN = "BEARISH_BREAKDOWN"
    PANIC_CAPITULATION = "PANIC_CAPITULATION"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class StatementType(str, Enum):
    """Categorical classification of financial statement verification rigor."""
    VERIFIED_FACT = "VERIFIED_FACT"
    ATTRIBUTED_CLAIM = "ATTRIBUTED_CLAIM"
    ANALYST_OPINION = "ANALYST_OPINION"
    ANONYMOUS_RUMOR = "ANONYMOUS_RUMOR"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class ProvenanceLevel(str, Enum):
    """Hierarchical tiering of information source authority and verification depth."""
    PRIMARY_OFFICIAL = "PRIMARY_OFFICIAL"
    TIER1_FINANCIAL = "TIER1_FINANCIAL"
    COMMERCIAL_NEWS = "COMMERCIAL_NEWS"
    UNVERIFIED_BLOG = "UNVERIFIED_BLOG"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)


class NoveltyState(str, Enum):
    """Tri-state novelty and market expectation gap classification."""
    GENUINE_SURPRISE = "GENUINE_SURPRISE"
    IN_LINE_CONFIRMATION = "IN_LINE_CONFIRMATION"
    PRICED_IN_OR_RECYCLED = "PRICED_IN_OR_RECYCLED"

    def __str__(self) -> str:
        return self.value

    @classmethod
    def _missing_(cls, value: object) -> Any:
        if isinstance(value, str):
            normalized = value.strip().upper().replace(" ", "_")
            for member in cls:
                if member.value == normalized:
                    return member
        return super()._missing_(value)



import urllib.parse

TRACKING_QUERY_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "ref", "fbclid", "gclid", "source", "ncid", "guccounter", "guce_referrer"
}


def canonicalize_url(url: str) -> str:
    """Normalize and canonicalize URL by stripping tracking parameters, fragments, and trailing slashes."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlsplit(url.strip())
        if parsed.query:
            query_pairs = urllib.parse.parse_qsl(parsed.query, keep_blank_values=False)
            filtered = [(k, v) for k, v in query_pairs if k.lower() not in TRACKING_QUERY_PARAMS]
            clean_query = urllib.parse.urlencode(filtered)
        else:
            clean_query = ""
        path = parsed.path
        if len(path) > 1 and path.endswith("/"):
            path = path[:-1]
        clean_url = urllib.parse.urlunsplit((parsed.scheme.lower(), parsed.netloc.lower(), path, clean_query, ""))
        return clean_url
    except Exception:
        return url.strip()


class RawArticle(BaseModel):
    """Normalized ingested news item before deduplication and intelligence analysis."""
    title: str = Field(..., description="Headline of the article.")
    url: str = Field(..., description="Canonical source URL.")
    source_name: str = Field(..., description="Publisher name (e.g. 'Reuters', 'CNBC').")
    published_at: Optional[datetime] = Field(default=None, description="Publication timestamp in UTC.")
    summary: str = Field(default="", description="Snippet or brief description from the feed.")
    content: Optional[str] = Field(default=None, description="Extracted full body text if available.")
    guid: Optional[str] = Field(default=None, description="Unique ID from RSS/Atom entry if provided.")

    @field_validator("url", mode="before")
    @classmethod
    def canonicalize_article_url(cls, v: Any) -> str:
        if isinstance(v, str):
            return canonicalize_url(v)
        return str(v)

    @property
    def content_hash(self) -> str:
        """Deterministic SHA-256 hash derived from the article URL and headline."""
        key = f"{self.url.strip()}::{self.title.strip()}".encode("utf-8")
        return hashlib.sha256(key).hexdigest()


class CausalChain(BaseModel):
    """Structured causal propagation breakdown from facts to ticker impacts."""
    event_facts: str = Field(..., description="Key verifiable factual events grounded in source articles.")
    economic_mechanism: str = Field(..., description="Causal transmission mechanism explaining how facts impact revenue, margins, or operations.")
    affected_sectors: List[str] = Field(default_factory=list, description="Market sectors directly or second-order affected.")
    beneficiaries: List[str] = Field(default_factory=list, description="Public company tickers positively exposed with '$' prefix.")
    negatively_exposed: List[str] = Field(default_factory=list, description="Public company tickers negatively exposed with '$' prefix.")
    counterarguments: str = Field(default="", description="Principal counterarguments, risks, headwinds, or offsetting factors.")
    invalidation_conditions: str = Field(default="", description="Measurable facts or catalysts that would disprove or invalidate this thesis.")

    @field_validator("beneficiaries", "negatively_exposed", mode="before")
    @classmethod
    def normalize_tickers(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",") if t.strip()]
        normalized = []
        for ticker in v:
            cleaned = str(ticker).strip().upper()
            if cleaned and not cleaned.startswith("$"):
                cleaned = f"${cleaned}"
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @field_validator("affected_sectors", mode="before")
    @classmethod
    def normalize_sectors(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [str(s).strip() for s in v if str(s).strip()]


class HardGates(BaseModel):
    """Quality gating evaluation before opportunity alert issuance."""
    is_novel: bool = Field(..., description="True if event represents new developments rather than stale/duplicate reporting.")
    is_verified: bool = Field(..., description="True if facts are corroborated by approved primary sources.")
    has_direct_exposure: bool = Field(..., description="True if named tickers have genuine direct operational or financial exposure.")
    passes_all_gates: bool = Field(
        default=False,
        description="True only if novelty, verification, and direct exposure hard gates all pass."
    )

    @model_validator(mode="before")
    @classmethod
    def compute_passes_all_gates(cls, data: Any) -> Any:
        if isinstance(data, dict):
            novel = bool(data.get("is_novel", False))
            verified = bool(data.get("is_verified", False))
            direct = bool(data.get("has_direct_exposure", False))
            all_passed = novel and verified and direct
            if "passes_all_gates" not in data or data.get("passes_all_gates") is None:
                data["passes_all_gates"] = all_passed
            else:
                data["passes_all_gates"] = bool(data["passes_all_gates"]) and all_passed
        return data

    @model_validator(mode="after")
    def validate_passes_all_gates(self) -> "HardGates":
        if not (self.is_novel and self.is_verified and self.has_direct_exposure):
            self.passes_all_gates = False
        return self


class TokenUsage(BaseModel):
    """LLM token telemetry and cost estimation metadata."""
    prompt_tokens: int = Field(default=0, description="Prompt/input tokens consumed.")
    completion_tokens: int = Field(default=0, description="Completion/output tokens generated.")
    total_tokens: int = Field(default=0, description="Total tokens consumed.")
    estimated_cost_usd: float = Field(default=0.0, description="Estimated total cost in USD.")
    model_name: str = Field(default="", description="Identifier of the model used.")


class SentimentScore(BaseModel):
    """Compound sentiment polarity score and probabilistic distribution."""
    score: float = Field(default=0.0, ge=-1.0, le=1.0, description="Compound sentiment score from -1.0 (extreme bearish) to +1.0 (extreme bullish)")
    label: SentimentLabel = Field(default=SentimentLabel.NEUTRAL)
    positive: float = Field(default=0.0, ge=0.0, le=1.0)
    negative: float = Field(default=0.0, ge=0.0, le=1.0)
    neutral: float = Field(default=1.0, ge=0.0, le=1.0)
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)

    @field_validator("label", mode="before")
    @classmethod
    def normalize_label(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return SentimentLabel(cleaned)
            except ValueError:
                return v
        return v


class TrendMetrics(BaseModel):
    """Short-term volume acceleration and sentiment velocity tracking."""
    buzz_ratio: float = Field(default=1.0, description="Current 24h volume relative to baseline")
    sentiment_velocity: float = Field(default=0.0, description="Delta in sentiment score vs previous window")
    sentiment_acceleration: float = Field(default=0.0, description="Second derivative of sentiment (rate of change of velocity)")
    is_divergent: bool = Field(default=False, description="True if buzz volume diverges significantly from sentiment direction")
    divergence_type: Optional[str] = Field(default=None, description="Type of divergence: 'BEARISH_PANIC_LIQUIDATION', 'BULLISH_ACCUMULATION_DIVERGENCE', or None")
    trend_verdict: TrendVerdict = Field(default=TrendVerdict.NEUTRAL_CONSOLIDATION)
    articles_24h_count: int = Field(default=1)

    @field_validator("trend_verdict", mode="before")
    @classmethod
    def normalize_trend_verdict(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return TrendVerdict(cleaned)
            except ValueError:
                return v
        return v


class SectorTrendMetrics(BaseModel):
    """Sector-level sentiment momentum and buzz tracking metrics."""
    sector: str = Field(..., description="Market sector name (e.g. 'Semiconductors')")
    avg_sentiment: float = Field(default=0.0, description="Average compound sentiment score across sector articles")
    sentiment_velocity: float = Field(default=0.0, description="Delta in sector sentiment vs prior window")
    sentiment_acceleration: float = Field(default=0.0, description="Rate of change in sector velocity")
    buzz_ratio: float = Field(default=1.0, description="Current volume relative to sector baseline")
    article_count: int = Field(default=1, description="Number of sector articles in analysis window")
    sentiment_label: SentimentLabel = Field(default=SentimentLabel.NEUTRAL)
    trend_verdict: TrendVerdict = Field(default=TrendVerdict.NEUTRAL_CONSOLIDATION)

    @field_validator("sentiment_label", mode="before")
    @classmethod
    def normalize_sentiment_label(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return SentimentLabel(cleaned)
            except ValueError:
                return v
        return v

    @field_validator("trend_verdict", mode="before")
    @classmethod
    def normalize_trend_verdict(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return TrendVerdict(cleaned)
            except ValueError:
                return v
class WatchlistSettings(BaseModel):
    """Configuration for user watchlist tickers and priority sectors."""
    enabled: bool = Field(default=True, description="Enable watchlist matching and priority scoring boost.")
    tickers: List[str] = Field(default_factory=list, description="Watched tickers e.g. ['$NVDA', '$AAPL', '$TSLA']")
    sectors: List[str] = Field(default_factory=list, description="Watched priority sectors e.g. ['Aviation', 'Defense', 'Semiconductors']")
    score_boost: int = Field(default=1, description="Confidence score boost applied to matched watchlist items (1-2)")

    @field_validator("tickers", mode="before")
    @classmethod
    def normalize_tickers(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",") if t.strip()]
        normalized = []
        for ticker in v:
            cleaned = str(ticker).strip().upper()
            if cleaned and not cleaned.startswith("$"):
                cleaned = f"${cleaned}"
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @field_validator("sectors", mode="before")
    @classmethod
    def normalize_sectors(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            return [s.strip() for s in v.split(",") if s.strip()]
        return [str(s).strip() for s in v if str(s).strip()]


class TelegramButton(BaseModel):
    """Inline URL button for Telegram notifications."""
    text: str = Field(..., description="Button display label text.")
    url: str = Field(..., description="Target URL opened upon button click.")


class TechnicalIndicatorSummary(BaseModel):
    """TradingView technical indicator metrics and confluence check."""
    symbol: str = Field(..., description="Ticker symbol (e.g. $NVDA)")
    recommendation: str = Field(default="NEUTRAL", description="Technical consensus: STRONG_BUY, BUY, NEUTRAL, SELL, STRONG_SELL")
    buy_votes: int = Field(default=0)
    sell_votes: int = Field(default=0)
    neutral_votes: int = Field(default=0)
    rsi: Optional[float] = Field(default=None, description="14-period Relative Strength Index")
    close_price: Optional[float] = Field(default=None, description="Current market / close price")
    sma50: Optional[float] = Field(default=None, description="50-day Simple Moving Average")
    sma200: Optional[float] = Field(default=None, description="200-day Simple Moving Average")
    confluence_note: Optional[str] = Field(default=None, description="Confluence check comparing news stance with technical setup")


class FourDimensionalSentiment(BaseModel):
    """Decomposed 4-dimensional sentiment and fundamental surprise metrics."""
    linguistic_sentiment: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Editorial and linguistic tone (-1.0 bearish to +1.0 bullish)."
    )
    fundamental_impact: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Direct impact on fundamental cash flows / earnings (-1.0 damage to +1.0 accretive)."
    )
    expectation_surprise: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Deviation from consensus expectations / guidance (-1.0 miss to +1.0 beat)."
    )
    price_implication: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description="Net directional price implication (-1.0 severe drop to +1.0 surge)."
    )
    surprise_verdict: str = Field(
        default="IN_LINE",
        description="Expectation gap verdict: BEAT, MISS, IN_LINE, UNEXPECTED_SHOCK."
    )

    @field_validator("linguistic_sentiment", "fundamental_impact", "expectation_surprise", "price_implication", mode="before")
    @classmethod
    def clamp_and_float(cls, v: Any) -> float:
        if v is None:
            return 0.0
        val = float(v)
        return max(-1.0, min(1.0, val))

    @field_validator("surprise_verdict", mode="before")
    @classmethod
    def normalize_surprise_verdict(cls, v: Any) -> str:
        if v is None:
            return "IN_LINE"
        return str(v).strip().upper()


class AdversarialReview(BaseModel):
    """Adversarial stress-test and falsification audit for news-derived thesis."""
    bearish_counter_thesis: str = Field(
        ...,
        description="Strongest counter-argument or bearish thesis challenging the consensus."
    )
    priced_in_risk: str = Field(
        ...,
        description="Evaluation of whether the catalyst is already reflected in valuation / expectations."
    )
    macro_confounding_risk: str = Field(
        ...,
        description="Risk of macro beta or market-wide confounding drivers instead of true alpha."
    )
    falsification_threshold: str = Field(
        ...,
        description="Concrete, measurable data point or event that invalidates the investment thesis."
    )
    is_thesis_robust: bool = Field(
        default=True,
        description="True if original thesis withstands adversarial audit."
    )


class TwoLayerGateDecision(BaseModel):
    """Decoupled quality and opportunity gating decision."""
    event_quality_score: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Evidence rigor, provenance, and corroboration score (0.0 to 10.0)."
    )
    opportunity_score: float = Field(
        default=5.0,
        ge=0.0,
        le=10.0,
        description="Asymmetric upside/downside opportunity score (0.0 to 10.0)."
    )
    passes_event_quality_gate: bool = Field(
        default=False,
        description="True if event quality score meets high-conviction threshold (e.g. >= 8.0)."
    )
    passes_opportunity_gate: bool = Field(
        default=False,
        description="True if opportunity score meets actionable threshold (e.g. >= 8.0)."
    )
    gate_verdict: str = Field(
        default="WATCHLIST",
        description="Decision gate verdict: ALERT, WATCHLIST, REJECT."
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Detailed reason if rejected or held back from alerting."
    )

    @field_validator("event_quality_score", "opportunity_score", mode="before")
    @classmethod
    def clamp_scores(cls, v: Any) -> float:
        if v is None:
            return 5.0
        val = float(v)
        return max(0.0, min(10.0, val))

    @field_validator("gate_verdict", mode="before")
    @classmethod
    def normalize_verdict(cls, v: Any) -> str:
        if v is None:
            return "WATCHLIST"
        return str(v).strip().upper()


class EventLineageRecord(BaseModel):
    """Persistent tracking record maintaining historical catalyst lineage and forward performance."""
    event_id: str = Field(..., description="Unique event lineage cluster identifier.")
    title: str = Field(..., description="Canonical representative event headline.")
    primary_ticker: Optional[str] = Field(default=None, description="Primary impacted ticker symbol.")
    first_seen: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp when first article for event was ingested.")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="UTC timestamp of most recent update / corroboration.")
    source_count: int = Field(default=1, description="Count of distinct sources reporting on this event.")
    all_sources: List[str] = Field(default_factory=list, description="List of source / publisher names covering this event.")
    event_quality_score: float = Field(default=0.0, description="Evaluated factual quality and evidence score.")
    opportunity_score: float = Field(default=0.0, description="Evaluated trade opportunity and asymmetry score.")
    status: str = Field(default="WATCHLIST", description="Workflow status (ALERT, WATCHLIST, REJECT).")
    initial_price: Optional[float] = Field(default=None, description="Asset price at the time of initial event detection.")
    subsequent_return_1d: Optional[float] = Field(default=None, description="Realized forward asset percentage return over 1 trading day.")
    subsequent_return_7d: Optional[float] = Field(default=None, description="Realized forward asset percentage return over 7 trading days.")

    @field_validator("primary_ticker", mode="before")
    @classmethod
    def normalize_primary_ticker(cls, v: Any) -> Optional[str]:
        if not v:
            return None
        cleaned = str(v).strip().upper()
        if cleaned and not cleaned.startswith("$"):
            cleaned = f"${cleaned}"
        return cleaned

    @field_validator("all_sources", mode="before")
    @classmethod
    def normalize_all_sources(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            try:
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return [str(s).strip() for s in parsed if str(s).strip()]
            except Exception:
                return [s.strip() for s in v.split(",") if s.strip()]
        if isinstance(v, (list, set, tuple)):
            return [str(s).strip() for s in v if str(s).strip()]
        return [str(v).strip()]


class MarketImpactAnalysis(BaseModel):
    """Institutional financial analyst reasoning and scoring output."""
    is_market_moving: bool = Field(
        ...,
        description="True if this news is a material catalyst likely to move stock prices or sectors."
    )
    confidence_score: int = Field(
        ...,
        ge=1,
        le=10,
        description="Confidence score from 1 (routine/negligible) to 10 (exceptional, high-conviction catalyst)."
    )
    impact_grade: ImpactGrade = Field(
        ...,
        description="Categorical grade: LOW (1-3), AVERAGE (4-6), HIGH (7-8), CRITICAL (9-10)."
    )
    affected_tickers: List[str] = Field(
        default_factory=list,
        description="Specific stock or ETF tickers impacted with '$' prefix (e.g. ['$NVDA', '$AMD', '$SMH'])."
    )
    affected_sectors: List[str] = Field(
        default_factory=list,
        description="Market sectors impacted (e.g. ['Semiconductors', 'Defense', 'Cloud Computing'])."
    )
    summary: str = Field(
        ...,
        description="Concise 2-3 sentence executive brief summarizing what happened."
    )
    catalyst_thesis: str = Field(
        ...,
        description="Financial analyst thesis detailing the exact mechanism of market impact (revenue, margin, regulation, supply chain)."
    )
    status: EventStatus = Field(
        default=EventStatus.OPPORTUNITY_CANDIDATE,
        description="Decision status gate (OPPORTUNITY_CANDIDATE, WATCHLIST, INFORMATIONAL, UNVERIFIED, NO_ACTION)."
    )
    directional_stance: DirectionalStance = Field(
        default=DirectionalStance.NEUTRAL,
        description="Expected directional market stance (BULLISH, BEARISH, VOLATILE, NEUTRAL)."
    )
    time_horizon: Optional[str] = Field(
        default=None,
        description="Expected catalyst time horizon (INTRADAY, SHORT_TERM, MEDIUM_TERM, LONG_TERM)."
    )
    causal_chain: Optional[CausalChain] = Field(
        default=None,
        description="Structured causal propagation breakdown from facts to ticker impacts."
    )
    hard_gates: Optional[HardGates] = Field(
        default=None,
        description="Quality gating evaluation (novelty, verification, direct exposure)."
    )
    token_usage: Optional[TokenUsage] = Field(
        default=None,
        description="LLM token usage and estimated cost tracking."
    )
    recommendation: Optional[str] = Field(
        default=None,
        description="Concise 1-sentence actionable analyst recommendation or monitoring posture (e.g. 'Accumulate $NVDA on dips; hedge long exposure with $SMH puts; avoid $INTC until next quarter')."
    )
    corroborating_sources: List[str] = Field(
        default_factory=list,
        description="List of distinct publisher names corroborating this event."
    )
    corroboration_count: int = Field(
        default=1,
        description="Number of distinct sources reporting this catalyst."
    )
    sec_filing_item: Optional[str] = Field(
        default=None,
        description="SEC 8-K regulatory item classification if applicable."
    )
    surge_level: SurgeLevel = Field(
        default=SurgeLevel.NORMAL,
        description="Corroboration surge velocity level (NORMAL, ELEVATED, SURGE_BREAKING)."
    )
    interactive_buttons: List[TelegramButton] = Field(
        default_factory=list,
        description="Interactive action buttons attached to notification alert."
    )
    sentiment: SentimentScore = Field(
        default_factory=SentimentScore,
        description="Compound sentiment score and probabilistic distribution."
    )
    trend: Optional[TrendMetrics] = Field(
        default=None,
        description="Short-term volume buzz ratio and sentiment velocity trend metrics."
    )
    is_watchlist_match: bool = Field(
        default=False,
        description="True if catalyst matches user watchlist tickers or priority sectors",
    )
    matched_watchlist_items: List[str] = Field(
        default_factory=list,
        description="List of matched tickers or sectors from user watchlist",
    )
    sector_trends: List[SectorTrendMetrics] = Field(
        default_factory=list,
        description="Sector sentiment momentum metrics for affected sectors",
    )
    technicals: List[TechnicalIndicatorSummary] = Field(
        default_factory=list,
        description="TradingView technical indicators for primary affected tickers",
    )
    disclaimer: str = Field(
        default="⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.",
        description="Mandatory compliance disclaimer separating news inference from trading validation."
    )
    is_llm_analyzed: bool = Field(
        default=True,
        description="True if analysis was successfully generated by an LLM; False if generated by fallback rule engine upon error/rate limit."
    )
    four_d_sentiment: Optional[FourDimensionalSentiment] = Field(
        default=None,
        description="Decomposed 4-dimensional sentiment (linguistic, fundamental, expectation surprise, price implication)."
    )
    adversarial_review: Optional[AdversarialReview] = Field(
        default=None,
        description="Adversarial stress-test, counter-thesis, priced-in risk, and falsification threshold."
    )
    two_layer_gate: Optional[TwoLayerGateDecision] = Field(
        default=None,
        description="Decoupled two-layer gating evaluation (event quality gate and opportunity gate)."
    )
    novelty_state: Optional[NoveltyState] = Field(
        default=None,
        description="Catalyst novelty classification (GENUINE_SURPRISE, IN_LINE_CONFIRMATION, PRICED_IN_OR_RECYCLED)."
    )
    provenance_level: Optional[ProvenanceLevel] = Field(
        default=None,
        description="Source provenance tier (PRIMARY_OFFICIAL, TIER1_FINANCIAL, COMMERCIAL_NEWS, UNVERIFIED_BLOG)."
    )
    event_id: Optional[str] = Field(
        default=None,
        description="Linked persistent event lineage identifier."
    )

    @field_validator("four_d_sentiment", mode="before")
    @classmethod
    def normalize_four_d_sentiment(cls, v: Any) -> Optional[FourDimensionalSentiment]:
        if v is None:
            return None
        if isinstance(v, dict):
            return FourDimensionalSentiment(**v)
        return v

    @field_validator("adversarial_review", mode="before")
    @classmethod
    def normalize_adversarial_review(cls, v: Any) -> Optional[AdversarialReview]:
        if v is None:
            return None
        if isinstance(v, dict):
            return AdversarialReview(**v)
        return v

    @field_validator("two_layer_gate", mode="before")
    @classmethod
    def normalize_two_layer_gate(cls, v: Any) -> Optional[TwoLayerGateDecision]:
        if v is None:
            return None
        if isinstance(v, dict):
            return TwoLayerGateDecision(**v)
        return v

    @field_validator("novelty_state", mode="before")
    @classmethod
    def normalize_novelty_state(cls, v: Any) -> Optional[NoveltyState]:
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return NoveltyState(cleaned)
            except ValueError:
                return None
        return v

    @field_validator("provenance_level", mode="before")
    @classmethod
    def normalize_provenance_level(cls, v: Any) -> Optional[ProvenanceLevel]:
        if v is None:
            return None
        if isinstance(v, str):
            cleaned = v.strip().upper().replace(" ", "_")
            try:
                return ProvenanceLevel(cleaned)
            except ValueError:
                return None
        return v

    @field_validator("sentiment", mode="before")
    @classmethod
    def normalize_sentiment(cls, v: Any) -> Any:
        if v is None:
            return SentimentScore()
        if isinstance(v, dict):
            return SentimentScore(**v)
        return v

    @field_validator("trend", mode="before")
    @classmethod
    def normalize_trend(cls, v: Any) -> Any:
        if v is None:
            return None
        if isinstance(v, dict):
            return TrendMetrics(**v)
        return v

    @field_validator("sector_trends", mode="before")
    @classmethod
    def normalize_sector_trends(cls, v: Any) -> List[SectorTrendMetrics]:
        if not v:
            return []
        if isinstance(v, list):
            res = []
            for item in v:
                if isinstance(item, dict):
                    res.append(SectorTrendMetrics(**item))
                else:
                    res.append(item)
            return res
        return v

    @field_validator("sec_filing_item", mode="before")
    @classmethod
    def normalize_sec_filing_item(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, SECFilingItem):
            return v.value
        return str(v)

    @field_validator("surge_level", mode="before")
    @classmethod
    def normalize_surge_level(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            try:
                return SurgeLevel(cleaned)
            except ValueError:
                return v
        return v

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            try:
                return EventStatus(cleaned)
            except ValueError:
                return v
        return v

    @field_validator("directional_stance", mode="before")
    @classmethod
    def normalize_directional_stance(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            try:
                return DirectionalStance(cleaned)
            except ValueError:
                return v
        return v

    @field_validator("time_horizon", mode="before")
    @classmethod
    def normalize_time_horizon(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        if isinstance(v, TimeHorizon):
            return v.value
        if isinstance(v, str):
            try:
                return TimeHorizon(v).value
            except ValueError:
                return v.strip()
        return str(v)

    @field_validator("impact_grade", mode="before")
    @classmethod
    def normalize_impact_grade(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned == "MEDIUM":
                return ImpactGrade.AVERAGE
            try:
                return ImpactGrade(cleaned)
            except ValueError:
                return v
        return v

    @field_validator("affected_tickers", mode="before")
    @classmethod
    def normalize_tickers(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [t.strip() for t in v.split(",") if t.strip()]
        normalized = []
        for ticker in v:
            cleaned = str(ticker).strip().upper()
            if cleaned and not cleaned.startswith("$"):
                cleaned = f"${cleaned}"
            if cleaned:
                normalized.append(cleaned)
        return normalized

    @model_validator(mode="after")
    def enforce_hard_gates_rule(self) -> "MarketImpactAnalysis":
        """Ensures that failed verification or indirect exposure blocks OPPORTUNITY_CANDIDATE."""
        if self.hard_gates is not None and not self.hard_gates.passes_all_gates:
            if self.status == EventStatus.OPPORTUNITY_CANDIDATE:
                if not self.hard_gates.is_verified:
                    self.status = EventStatus.UNVERIFIED
                else:
                    self.status = EventStatus.WATCHLIST
        return self

