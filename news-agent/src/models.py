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


class RawArticle(BaseModel):
    """Normalized ingested news item before deduplication and intelligence analysis."""
    title: str = Field(..., description="Headline of the article.")
    url: str = Field(..., description="Canonical source URL.")
    source_name: str = Field(..., description="Publisher name (e.g. 'Reuters', 'CNBC').")
    published_at: Optional[datetime] = Field(default=None, description="Publication timestamp in UTC.")
    summary: str = Field(default="", description="Snippet or brief description from the feed.")
    content: Optional[str] = Field(default=None, description="Extracted full body text if available.")
    guid: Optional[str] = Field(default=None, description="Unique ID from RSS/Atom entry if provided.")

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
    disclaimer: str = Field(
        default="⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.",
        description="Mandatory compliance disclaimer separating news inference from trading validation."
    )

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

