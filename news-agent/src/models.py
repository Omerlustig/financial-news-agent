import hashlib
from datetime import datetime
from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class ImpactGrade(str, Enum):
    """Categorical severity grade for financial news catalyst impact."""
    LOW = "LOW"
    AVERAGE = "AVERAGE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


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

    @field_validator("affected_tickers", mode="before")
    @classmethod
    def normalize_tickers(cls, v: List[str]) -> List[str]:
        if not v:
            return []
        normalized = []
        for ticker in v:
            cleaned = str(ticker).strip().upper()
            if cleaned and not cleaned.startswith("$"):
                cleaned = f"${cleaned}"
            if cleaned:
                normalized.append(cleaned)
        return normalized
