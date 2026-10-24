import os
import re
from pathlib import Path
from typing import Any, List, Optional
import yaml
from pydantic import BaseModel, Field, field_validator
from src.models import ImpactGrade, WatchlistSettings

def _load_env_files() -> None:
    """Robustly discover and load environment variables from .env.local and .env
    across the current working directory, news-agent directory, and repository root.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    search_dirs = [
        Path.cwd(),
        Path.cwd() / "news-agent",
        Path(__file__).resolve().parent.parent,  # news-agent root
        Path(__file__).resolve().parent.parent.parent,  # workspace / repository root
    ]

    seen = set()
    for d in search_dirs:
        for fname in [".env.local", ".env"]:
            p = (d / fname).resolve()
            if p.is_file() and p not in seen:
                seen.add(p)
                load_dotenv(p, override=False)


_load_env_files()


class SourceConfig(BaseModel):
    """Configuration definition for an individual news feed source."""
    name: str = Field(..., description="Display name of the news source.")
    type: str = Field(default="rss", description="Source type ('rss' or 'scraper').")
    url: str = Field(..., description="RSS/Atom feed URL or scraping endpoint.")
    enabled: bool = Field(default=True, description="Whether this feed is actively polled.")
    max_items: Optional[int] = Field(default=None, description="Per-source maximum articles fetched per cycle (overrides global batch_size_per_source).")
    max_article_age_minutes: Optional[int] = Field(default=None, description="Per-source maximum article age in minutes (overrides global max_article_age_minutes).")
    fallback_url: Optional[str] = Field(default=None, description="Fallback URL if primary feed fails, returns 0 entries, or is blocked.")


class SourcesContainer(BaseModel):
    """Container schema for parsing sources.yaml."""
    sources: List[SourceConfig] = Field(default_factory=list)


class AgentSettings(BaseModel):
    polling_interval_minutes: int = Field(default=15, description="Minutes between poll cycles in daemon mode.")
    batch_size_per_source: int = Field(default=15, description="Maximum articles fetched per feed per cycle.")
    max_article_age_minutes: Optional[int] = Field(default=15, description="Maximum age of articles in minutes to process (None or 0 to disable).")
    database_path: str = Field(default="data/agent.db", description="Path to SQLite database.")
    enable_etag_cache: bool = Field(default=True, description="Enable HTTP conditional GET caching for feeds.")
    extract_full_body: bool = Field(default=True, description="Fetch full article text for deep comprehension instead of RSS snippet only.")
    fetch_max_retries: int = Field(default=3, description="Maximum retry attempts on feed network failures.")
    fetch_retry_backoff: float = Field(default=1.0, description="Exponential backoff factor in seconds for feed fetch retries.")


class FilterSettings(BaseModel):
    min_confidence_score: int = Field(default=6, ge=1, le=10, description="Minimum score to trigger notification.")
    min_impact_grade: ImpactGrade = Field(default=ImpactGrade.HIGH, description="Minimum severity grade (e.g. HIGH, CRITICAL).")
    alert_on_all: bool = Field(default=False, description="If True, bypasses filter threshold for testing.")
    enable_fact_verification: bool = Field(default=False, description="Enforce strict primary/multi-source fact verification gate.")
    enable_batch_triage: bool = Field(default=True, description="Pre-screen headlines with low-token batch triage to optimize LLM calls & token usage.")

    @field_validator("min_impact_grade", mode="before")
    @classmethod
    def normalize_min_impact_grade(cls, v: Any) -> Any:
        if isinstance(v, str):
            cleaned = v.strip().upper()
            if cleaned == "MEDIUM":
                return ImpactGrade.AVERAGE
            try:
                return ImpactGrade(cleaned)
            except ValueError:
                return v
        return v


def _normalize_model_name(v: Any) -> Any:
    if not v or not isinstance(v, str):
        return v
    cleaned = v.strip()
    if not cleaned:
        return cleaned
    if " " in cleaned:
        cleaned = cleaned.lower().replace(" ", "-")
    if not cleaned.startswith("gemini/") and "gemini" in cleaned.lower():
        cleaned = f"gemini/{cleaned.lower()}"
    return cleaned


class LLMSettings(BaseModel):
    model: str = Field(default="gemini/gemini-3.8-flash", description="LiteLLM model identifier.")
    fallback_model: Optional[str] = Field(default="gemini/gemini-3.7-flash", description="Primary fallback LiteLLM model identifier.")
    fallback_models: List[str] = Field(
        default_factory=lambda: [
            "gemini/gemini-3.7-flash",
            "gemini/gemini-3.6-flash",
            "gpt-4o-mini",
        ],
        description="Ordered list of fallback models on rate limit or failure."
    )
    enable_cache: bool = Field(default=True, description="Enable local response caching.")
    max_content_length: int = Field(default=3500, description="Max characters of article body passed to LLM to prevent prompt token bloat.")
    monthly_budget_usd: Optional[float] = Field(default=10.0, description="Monthly budget limit in USD.")
    temperature: float = Field(default=0.1, description="Sampling temperature for deterministic analysis.")
    max_retries: int = Field(default=3, description="Maximum API retries upon rate limit or transient errors.")
    backoff_delay: float = Field(default=2.0, description="Initial retry backoff delay in seconds.")
    backoff_factor: float = Field(default=2.0, description="Exponential backoff factor.")

    @field_validator("model", "fallback_model", mode="before")
    @classmethod
    def normalize_model(cls, v: Any) -> Any:
        return _normalize_model_name(v)

    @field_validator("fallback_models", mode="before")
    @classmethod
    def normalize_fallback_models(cls, v: Any) -> List[str]:
        if not v:
            return []
        if isinstance(v, str):
            v = [m.strip() for m in v.split(",") if m.strip()]
        return [_normalize_model_name(m) for m in v if m]


class TelegramSettings(BaseModel):
    bot_token: Optional[str] = Field(default=None, description="Telegram bot authorization token.")
    chat_id: Optional[str] = Field(default=None, description="Destination Telegram chat or channel ID.")
    batch_notifications: bool = Field(default=True, description="Send multiple recommendations from the same cycle in 1 Telegram push.")

    @field_validator("bot_token", "chat_id", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s if s else None

    @property
    def is_configured(self) -> bool:
        """Return True if both bot_token and chat_id are present."""
        return bool(self.bot_token and self.chat_id)





class Settings(BaseModel):
    """Root application settings schema."""
    agent: AgentSettings = Field(default_factory=AgentSettings)
    filter: FilterSettings = Field(default_factory=FilterSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    telegram: TelegramSettings = Field(default_factory=TelegramSettings)
    watchlist: WatchlistSettings = Field(default_factory=WatchlistSettings)


def _expand_env_vars(data: Any) -> Any:
    """Recursively expands ${VAR_NAME} or ${VAR_NAME:-default} expressions."""
    if isinstance(data, dict):
        return {k: _expand_env_vars(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_expand_env_vars(item) for item in data]
    elif isinstance(data, str):
        exact_match = re.fullmatch(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}", data.strip())
        if exact_match:
            env_var = exact_match.group(1)
            default_val = exact_match.group(2) if exact_match.group(2) is not None else None
            val = os.environ.get(env_var, default_val)
            if val is None or val == "":
                return None
            return val

        pattern = re.compile(r"\$\{([A-Za-z0-9_]+)(?::-([^}]*))?\}")
        def replace(match: re.Match) -> str:
            env_var = match.group(1)
            default_val = match.group(2) if match.group(2) is not None else ""
            return os.environ.get(env_var, default_val)
        expanded = pattern.sub(replace, data)
        return expanded if expanded != "" else None
    return data


def _find_default_config_path(filename: str) -> Path:
    """Locates the default config file in candidate standard locations."""
    # Location 1: news-agent/config/<filename> relative to this file
    base_dir = Path(__file__).resolve().parent.parent
    candidate = base_dir / "config" / filename
    if candidate.exists():
        return candidate

    # Location 2: Current working directory
    cwd_candidate = Path.cwd() / "news-agent" / "config" / filename
    if cwd_candidate.exists():
        return cwd_candidate

    cwd_config = Path.cwd() / "config" / filename
    if cwd_config.exists():
        return cwd_config

    return candidate


def load_settings(config_path: Optional[str] = None) -> Settings:
    """Loads and validates application settings from YAML with env var substitution."""
    _load_env_files()
    path = Path(config_path) if config_path else _find_default_config_path("settings.yaml")
    if not path.exists():
        return Settings()

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    processed_data = _expand_env_vars(raw_data)
    settings = Settings.model_validate(processed_data)

    # Normalize relative database_path to news-agent root if news-agent/data exists
    if settings.agent and settings.agent.database_path:
        db_p = Path(settings.agent.database_path)
        if not db_p.is_absolute():
            news_agent_db = Path(__file__).resolve().parent.parent / db_p
            if news_agent_db.exists() or (not db_p.exists() and news_agent_db.parent.exists()):
                settings.agent.database_path = str(news_agent_db)

    return settings


def load_sources(sources_path: Optional[str] = None) -> List[SourceConfig]:
    """Loads and validates configured news feed sources from YAML."""
    path = Path(sources_path) if sources_path else _find_default_config_path("sources.yaml")
    if not path.exists():
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw_data = yaml.safe_load(f) or {}

    processed_data = _expand_env_vars(raw_data)
    container = SourcesContainer.model_validate(processed_data)
    return container.sources
