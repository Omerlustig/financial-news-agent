from abc import ABC, abstractmethod
import random
import time
from typing import Dict, List, Optional
import urllib.parse
import requests

from src.config import SourceConfig
from src.models import RawArticle


class DomainCooldownTracker:
    """Thread-safe domain-level rate-limit and cooldown tracker."""
    _cooldowns: Dict[str, float] = {}

    @classmethod
    def get_host(cls, url: str) -> str:
        """Extract normalized hostname from URL."""
        try:
            return urllib.parse.urlsplit(url).netloc.lower()
        except Exception:
            return ""

    @classmethod
    def is_cooling_down(cls, url_or_host: str) -> bool:
        """Check if domain is currently in an active cooldown window."""
        host = cls.get_host(url_or_host) if "://" in url_or_host else url_or_host.lower()
        if not host:
            return False
        expiry = cls._cooldowns.get(host)
        if expiry is None:
            return False
        if time.time() < expiry:
            return True
        cls._cooldowns.pop(host, None)
        return False

    @classmethod
    def get_remaining_cooldown(cls, url_or_host: str) -> float:
        """Return remaining cooldown seconds for domain, or 0.0 if not cooling down."""
        host = cls.get_host(url_or_host) if "://" in url_or_host else url_or_host.lower()
        if not host:
            return 0.0
        expiry = cls._cooldowns.get(host)
        if expiry is None:
            return 0.0
        remaining = expiry - time.time()
        return max(0.0, remaining)

    @classmethod
    def set_cooldown(cls, url_or_host: str, seconds: float = 30.0) -> None:
        """Set a cooldown duration for the given domain."""
        host = cls.get_host(url_or_host) if "://" in url_or_host else url_or_host.lower()
        if host:
            cls._cooldowns[host] = time.time() + max(1.0, seconds)

    @classmethod
    def clear(cls) -> None:
        """Clear all active domain cooldowns (useful for tests)."""
        cls._cooldowns.clear()


def parse_retry_after(response: Optional[requests.Response], default: float = 2.0) -> float:
    """Extract and parse Retry-After header from response, clamping between 1.0 and 60.0s."""
    if response is None:
        return default
    raw = response.headers.get("Retry-After")
    if raw:
        try:
            val = float(raw)
            return min(max(1.0, val), 60.0)
        except (ValueError, TypeError):
            pass
    return default


def calculate_backoff(attempt: int, base_delay: float, retry_after: Optional[float] = None) -> float:
    """Calculate exponential backoff with randomized jitter and optional Retry-After constraint."""
    exp_delay = base_delay * (2 ** (attempt - 1))
    target_delay = retry_after if (retry_after is not None and retry_after > exp_delay) else exp_delay
    jittered = target_delay * random.uniform(0.85, 1.15)
    return max(0.5, jittered)


class BaseFetcher(ABC):
    """Abstract base class for all news source ingestion fetchers."""

    def __init__(self, source_config: SourceConfig) -> None:
        self.source_config = source_config

    @abstractmethod
    def fetch(self) -> List[RawArticle]:
        """Fetch and normalize articles from the configured source.

        Returns:
            List[RawArticle]: A list of parsed and normalized raw articles.
        """
        pass
