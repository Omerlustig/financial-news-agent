import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from src.models import RawArticle

logger = logging.getLogger(__name__)

DEFAULT_NOISE_PATTERNS = [
    r"\b\d+\s+stocks?\s+to\s+buy\b",
    r"\bmotley\s+fool\b",
    r"\bzacks\s+investment\b",
    r"\bsponsored\b",
    r"\bpartner\s+content\b",
    r"\bpromoted\s+post\b",
    r"\bwhere\s+to\s+invest\s+\$?\d+",
    r"\bbefore\s+you\s+buy\s+stock\b",
    r"\bdividend\s+stocks?\s+to\s+retire\b",
]

DEFAULT_TICKER_BLACKLIST = {
    "CEO", "CFO", "COO", "CTO", "AI", "EV", "US", "FED", "GDP",
    "SEC", "FDA", "IPO", "ETF", "CAGR", "EPS", "PE", "COVID",
    "WSJ", "PR", "LLC", "INC", "CORP", "USA", "UK", "EU", "IMF",
    "FOMC", "CPI", "PPI", "PMI", "DOW", "NASDAQ", "NYSE", "S&P"
}


@dataclass
class PreFilterConfig:
    """Configuration for raw article pre-filtering and noise suppression."""
    enabled: bool = True
    drop_sponsored: bool = True
    min_title_length: int = 15
    noise_patterns: List[str] = field(default_factory=lambda: list(DEFAULT_NOISE_PATTERNS))
    ticker_blacklist: Set[str] = field(default_factory=lambda: set(DEFAULT_TICKER_BLACKLIST))


class NewsPreFilter:
    """Pre-filters raw news articles to drop promotional noise and clickbait,
    and removes common acronym collisions from extracted ticker symbols.
    """

    def __init__(self, config: Optional[PreFilterConfig] = None) -> None:
        self.config = config or PreFilterConfig()
        self._compiled_patterns = [
            re.compile(p, re.IGNORECASE) for p in self.config.noise_patterns
        ]

    def is_noise(self, article: RawArticle) -> bool:
        """Determines if a raw article matches known clickbait, sponsored content,
        or fails minimal title length constraints.
        """
        if not self.config.enabled:
            return False

        title = (article.title or "").strip()
        if len(title) < self.config.min_title_length:
            return True

        if self.config.drop_sponsored:
            for pattern in self._compiled_patterns:
                if pattern.search(title):
                    logger.debug("Dropped noise article: '%s' (matched %s)", title, pattern.pattern)
                    return True

        return False

    def filter_tickers(self, tickers: List[str]) -> List[str]:
        """Strips out false-positive ticker collisions matching macro/regulatory acronyms."""
        if not tickers:
            return []
        cleaned: List[str] = []
        for t in tickers:
            symbol = t.strip().upper()
            if symbol and symbol not in self.config.ticker_blacklist:
                cleaned.append(symbol)
        return cleaned
