"""HTML scraper fetcher for financial news aggregators and curated websites."""

from datetime import datetime, timezone
import logging
from typing import List, Optional, Set
from urllib.parse import urlparse
import requests
from bs4 import BeautifulSoup

from src.config import SourceConfig
from src.ingestion.base import BaseFetcher
from src.models import RawArticle

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9,he;q=0.8",
}


class HTMLFetcher(BaseFetcher):
    """Fetcher for HTML-based news aggregators (such as FinURLs) and websites without native RSS."""

    def __init__(
        self,
        source_config: SourceConfig,
        timeout: float = 15.0,
        max_items: Optional[int] = None,
    ) -> None:
        super().__init__(source_config)
        self.timeout = timeout
        self.max_items = max_items or source_config.max_items

    def fetch(self) -> List[RawArticle]:
        """Fetch and parse headline links from the configured HTML aggregator page."""
        url = self.source_config.url
        logger.info("Fetching HTML news source '%s' from %s", self.source_config.name, url)

        try:
            response = requests.get(
                url,
                headers=DEFAULT_HEADERS,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except Exception as exc:
            logger.warning("Failed fetching HTML news source '%s' (%s): %s", self.source_config.name, url, exc)
            return []

        soup = BeautifulSoup(response.text, "html.parser")
        parsed_origin = urlparse(url)
        origin_netloc = parsed_origin.netloc.replace("www.", "").lower()

        seen_urls: Set[str] = set()
        articles: List[RawArticle] = []
        now_utc = datetime.now(timezone.utc)

        for a in soup.find_all("a", href=True):
            href = a["href"].strip()
            text = a.get_text(strip=True)

            if not href.startswith("http"):
                continue
            if len(text) < 15:
                continue

            dest_netloc = urlparse(href).netloc.replace("www.", "").lower()
            if origin_netloc in dest_netloc:
                # Skip internal navigation links to self
                continue

            if href in seen_urls:
                continue
            seen_urls.add(href)

            articles.append(
                RawArticle(
                    title=text,
                    url=href,
                    source_name=f"{self.source_config.name} ({dest_netloc})",
                    published_at=now_utc,
                    summary=text,
                )
            )

            if self.max_items and len(articles) >= self.max_items:
                break

        logger.info("Ingested %d articles from HTML source '%s'", len(articles), self.source_config.name)
        return articles
