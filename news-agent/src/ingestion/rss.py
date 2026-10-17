import calendar
from datetime import datetime, timezone
import email.utils
import html
import logging
import re
import socket
import time
from typing import Any, List, Optional
import feedparser
import requests

try:
    from dateutil import parser as dateutil_parser
except ImportError:
    dateutil_parser = None

from src.config import SourceConfig
from src.ingestion.base import BaseFetcher
from src.ingestion.extractor import ArticleBodyExtractor
from src.models import RawArticle
from src.storage.base import BaseStorage

logger = logging.getLogger(__name__)

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36 (MarketNewsAgent/1.0)"
)
DEFAULT_TIMEOUT_SECONDS = 15.0


def clean_html(text: Optional[str]) -> str:
    """Strip HTML tags and decode HTML entities into normalized plain text.

    Args:
        text: Raw HTML string or None.

    Returns:
        Cleaned plain text without HTML tags and with normalized whitespace.
    """
    if not text:
        return ""
    # Strip <script> and <style> blocks along with their content first
    cleaned = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    # Replace block-level tags and line breaks with spaces
    cleaned = re.sub(r"(?i)<(br|p|div|li|tr|h[1-6])[^>]*>", " ", cleaned)
    # Strip all remaining tags
    cleaned = re.sub(r"<[^>]+>", "", cleaned)
    # Unescape HTML entities (&amp;, &lt;, &gt;, &quot;, &#39;, etc.)
    cleaned = html.unescape(cleaned)
    # Normalize consecutive whitespace
    return " ".join(cleaned.split())


def parse_datetime(raw_date: Any, struct_time: Optional[Any] = None) -> Optional[datetime]:
    """Parse various RSS/Atom date representations into a UTC datetime.

    Args:
        raw_date: Raw date string or object from feed entry.
        struct_time: Optional parsed struct_time provided by feedparser.

    Returns:
        Timezone-aware datetime in UTC, or None if parsing fails.
    """
    if struct_time:
        try:
            ts = calendar.timegm(struct_time)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception as e:
            logger.debug("Failed to convert struct_time %s: %s", struct_time, e)

    if not raw_date or not isinstance(raw_date, str):
        return None

    raw_date = raw_date.strip()
    if not raw_date:
        return None

    # Try RFC 2822 / RFC 822 format standard for RSS
    try:
        dt = email.utils.parsedate_to_datetime(raw_date)
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Try ISO 8601 standard for Atom
    try:
        iso_str = raw_date.replace("Z", "+00:00")
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        pass

    # Try dateutil parser if available
    if dateutil_parser is not None:
        try:
            dt = dateutil_parser.parse(raw_date)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt
        except Exception:
            pass

    logger.debug("Could not parse date string: %s", raw_date)
    return None


class RSSFetcher(BaseFetcher):
    """Fetcher for RSS and Atom news feeds using feedparser and HTTP conditional caching."""

    def __init__(
        self,
        source_config: SourceConfig,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        max_items: Optional[int] = None,
        max_article_age_minutes: Optional[int] = None,
        user_agent: str = USER_AGENT,
        storage: Optional[BaseStorage] = None,
        enable_etag_cache: bool = True,
        max_retries: int = 3,
        retry_backoff: float = 1.0,
        extract_full_body: bool = False,
    ) -> None:
        super().__init__(source_config)
        self.user_agent = user_agent
        self.timeout = timeout

        # Resolve effective max_items: source-specific override takes precedence over global
        effective_max_items = (
            getattr(source_config, "max_items", None)
            if getattr(source_config, "max_items", None) is not None
            else max_items
        )
        self.max_items = effective_max_items

        # Resolve effective max_article_age_minutes: source-specific override takes precedence over global
        effective_max_age = (
            getattr(source_config, "max_article_age_minutes", None)
            if getattr(source_config, "max_article_age_minutes", None) is not None
            else max_article_age_minutes
        )
        self.max_article_age_minutes = effective_max_age

        self.storage = storage
        self.enable_etag_cache = enable_etag_cache
        self.max_retries = max(1, max_retries)
        self.retry_backoff = max(0.0, retry_backoff)
        self.extract_full_body = extract_full_body

    def fetch(self) -> List[RawArticle]:
        """Fetch and parse articles from the configured RSS/Atom feed.

        Returns:
            List[RawArticle]: Successfully parsed articles from this feed.
        """
        url = self.source_config.url
        logger.info("Fetching RSS feed for source '%s' from %s", self.source_config.name, url)

        parsed = None
        raw_bytes: Optional[bytes] = None

        if (
            self.storage is not None
            and self.enable_etag_cache
            and isinstance(url, str)
            and url.startswith(("http://", "https://"))
        ):
            cached_etag, cached_last_modified = self.storage.get_feed_cache(url)
            headers = {
                "User-Agent": self.user_agent,
                "Accept": "application/rss+xml, application/xml, text/xml, */*",
                "Accept-Language": "en-US,en;q=0.9,he;q=0.8",
            }
            if cached_etag:
                headers["If-None-Match"] = cached_etag
            if cached_last_modified:
                headers["If-Modified-Since"] = cached_last_modified

            raw_bytes: Optional[bytes] = None
            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.get(url, headers=headers, timeout=self.timeout)

                    if response.status_code == 304:
                        logger.debug(
                            "Feed '%s' (%s) unchanged (HTTP 304 Not Modified); skipping parse.",
                            self.source_config.name,
                            url,
                        )
                        return []
                    elif response.status_code == 200:
                        new_etag = response.headers.get("ETag")
                        new_last_modified = response.headers.get("Last-Modified")
                        raw_bytes = response.content
                        parsed = feedparser.parse(raw_bytes)
                        if self.storage is not None:
                            self.storage.set_feed_cache(url, new_etag, new_last_modified)
                        break
                    elif response.status_code in (500, 502, 503, 504, 429) and attempt < self.max_retries:
                        sleep_time = self.retry_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Transient HTTP %d from '%s' (%s). Retrying %d/%d after %.1fs...",
                            response.status_code,
                            self.source_config.name,
                            url,
                            attempt + 1,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                        continue
                    else:
                        logger.warning(
                            "HTTP error %d fetching RSS feed '%s' (%s)",
                            response.status_code,
                            self.source_config.name,
                            url,
                        )
                        return []
                except (requests.exceptions.RequestException, socket.timeout) as exc:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff * (2 ** (attempt - 1))
                        logger.warning(
                            "Network error fetching RSS feed '%s' (%s): %s. Retrying %d/%d after %.1fs...",
                            self.source_config.name,
                            url,
                            exc,
                            attempt + 1,
                            self.max_retries,
                            sleep_time,
                        )
                        time.sleep(sleep_time)
                    else:
                        logger.warning(
                            "Network or parser error fetching RSS feed '%s' (%s) after %d retries: %s",
                            self.source_config.name,
                            url,
                            self.max_retries,
                            exc,
                        )
                        return []
        else:
            old_timeout = socket.getdefaulttimeout()
            for attempt in range(1, self.max_retries + 1):
                try:
                    socket.setdefaulttimeout(self.timeout)
                    parsed = feedparser.parse(
                        url,
                        agent=self.user_agent,
                        request_headers={"User-Agent": self.user_agent},
                    )
                    break
                except Exception as exc:
                    if attempt < self.max_retries:
                        sleep_time = self.retry_backoff * (2 ** (attempt - 1))
                        time.sleep(sleep_time)
                    else:
                        logger.warning(
                            "Network or parser error fetching RSS feed '%s' (%s) after %d retries: %s",
                            self.source_config.name,
                            url,
                            self.max_retries,
                            exc,
                        )
                        return []
                finally:
                    socket.setdefaulttimeout(old_timeout)

        if parsed is None:
            logger.warning("Feedparser returned None for '%s' (%s)", self.source_config.name, url)
            return []

        # Safe attribute/dict access for parsed response
        entries = getattr(parsed, "entries", None)
        if entries is None and isinstance(parsed, dict):
            entries = parsed.get("entries", [])
        if entries is None:
            entries = []

        is_bozo = getattr(parsed, "bozo", 0) or (parsed.get("bozo", 0) if isinstance(parsed, dict) else 0)
        bozo_exc = getattr(parsed, "bozo_exception", None) or (
            parsed.get("bozo_exception") if isinstance(parsed, dict) else None
        )

        if is_bozo and not entries and raw_bytes:
            try:
                text_clean = raw_bytes.decode("utf-8", errors="replace")
                sanitized_xml = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text_clean)
                re_parsed = feedparser.parse(sanitized_xml)
                re_entries = getattr(re_parsed, "entries", None) or []
                if re_entries:
                    parsed = re_parsed
                    entries = re_entries
                    is_bozo = getattr(re_parsed, "bozo", 0)
            except Exception:
                pass

        if is_bozo and not entries:
            logger.warning(
                "Malformed feed '%s' (%s) produced no entries: %s",
                self.source_config.name,
                url,
                bozo_exc,
            )
            return []
        elif is_bozo:
            logger.debug(
                "Feed '%s' flagged bozo warning (%s) but recovered %d entries",
                self.source_config.name,
                bozo_exc,
                len(entries),
            )

        if self.max_items is not None and self.max_items > 0:
            entries = entries[: self.max_items]

        articles: List[RawArticle] = []
        now_utc = datetime.now(timezone.utc)
        for entry in entries:
            try:
                article = self._parse_entry(entry)
                if article is not None:
                    # Enforce recency filter at fetch time if configured
                    if (
                        self.max_article_age_minutes is not None
                        and self.max_article_age_minutes > 0
                        and article.published_at is not None
                    ):
                        pub = (
                            article.published_at
                            if article.published_at.tzinfo is not None
                            else article.published_at.replace(tzinfo=timezone.utc)
                        )
                        age_min = (now_utc - pub).total_seconds() / 60.0
                        if age_min > self.max_article_age_minutes:
                            # Record stale article in storage if present so future cycles skip it immediately
                            if self.storage is not None:
                                try:
                                    if not self.storage.is_article_seen(article.content_hash):
                                        self.storage.record_article(article, alerted=False)
                                except Exception:
                                    pass
                            logger.debug(
                                "Skipping stale article '%s' (age %.1f min > %d min limit)",
                                article.title,
                                age_min,
                                self.max_article_age_minutes,
                            )
                            continue

                    articles.append(article)
            except Exception as exc:
                logger.debug(
                    "Error parsing individual entry in feed '%s': %s",
                    self.source_config.name,
                    exc,
                )

        logger.info("Fetched %d articles from '%s'", len(articles), self.source_config.name)
        return articles

    def _parse_entry(self, entry: Any) -> Optional[RawArticle]:
        """Parse and sanitize a single feed entry into a RawArticle."""
        # Extract title
        raw_title = entry.get("title") if isinstance(entry, dict) else getattr(entry, "title", None)
        title = clean_html(raw_title)
        if not title:
            return None

        # Extract url / link
        raw_url = entry.get("link") if isinstance(entry, dict) else getattr(entry, "link", None)
        if not raw_url:
            raw_url = entry.get("url") if isinstance(entry, dict) else getattr(entry, "url", None)
        if not raw_url:
            return None
        url = str(raw_url).strip()
        if not url:
            return None

        # Extract summary / description
        raw_summary = (
            entry.get("summary")
            if isinstance(entry, dict)
            else getattr(entry, "summary", None)
        )
        if not raw_summary:
            raw_summary = (
                entry.get("description")
                if isinstance(entry, dict)
                else getattr(entry, "description", None)
            )
        if not raw_summary:
            raw_summary = (
                entry.get("subtitle")
                if isinstance(entry, dict)
                else getattr(entry, "subtitle", None)
            )
        summary = clean_html(raw_summary)

        # Extract content if available
        content: Optional[str] = None
        raw_content = entry.get("content") if isinstance(entry, dict) else getattr(entry, "content", None)
        if raw_content and isinstance(raw_content, list):
            content_chunks = []
            for chunk in raw_content:
                if isinstance(chunk, dict) and "value" in chunk:
                    cleaned_chunk = clean_html(chunk["value"])
                    if cleaned_chunk:
                        content_chunks.append(cleaned_chunk)
            if content_chunks:
                content = "\n\n".join(content_chunks)
        elif isinstance(raw_content, str):
            content = clean_html(raw_content)

        # Full-text body extraction if enabled and article URL is present
        if self.extract_full_body and url:
            try:
                full_body = ArticleBodyExtractor().extract_full_text(url)
                if full_body:
                    content = full_body
            except Exception as exc:
                logger.warning("Failed to extract full body for '%s': %s", url, exc)

        # Extract GUID / ID
        guid: Optional[str] = None
        raw_id = entry.get("id") if isinstance(entry, dict) else getattr(entry, "id", None)
        if not raw_id:
            raw_id = entry.get("guid") if isinstance(entry, dict) else getattr(entry, "guid", None)
        if isinstance(raw_id, dict):
            guid = str(raw_id.get("value", "")).strip() or None
        elif raw_id is not None:
            guid = str(raw_id).strip() or None

        # Extract publication timestamp
        struct_time = (
            entry.get("published_parsed")
            if isinstance(entry, dict)
            else getattr(entry, "published_parsed", None)
        )
        if not struct_time:
            struct_time = (
                entry.get("updated_parsed")
                if isinstance(entry, dict)
                else getattr(entry, "updated_parsed", None)
            )

        raw_date = (
            entry.get("published")
            if isinstance(entry, dict)
            else getattr(entry, "published", None)
        )
        if not raw_date:
            raw_date = (
                entry.get("updated")
                if isinstance(entry, dict)
                else getattr(entry, "updated", None)
            )
        if not raw_date:
            raw_date = (
                entry.get("pubDate")
                if isinstance(entry, dict)
                else getattr(entry, "pubDate", None)
            )

        published_at = parse_datetime(raw_date, struct_time=struct_time)

        return RawArticle(
            title=title,
            url=url,
            source_name=self.source_config.name,
            published_at=published_at,
            summary=summary,
            content=content,
            guid=guid,
        )


def fetch_all_sources(
    sources: List[SourceConfig],
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_items_per_source: Optional[int] = None,
    max_article_age_minutes: Optional[int] = None,
    storage: Optional[BaseStorage] = None,
    enable_etag_cache: bool = True,
    max_retries: int = 3,
    retry_backoff: float = 1.0,
    extract_full_body: bool = False,
) -> List[RawArticle]:
    """Fetch articles from all enabled sources with feed-level error containment.

    Iterates through all configured sources, skips disabled sources, and executes
    the appropriate fetcher per source type. Errors on any individual source are
    isolated and logged so the pipeline continues processing remaining sources.

    Args:
        sources: List of SourceConfig instances to poll.
        timeout: Socket timeout in seconds for feed requests.
        max_items_per_source: Optional maximum number of recent articles to ingest per feed.
        max_article_age_minutes: Optional maximum age of articles in minutes to ingest.
        storage: Optional BaseStorage instance for conditional ETag and Last-Modified caching.
        enable_etag_cache: Whether to use HTTP conditional GET headers.
        max_retries: Maximum retry attempts on network failures or transient 5xx HTTP errors.
        retry_backoff: Backoff multiplier in seconds.
        extract_full_body: Whether to fetch full article body from article links.

    Returns:
        List[RawArticle]: Aggregated list of raw articles from all successfully polled sources.
    """
    if not sources:
        return []

    all_articles: List[RawArticle] = []
    for source in sources:
        if not source.enabled:
            logger.debug("Skipping disabled source '%s'", source.name)
            continue

        if source.type.lower() != "rss":
            logger.info("Unsupported source type '%s' for '%s', skipping", source.type, source.name)
            continue

        try:
            fetcher = RSSFetcher(
                source,
                timeout=timeout,
                max_items=max_items_per_source,
                max_article_age_minutes=max_article_age_minutes,
                storage=storage,
                enable_etag_cache=enable_etag_cache,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
                extract_full_body=extract_full_body,
            )
            articles = fetcher.fetch()
            all_articles.extend(articles)
        except Exception as exc:
            logger.error(
                "Unhandled error fetching source '%s' (%s): %s",
                source.name,
                source.url,
                exc,
                exc_info=True,
            )

    return all_articles
