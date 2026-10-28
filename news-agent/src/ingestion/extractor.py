"""Full-text article body extractor for RSS and web links.

Uses Trafilatura for robust main-text and table extraction when available,
with graceful fallback to BeautifulSoup or regex-based paragraph extraction.
Includes content-type filtering, network error containment, and safe-boundary truncation.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
from typing import Optional

import requests

try:
    import trafilatura  # type: ignore[import-untyped]
except ImportError:
    trafilatura = None

try:
    from bs4 import BeautifulSoup  # type: ignore[import-untyped]
except ImportError:
    BeautifulSoup = None

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_SECONDS: float = 10.0
DEFAULT_MAX_LENGTH: int = 15000
DEFAULT_USER_AGENT: str = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

NON_HTML_EXTENSIONS: tuple[str, ...] = (
    ".pdf",
    ".zip",
    ".tar",
    ".gz",
    ".tgz",
    ".bz2",
    ".xz",
    ".7z",
    ".rar",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".svg",
    ".ico",
    ".mp3",
    ".mp4",
    ".m4a",
    ".wav",
    ".mov",
    ".avi",
    ".mkv",
    ".exe",
    ".bin",
    ".dmg",
    ".iso",
    ".docx",
    ".xlsx",
    ".pptx",
)


class ArticleBodyExtractor:
    """Extracts clean full-text article content from URLs with fallback mechanisms."""

    def __init__(
        self,
        max_length: int = DEFAULT_MAX_LENGTH,
        user_agent: Optional[str] = None,
    ) -> None:
        """Initialize the extractor.

        Args:
            max_length: Maximum allowed character length for extracted article body.
            user_agent: HTTP User-Agent header for fetching web pages.
        """
        self.max_length = max_length
        self.user_agent = user_agent or DEFAULT_USER_AGENT

    def extract_full_text(
        self,
        url: str,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> Optional[str]:
        """Extract article body text from a web page URL.

        Attempts high-accuracy extraction with Trafilatura first. If Trafilatura is
        unavailable, returns None, or fails, falls back to requests + BeautifulSoup/regex.
        Enforces content guardrails (skipping PDFs and non-HTML payloads), request timeouts,
        safe error catching, whitespace normalization, and length truncation.

        Args:
            url: The HTTP/HTTPS web page URL to extract.
            timeout: Network request timeout in seconds.

        Returns:
            Clean extracted body text, or None if extraction failed or content is invalid.
        """
        if not url or not isinstance(url, str):
            return None

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            return None

        # Guardrail: Check non-HTML file extension in URL
        if self._is_non_html_url(url):
            logger.debug("Skipping full-text extraction for non-HTML URL: %s", url)
            return None

        # Guardrail: Check domain rate-limit cooldown
        from src.ingestion.base import DomainCooldownTracker
        if DomainCooldownTracker.is_cooling_down(url):
            logger.debug("Skipping full-text extraction for %s: domain is in cooldown", url)
            return None

        text: Optional[str] = None

        # Primary extraction: Trafilatura (if available)
        if trafilatura is not None:
            try:
                downloaded = trafilatura.fetch_url(url)
                if downloaded:
                    text = trafilatura.extract(
                        downloaded,
                        include_comments=False,
                        include_tables=True,
                        no_fallback=False,
                    )
            except Exception as exc:
                logger.debug("Trafilatura failed on %s: %s", url, exc)
                text = None

        # Fallback extraction: requests + BeautifulSoup or regex
        if not text:
            try:
                text = self._requests_fallback(url, timeout=timeout)
            except Exception as exc:
                logger.debug("Requests fallback failed on %s: %s", url, exc)
                text = None

        if not text:
            return None

        # Clean whitespace and normalize line breaks
        cleaned = self._clean_whitespace(text)
        if not cleaned:
            return None

        # Truncate at safe word boundary if exceeding max_length
        return self._truncate_text(cleaned)

    def extract_from_html(self, html_content: str) -> Optional[str]:
        """Extract and sanitize main text directly from raw HTML string.

        Args:
            html_content: Raw HTML text.

        Returns:
            Clean, truncated article body text, or None if extraction yields nothing.
        """
        if not html_content or not isinstance(html_content, str):
            return None

        text: Optional[str] = None
        if trafilatura is not None:
            try:
                text = trafilatura.extract(
                    html_content,
                    include_comments=False,
                    include_tables=True,
                    no_fallback=False,
                )
            except Exception as exc:
                logger.debug("Trafilatura direct extraction failed: %s", exc)
                text = None

        if not text:
            if BeautifulSoup is not None:
                text = self._extract_with_bs4(html_content)
            else:
                text = self._extract_with_regex(html_content)

        if not text:
            return None

        cleaned = self._clean_whitespace(text)
        if not cleaned:
            return None

        return self._truncate_text(cleaned)

    def _is_non_html_url(self, url: str) -> bool:
        """Check if URL points to a non-HTML file based on extension."""
        try:
            parsed = urllib.parse.urlsplit(url)
            path = parsed.path.lower()
            return any(path.endswith(ext) for ext in NON_HTML_EXTENSIONS)
        except Exception:
            return False

    def _requests_fallback(self, url: str, timeout: float) -> Optional[str]:
        """Fallback extractor using requests and BeautifulSoup/regex."""
        from src.ingestion.base import DomainCooldownTracker, parse_retry_after

        if DomainCooldownTracker.is_cooling_down(url):
            logger.debug("Skipping fallback extraction for %s: host is in cooldown", url)
            return None

        headers = {"User-Agent": self.user_agent}
        try:
            response = requests.get(url, headers=headers, timeout=timeout)
        except Exception as exc:
            logger.debug("Fallback request to %s raised %s", url, exc)
            return None

        if response.status_code == 429:
            retry_after = parse_retry_after(response, default=30.0)
            DomainCooldownTracker.set_cooldown(url, seconds=retry_after)
            logger.warning("HTTP 429 when extracting article from %s. Domain cooling down for %.1fs.", url, retry_after)
            return None

        if response.status_code != 200:
            logger.debug("Fallback request to %s returned HTTP %d", url, response.status_code)
            return None

        content_type = response.headers.get("Content-Type", "").lower()
        if not self._is_html_content_type(content_type):
            logger.debug("Skipping non-HTML content-type '%s' for %s", content_type, url)
            return None

        html_text = response.text
        if not html_text or not html_text.strip():
            return None

        if BeautifulSoup is not None:
            return self._extract_with_bs4(html_text)
        return self._extract_with_regex(html_text)

    def _is_html_content_type(self, content_type: str) -> bool:
        """Validate if Content-Type header represents HTML or XML."""
        if not content_type:
            return True
        ct = content_type.lower()
        if "pdf" in ct:
            return False
        if any(
            ct.startswith(prefix)
            for prefix in ("image/", "audio/", "video/", "application/zip", "application/octet-stream")
        ):
            return False
        return any(valid in ct for valid in ("text/html", "application/xhtml", "text/plain", "xml"))

    def _extract_with_bs4(self, html_content: str) -> Optional[str]:
        """Extract clean paragraph and heading text using BeautifulSoup."""
        if BeautifulSoup is None:
            return None
        try:
            soup = BeautifulSoup(html_content, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript", "form", "svg"]):
                tag.decompose()

            article = (
                soup.find("article")
                or soup.find("main")
                or soup.find(class_=re.compile(r"(article|post|story|entry-content|body)", re.I))
            )
            root = article if article else (soup.body or soup)

            paragraphs: list[str] = []
            for el in root.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
                text = el.get_text(separator=" ", strip=True)
                if text:
                    paragraphs.append(text)

            if paragraphs:
                return "\n\n".join(paragraphs)

            fallback_text = root.get_text(separator="\n", strip=True)
            return fallback_text if fallback_text else None
        except Exception as exc:
            logger.debug("BeautifulSoup extraction failed: %s", exc)
            return None

    def _extract_with_regex(self, html_content: str) -> Optional[str]:
        """Extract paragraph text using regex when BeautifulSoup is unavailable."""
        try:
            cleaned = re.sub(
                r"(?is)<(script|style|nav|footer|header|aside|noscript|form)[^>]*>.*?</\1>",
                " ",
                html_content,
            )
            p_matches = re.findall(r"(?is)<p[^>]*>(.*?)</p>", cleaned)
            if p_matches:
                paragraphs: list[str] = []
                for p in p_matches:
                    stripped = re.sub(r"<[^>]+>", " ", p)
                    unescaped = html.unescape(stripped).strip()
                    if unescaped:
                        paragraphs.append(unescaped)
                if paragraphs:
                    return "\n\n".join(paragraphs)

            stripped = re.sub(r"<[^>]+>", " ", cleaned)
            unescaped = html.unescape(stripped).strip()
            return unescaped if unescaped else None
        except Exception as exc:
            logger.debug("Regex extraction failed: %s", exc)
            return None

    def _clean_whitespace(self, text: str) -> str:
        """Normalize internal whitespace and collapse excessive newlines."""
        lines = [re.sub(r"[ \t\r\f\v]+", " ", line).strip() for line in text.splitlines()]
        cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(lines))
        return cleaned.strip()

    def _truncate_text(self, text: str) -> str:
        """Truncate text to max_length at a clean word boundary."""
        if len(text) <= self.max_length:
            return text

        truncated = text[: self.max_length]
        last_break = max(truncated.rfind(" "), truncated.rfind("\n"))
        if last_break > int(self.max_length * 0.8):
            truncated = truncated[:last_break]
        return truncated.strip()
