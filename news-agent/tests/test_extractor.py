"""Unit tests for ArticleBodyExtractor and full-text ingestion integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
import pytest
import requests

from src.config import SourceConfig
from src.ingestion.extractor import (
    DEFAULT_MAX_LENGTH,
    ArticleBodyExtractor,
)
from src.ingestion.rss import RSSFetcher, fetch_all_sources


SAMPLE_HTML = """
<!DOCTYPE html>
<html>
<head><title>Federal Reserve Signals Rate Decision</title></head>
<body>
    <header><nav><a href="/">Home</a></nav></header>
    <article>
        <h1>Federal Reserve Signals Rate Decision</h1>
        <p class="byline">By Market Reporter</p>
        <p>The Federal Reserve announced on Wednesday that benchmark borrowing rates will remain unchanged at 5.25%-5.50%.</p>
        <p>Chair Jerome Powell highlighted that economic growth remains resilient and inflationary pressures continue to subside.</p>
        <aside><p>Related: Stock market rally continues</p></aside>
        <p>Markets rallied following the policy announcement, with treasury yields declining across the curve.</p>
    </article>
    <footer><p>&copy; 2026 Financial News Network</p></footer>
</body>
</html>
"""


@pytest.fixture
def extractor() -> ArticleBodyExtractor:
    return ArticleBodyExtractor(max_length=15000)


def test_extractor_initialization():
    """Verify ArticleBodyExtractor default configurations."""
    ext = ArticleBodyExtractor()
    assert ext.max_length == DEFAULT_MAX_LENGTH
    assert "Mozilla" in ext.user_agent


def test_extractor_successful_extraction_trafilatura(extractor: ArticleBodyExtractor):
    """Test successful text extraction using trafilatura."""
    mock_url = "https://example.com/article/fed-rates"
    expected_text = (
        "Federal Reserve Signals Rate Decision\n\n"
        "The Federal Reserve announced on Wednesday that benchmark borrowing rates will remain unchanged."
    )

    with patch("src.ingestion.extractor.trafilatura") as mock_traf:
        mock_traf.fetch_url.return_value = SAMPLE_HTML
        mock_traf.extract.return_value = expected_text

        result = extractor.extract_full_text(mock_url)
        assert result is not None
        assert "Federal Reserve Signals Rate Decision" in result
        mock_traf.fetch_url.assert_called_once_with(mock_url)
        mock_traf.extract.assert_called_once_with(
            SAMPLE_HTML,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
        )


def test_extractor_from_html_direct(extractor: ArticleBodyExtractor):
    """Test extraction directly from HTML string using extract_from_html."""
    result = extractor.extract_from_html(SAMPLE_HTML)
    assert result is not None
    assert "Federal Reserve" in result
    assert "unchanged" in result
    # Nav/header/footer content should be cleaned out
    assert "Home" not in result or "Financial News Network" not in result


def test_extractor_fallback_when_trafilatura_returns_none(extractor: ArticleBodyExtractor):
    """Test fallback to requests + BeautifulSoup/regex when trafilatura returns None."""
    mock_url = "https://example.com/article/fallback-test"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html; charset=utf-8"}
    mock_response.text = SAMPLE_HTML

    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get", return_value=mock_response) as mock_get:
        mock_traf.fetch_url.return_value = None

        result = extractor.extract_full_text(mock_url)
        assert result is not None
        assert "Federal Reserve" in result
        assert "benchmark borrowing rates" in result
        mock_get.assert_called_once()


def test_extractor_fallback_when_trafilatura_extract_returns_none(extractor: ArticleBodyExtractor):
    """Test fallback when trafilatura download succeeds but extraction returns None."""
    mock_url = "https://example.com/article/empty-extract"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<article><p>Crucial market intelligence report text.</p></article>"

    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get", return_value=mock_response):
        mock_traf.fetch_url.return_value = "<html>some obscure layout</html>"
        mock_traf.extract.return_value = None

        result = extractor.extract_full_text(mock_url)
        assert result is not None
        assert "Crucial market intelligence report text." in result


def test_extractor_fallback_when_trafilatura_not_installed():
    """Test fallback when trafilatura is unavailable (ImportError)."""
    ext = ArticleBodyExtractor()
    mock_url = "https://example.com/article/no-trafilatura"

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "text/html"}
    mock_response.text = "<html><body><article><p>Text without trafilatura dependency.</p></article></body></html>"

    with patch("src.ingestion.extractor.trafilatura", None), \
         patch("src.ingestion.extractor.requests.get", return_value=mock_response):
        result = ext.extract_full_text(mock_url)
        assert result is not None
        assert "Text without trafilatura dependency." in result


def test_extractor_fallback_regex_without_bs4():
    """Test pure regex fallback when both trafilatura and BeautifulSoup are unavailable."""
    ext = ArticleBodyExtractor()
    html_content = (
        "<html><head><script>var x = 1;</script></head>"
        "<body>"
        "<p>First paragraph with &amp; entity.</p>"
        "<p>Second paragraph with &quot;quotes&quot;.</p>"
        "</body></html>"
    )

    with patch("src.ingestion.extractor.trafilatura", None), \
         patch("src.ingestion.extractor.BeautifulSoup", None):
        result = ext.extract_from_html(html_content)
        assert result is not None
        assert "First paragraph with & entity." in result
        assert 'Second paragraph with "quotes".' in result
        assert "var x = 1" not in result


def test_extractor_network_error_handling_returns_none(extractor: ArticleBodyExtractor):
    """Test that connection and timeout exceptions are caught safely without crashing."""
    mock_url = "https://example.com/article/network-failure"

    # Trafilatura fetch raises exception
    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get", side_effect=requests.exceptions.ConnectTimeout("Timeout")):
        mock_traf.fetch_url.side_effect = Exception("Urllib connection broken")

        result = extractor.extract_full_text(mock_url)
        assert result is None


def test_extractor_requests_http_error_returns_none(extractor: ArticleBodyExtractor):
    """Test that HTTP errors in fallback (e.g. 404, 500) return None cleanly."""
    mock_url = "https://example.com/article/not-found"

    mock_response = MagicMock()
    mock_response.status_code = 404

    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get", return_value=mock_response):
        mock_traf.fetch_url.return_value = None

        result = extractor.extract_full_text(mock_url)
        assert result is None


def test_extractor_max_length_truncation():
    """Test truncation at safe word boundaries when text exceeds max_length."""
    short_limit_extractor = ArticleBodyExtractor(max_length=60)
    long_html = (
        "<html><body><p>"
        "The global economy is experiencing unexpected resilience across multiple key sectors "
        "including semiconductors, cloud infrastructure, and biotechnology innovations."
        "</p></body></html>"
    )

    result = short_limit_extractor.extract_from_html(long_html)
    assert result is not None
    assert len(result) <= 60
    # Should end on a clean word boundary, not cutting a word in half
    assert not result.endswith(" ")
    assert not result.endswith("innova")


def test_extractor_guardrail_skips_pdf_and_binary_urls(extractor: ArticleBodyExtractor):
    """Test guardrail immediately skipping PDF and non-HTML URL extensions without network calls."""
    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get") as mock_get:
        assert extractor.extract_full_text("https://example.com/reports/quarterly.pdf") is None
        assert extractor.extract_full_text("https://example.com/filing.pdf?download=true#page=1") is None
        assert extractor.extract_full_text("https://example.com/data.zip") is None
        assert extractor.extract_full_text("https://example.com/chart.png") is None

        mock_traf.fetch_url.assert_not_called()
        mock_get.assert_not_called()


def test_extractor_guardrail_skips_pdf_content_type(extractor: ArticleBodyExtractor):
    """Test guardrail skipping non-HTML Content-Type responses in fallback."""
    mock_url = "https://example.com/download-doc"
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.headers = {"Content-Type": "application/pdf"}
    mock_response.text = "%PDF-1.4 binary content"

    with patch("src.ingestion.extractor.trafilatura") as mock_traf, \
         patch("src.ingestion.extractor.requests.get", return_value=mock_response):
        mock_traf.fetch_url.return_value = None

        result = extractor.extract_full_text(mock_url)
        assert result is None


def test_extractor_invalid_and_empty_inputs(extractor: ArticleBodyExtractor):
    """Test guardrails on empty, None, and non-HTTP URLs."""
    assert extractor.extract_full_text("") is None
    assert extractor.extract_full_text(None) is None  # type: ignore[arg-type]
    assert extractor.extract_full_text("ftp://example.com/feed") is None
    assert extractor.extract_full_text("javascript:alert(1)") is None
    assert extractor.extract_from_html("") is None
    assert extractor.extract_from_html(None) is None  # type: ignore[arg-type]


def test_rss_fetcher_extract_full_body_enabled():
    """Verify RSSFetcher extracts full body when extract_full_body is True."""
    src = SourceConfig(name="TechNews", type="rss", url="https://example.com/rss")
    fetcher = RSSFetcher(src, extract_full_body=True)

    feed_content = b"""<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
        <channel>
            <title>Tech News</title>
            <item>
                <title>NVIDIA Unveils Next-Gen AI Chip</title>
                <link>https://example.com/nvidia-ai-chip</link>
                <description>Brief teaser description of the chip.</description>
            </item>
        </channel>
    </rss>"""

    extracted_body = (
        "NVIDIA has unveiled its next-generation AI chip architecture today. "
        "The new accelerator promises 4x throughput improvements for large language models."
    )

    with patch("feedparser.parse") as mock_parse, \
         patch("src.ingestion.rss.ArticleBodyExtractor") as mock_extractor_cls:
        mock_instance = MagicMock()
        mock_instance.extract_full_text.return_value = extracted_body
        mock_extractor_cls.return_value = mock_instance

        mock_parse.return_value = MagicMock(
            entries=[{
                "title": "NVIDIA Unveils Next-Gen AI Chip",
                "link": "https://example.com/nvidia-ai-chip",
                "description": "Brief teaser description of the chip.",
            }],
            bozo=0,
        )

        articles = fetcher.fetch()
        assert len(articles) == 1
        assert articles[0].title == "NVIDIA Unveils Next-Gen AI Chip"
        assert articles[0].summary == "Brief teaser description of the chip."
        # Full content populated from extractor
        assert articles[0].content == extracted_body
        mock_instance.extract_full_text.assert_called_once_with("https://example.com/nvidia-ai-chip")


def test_rss_fetcher_extract_full_body_disabled_by_default():
    """Verify RSSFetcher does not trigger ArticleBodyExtractor when extract_full_body is False."""
    src = SourceConfig(name="TechNews", type="rss", url="https://example.com/rss")
    fetcher = RSSFetcher(src)  # default extract_full_body=False

    with patch("feedparser.parse") as mock_parse, \
         patch("src.ingestion.rss.ArticleBodyExtractor") as mock_extractor_cls:
        mock_parse.return_value = MagicMock(
            entries=[{
                "title": "NVIDIA Unveils Next-Gen AI Chip",
                "link": "https://example.com/nvidia-ai-chip",
                "description": "Brief teaser description of the chip.",
            }],
            bozo=0,
        )

        articles = fetcher.fetch()
        assert len(articles) == 1
        assert articles[0].content is None
        mock_extractor_cls.assert_not_called()


def test_fetch_all_sources_forwards_extract_full_body():
    """Verify fetch_all_sources forwards extract_full_body parameter to RSSFetcher."""
    sources = [SourceConfig(name="S1", type="rss", url="https://example.com/rss")]

    with patch("src.ingestion.rss.RSSFetcher") as mock_fetcher_cls:
        mock_instance = MagicMock()
        mock_instance.fetch.return_value = []
        mock_fetcher_cls.return_value = mock_instance

        fetch_all_sources(sources, extract_full_body=True)

        mock_fetcher_cls.assert_called_once()
        _, kwargs = mock_fetcher_cls.call_args
        assert kwargs.get("extract_full_body") is True
