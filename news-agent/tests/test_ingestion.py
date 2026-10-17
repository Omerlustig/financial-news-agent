import datetime
import time
from unittest.mock import patch
import pytest

from src.config import SourceConfig
from src.models import RawArticle
from src.ingestion.base import BaseFetcher
from src.ingestion.rss import RSSFetcher, clean_html, fetch_all_sources, parse_datetime


MOCK_RSS_XML = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>Financial Feed</title>
    <item>
      <title>Fed Signals Pause on Rates</title>
      <link>https://example.com/fed-pause</link>
      <description><![CDATA[<p>The Federal Reserve announced &amp; signaled a pause...</p>]]></description>
      <pubDate>Sat, 17 Oct 2026 12:00:00 GMT</pubDate>
      <guid>https://example.com/fed-pause-123</guid>
    </item>
  </channel>
</rss>"""


def test_base_fetcher_cannot_be_instantiated_directly():
    src = SourceConfig(name="Mock", type="rss", url="https://example.com/rss", enabled=True)
    with pytest.raises(TypeError):
        BaseFetcher(src)  # type: ignore[abstract]


def test_rss_fetcher_parses_feed():
    src = SourceConfig(name="Mock Feed", type="rss", url="https://example.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entry = {
            "title": "Fed Signals Pause on Rates",
            "link": "https://example.com/fed-pause",
            "summary": "The Federal Reserve announced...",
            "published": "Sat, 17 Oct 2026 12:00:00 GMT",
            "id": "item-123",
        }
        mock_parse.return_value = {"entries": [mock_entry], "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 1
        article = articles[0]
        assert isinstance(article, RawArticle)
        assert article.title == "Fed Signals Pause on Rates"
        assert article.url == "https://example.com/fed-pause"
        assert article.source_name == "Mock Feed"
        assert article.summary == "The Federal Reserve announced..."
        assert article.guid == "item-123"
        assert article.published_at == datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.timezone.utc)
        assert article.content_hash is not None


def test_rss_fetcher_with_actual_xml_string():
    # End-to-end test against real feedparser parsing
    src = SourceConfig(name="Real XML Feed", type="rss", url=MOCK_RSS_XML, enabled=True)
    fetcher = RSSFetcher(src)
    articles = fetcher.fetch()

    assert len(articles) == 1
    article = articles[0]
    assert article.title == "Fed Signals Pause on Rates"
    assert article.url == "https://example.com/fed-pause"
    assert article.source_name == "Real XML Feed"
    assert article.summary == "The Federal Reserve announced & signaled a pause..."
    assert article.guid == "https://example.com/fed-pause-123"
    assert article.published_at == datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.timezone.utc)


def test_rss_fetcher_handles_feed_failure_gracefully():
    src = SourceConfig(name="Broken Feed", type="rss", url="https://broken.com/rss", enabled=True)
    with patch("feedparser.parse", side_effect=Exception("Connection timeout")):
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()
        assert articles == []


def test_rss_fetcher_handles_malformed_bozo_feed():
    src = SourceConfig(name="Malformed Feed", type="rss", url="https://malformed.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_parse.return_value = {
            "entries": [],
            "bozo": 1,
            "bozo_exception": Exception("XML syntax error"),
        }
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()
        assert articles == []


def test_rss_fetcher_recovers_entries_despite_bozo():
    src = SourceConfig(name="Tolerant Feed", type="rss", url="https://tolerant.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entry = {
            "title": "ECB Holds Policy Steady",
            "link": "https://example.com/ecb-steady",
            "summary": "Policy maintained as expected.",
            "published": "2026-10-17T14:30:00Z",
        }
        mock_parse.return_value = {
            "entries": [mock_entry],
            "bozo": 1,
            "bozo_exception": Exception("Non-fatal XML warning"),
        }
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()
        assert len(articles) == 1
        assert articles[0].title == "ECB Holds Policy Steady"
        assert articles[0].published_at == datetime.datetime(2026, 10, 17, 14, 30, tzinfo=datetime.timezone.utc)


def test_rss_fetcher_strips_html_and_unescapes():
    src = SourceConfig(name="HTML Feed", type="rss", url="https://example.com/html-rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entry = {
            "title": "<b>NVIDIA</b> Reports &amp; Beats Estimates",
            "link": "https://example.com/nvda-beats",
            "summary": "<p>Q3 earnings revenue jumped 120% YoY.<br/>Data center segment was &quot;stellar&quot;.</p>",
            "content": [{"value": "<div>Full body text &amp; analysis.</div>", "type": "text/html"}],
        }
        mock_parse.return_value = {"entries": [mock_entry], "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 1
        article = articles[0]
        assert article.title == "NVIDIA Reports & Beats Estimates"
        assert "<p>" not in article.summary
        assert "<br/>" not in article.summary
        assert "&quot;" not in article.summary
        assert 'Data center segment was "stellar".' in article.summary
        assert article.content == "Full body text & analysis."


def test_rss_fetcher_skips_invalid_entries():
    src = SourceConfig(name="Sparse Feed", type="rss", url="https://example.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entries = [
            {"title": "", "link": "https://example.com/1"},  # empty title
            {"title": "Valid Headline", "link": ""},  # empty link
            {"title": "   ", "link": "   "},  # whitespace only
            {"title": "Valid News", "link": "https://example.com/valid"},
        ]
        mock_parse.return_value = {"entries": mock_entries, "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 1
        assert articles[0].title == "Valid News"
        assert articles[0].url == "https://example.com/valid"


def test_rss_fetcher_handles_struct_time_date():
    src = SourceConfig(name="Struct Feed", type="rss", url="https://example.com/rss", enabled=True)
    mock_struct = time.gmtime(1792238400)  # fixed timestamp
    expected_dt = datetime.datetime.fromtimestamp(1792238400, tz=datetime.timezone.utc)

    with patch("feedparser.parse") as mock_parse:
        mock_entry = {
            "title": "Treasury Yields Rise",
            "link": "https://example.com/yields",
            "published_parsed": mock_struct,
        }
        mock_parse.return_value = {"entries": [mock_entry], "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 1
        assert articles[0].published_at == expected_dt


def test_rss_fetcher_handles_iso_and_timezone_offsets():
    src = SourceConfig(name="Timezone Feed", type="rss", url="https://example.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entries = [
            {
                "title": "Asian Markets Open",
                "link": "https://example.com/tokyo",
                "published": "2026-10-17T09:00:00+09:00",
            },
            {
                "title": "Wall Street Pre-market",
                "link": "https://example.com/nyc",
                "published": "2026-10-17T08:00:00-04:00",
            },
            {
                "title": "No Date News",
                "link": "https://example.com/nodate",
                "published": "not-a-valid-date",
            },
        ]
        mock_parse.return_value = {"entries": mock_entries, "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 3
        # 09:00 +09:00 -> 00:00 UTC
        assert articles[0].published_at == datetime.datetime(2026, 10, 17, 0, 0, tzinfo=datetime.timezone.utc)
        # 08:00 -04:00 -> 12:00 UTC
        assert articles[1].published_at == datetime.datetime(2026, 10, 17, 12, 0, tzinfo=datetime.timezone.utc)
        # Invalid date defaults to None
        assert articles[2].published_at is None


def test_rss_fetcher_handles_guid_dict():
    src = SourceConfig(name="Dict Guid Feed", type="rss", url="https://example.com/rss", enabled=True)
    with patch("feedparser.parse") as mock_parse:
        mock_entry = {
            "title": "Fed Minutes",
            "link": "https://example.com/minutes",
            "id": {"isPermaLink": "false", "value": "guid-dict-999"},
        }
        mock_parse.return_value = {"entries": [mock_entry], "bozo": 0}
        fetcher = RSSFetcher(src)
        articles = fetcher.fetch()

        assert len(articles) == 1
        assert articles[0].guid == "guid-dict-999"


def test_clean_html_utility():
    assert clean_html(None) == ""
    assert clean_html("") == ""
    assert clean_html("Simple text") == "Simple text"
    assert clean_html("<div>Line 1</div><p>Line 2</p>") == "Line 1 Line 2"
    assert clean_html("&lt;tag&gt; &amp; &quot;quote&quot;") == '<tag> & "quote"'


def test_parse_datetime_utility():
    assert parse_datetime(None) is None
    assert parse_datetime("") is None
    assert parse_datetime("invalid-date-string-xyz") is None


def test_fetch_all_sources_isolation():
    sources = [
        SourceConfig(name="Source 1", type="rss", url="https://s1.com/rss", enabled=True),
        SourceConfig(name="Source 2 Broken", type="rss", url="https://s2-broken.com/rss", enabled=True),
        SourceConfig(name="Source 3 Disabled", type="rss", url="https://s3.com/rss", enabled=False),
        SourceConfig(name="Source 4", type="rss", url="https://s4.com/rss", enabled=True),
    ]

    def side_effect(url, **kwargs):
        if "s1.com" in url:
            return {"entries": [{"title": "Article 1", "link": "https://s1.com/art1"}], "bozo": 0}
        elif "s2-broken.com" in url:
            raise ConnectionError("DNS resolution failed")
        elif "s4.com" in url:
            return {"entries": [{"title": "Article 4", "link": "https://s4.com/art4"}], "bozo": 0}
        raise AssertionError("Disabled source should not be queried")

    with patch("feedparser.parse", side_effect=side_effect):
        articles = fetch_all_sources(sources)

        assert len(articles) == 2
        titles = [a.title for a in articles]
        assert "Article 1" in titles
        assert "Article 4" in titles
        source_names = [a.source_name for a in articles]
        assert "Source 1" in source_names
        assert "Source 4" in source_names


def test_fetch_all_sources_empty():
    assert fetch_all_sources([]) == []
    assert fetch_all_sources([SourceConfig(name="Off", type="rss", url="https://off.com", enabled=False)]) == []
