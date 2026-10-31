from datetime import datetime, timezone
import pytest
from src.models import RawArticle
from src.ingestion.prefilter import NewsPreFilter, PreFilterConfig


@pytest.fixture
def prefilter() -> NewsPreFilter:
    return NewsPreFilter()


def test_prefilter_drops_sponsored_and_clickbait(prefilter: NewsPreFilter) -> None:
    clickbait_titles = [
        "3 stocks to buy now before the market opens",
        "Zacks investment ideas: top growth stocks",
        "Motley Fool issues rare double down buy alert",
        "Sponsored: why this microcap could 10x",
        "Partner Content: best high dividend funds",
    ]
    for title in clickbait_titles:
        article = RawArticle(
            title=title,
            url=f"https://example.com/{abs(hash(title))}",
            source_name="TestWire",
            published_at=datetime.now(timezone.utc),
            summary="Routine marketing roundup.",
        )
        assert prefilter.is_noise(article) is True, f"Failed to drop clickbait: {title}"


def test_prefilter_allows_genuine_catalyst_news(prefilter: NewsPreFilter) -> None:
    catalyst_titles = [
        "Nvidia unveils next-generation Rubin Blackwell architecture in Taipei",
        "Eli Lilly receives FDA approval for accelerated Alzheimer drug formulation",
        "Microsoft acquires AI cloud infrastructure provider for $650M",
        "Federal Reserve cuts benchmark rate by 25 basis points in unanimous vote",
    ]
    for title in catalyst_titles:
        article = RawArticle(
            title=title,
            url=f"https://example.com/{abs(hash(title))}",
            source_name="Reuters",
            published_at=datetime.now(timezone.utc),
            summary="Detailed reporting on major corporate catalyst.",
        )
        assert prefilter.is_noise(article) is False, f"Erroneously dropped catalyst: {title}"


def test_prefilter_strips_blacklisted_ticker_acronyms(prefilter: NewsPreFilter) -> None:
    raw_tickers = ["AAPL", "CEO", "NVDA", "AI", "FED", "MSFT", "US", "GDP", "SEC", "FDA", "EV"]
    filtered = prefilter.filter_tickers(raw_tickers)
    assert set(filtered) == {"AAPL", "NVDA", "MSFT"}


def test_prefilter_handles_empty_or_short_titles(prefilter: NewsPreFilter) -> None:
    article = RawArticle(
        title="Short",
        url="https://example.com/short",
        source_name="TestWire",
        published_at=datetime.now(timezone.utc),
    )
    assert prefilter.is_noise(article) is True
