from datetime import datetime, timezone, timedelta
import pytest

from src.models import (
    DirectionalStance,
    EventStatus,
    HardGates,
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
)
from src.analysis.corroborator import EventCluster, EventCorroborator
from src.pipeline import Pipeline
from src.storage.sqlite import SQLiteStorage
from src.notification.base import BaseNotifier


class MockNotifier(BaseNotifier):
    def __init__(self):
        self.sent_alerts = []

    def send_alert(self, article, analysis) -> bool:
        self.sent_alerts.append((article, analysis))
        return True

    def send_message(self, text: str) -> bool:
        return True


class MockAnalyzer:
    def __init__(self, base_score: int = 6, base_status: EventStatus = EventStatus.WATCHLIST):
        self.base_score = base_score
        self.base_status = base_status

    def analyze(self, article: RawArticle) -> MarketImpactAnalysis:
        return MarketImpactAnalysis(
            is_market_moving=False,
            confidence_score=self.base_score,
            impact_grade=ImpactGrade.AVERAGE,
            affected_tickers=["$NVDA"],
            affected_sectors=["Semiconductors"],
            summary="US Commerce Department drafting strict chip export caps.",
            catalyst_thesis="Export bans reduce revenue for GPU manufacturers.",
            status=self.base_status,
            directional_stance=DirectionalStance.BEARISH,
            hard_gates=HardGates(
                is_novel=True,
                is_verified=False,  # Unverified initially
                has_direct_exposure=True,
                passes_all_gates=False,
            ),
        )


def test_token_extraction_and_cleaning():
    corroborator = EventCorroborator()
    tokens = corroborator.extract_tokens("Exclusive: US plans sweeping export bans on $NVDA AI chips amid new report!")
    assert "$nvda" in tokens
    assert "chips" in tokens
    assert "export" in tokens
    assert "bans" in tokens
    assert "sweeping" in tokens
    # Stopwords filtered out
    assert "the" not in tokens
    assert "on" not in tokens
    assert "amid" not in tokens
    assert "exclusive" not in tokens


def test_compute_similarity_matching_and_unrelated():
    corroborator = EventCorroborator()
    now = datetime.now(timezone.utc)

    art1 = RawArticle(
        title="US plans new restrictions on $NVDA chip exports to Middle East",
        url="https://reuters.com/article1",
        source_name="Reuters",
        published_at=now,
    )
    art2 = RawArticle(
        title="White House drafting restrictions on $NVDA AI chip exports to Middle East",
        url="https://bloomberg.com/article2",
        source_name="Bloomberg",
        published_at=now,
    )
    art3 = RawArticle(
        title="Starbucks CEO outlines new operational turnaround strategy in cafes",
        url="https://cnbc.com/article3",
        source_name="CNBC",
        published_at=now,
    )

    sim_matched = corroborator.compute_similarity(art1, art2)
    sim_unrelated = corroborator.compute_similarity(art1, art3)

    assert sim_matched >= 0.40
    assert sim_unrelated < 0.15
    assert corroborator.are_same_event(art1, art2) is True
    assert corroborator.are_same_event(art1, art3) is False


def test_time_window_separation():
    corroborator = EventCorroborator(time_window_hours=12.0)
    now = datetime.now(timezone.utc)

    art1 = RawArticle(
        title="Federal Reserve signals unexpected rate cut following inflation data",
        url="https://reuters.com/fed1",
        source_name="Reuters",
        published_at=now,
    )
    art_stale = RawArticle(
        title="Federal Reserve signals unexpected rate cut following inflation data",
        url="https://bloomberg.com/fed2",
        source_name="Bloomberg",
        published_at=now - timedelta(hours=48),
    )

    assert corroborator.are_same_event(art1, art_stale) is False


def test_cluster_articles_groups_multi_source():
    corroborator = EventCorroborator()
    now = datetime.now(timezone.utc)

    articles = [
        RawArticle(
            title="FTC prepares antitrust lawsuit against Amazon cloud division",
            url="https://reuters.com/amzn-ftc",
            source_name="Reuters",
            published_at=now,
        ),
        RawArticle(
            title="FTC drafting antitrust suit targeting Amazon Web Services",
            url="https://bloomberg.com/amzn-ftc",
            source_name="Bloomberg",
            published_at=now,
        ),
        RawArticle(
            title="Amazon faces FTC antitrust scrutiny over cloud business practices",
            url="https://marketwatch.com/amzn-ftc",
            source_name="MarketWatch",
            published_at=now,
        ),
        RawArticle(
            title="Tesla vehicle deliveries beat quarterly expectations in Europe",
            url="https://cnbc.com/tsla-deliveries",
            source_name="CNBC",
            published_at=now,
        ),
    ]

    clusters = corroborator.cluster_articles(articles)
    assert len(clusters) == 2

    # Find the FTC Amazon cluster
    amzn_cluster = next(c for c in clusters if "FTC" in c.canonical_article.title)
    assert amzn_cluster.corroboration_count == 3
    assert amzn_cluster.is_multi_source is True
    assert amzn_cluster.sources == {"Reuters", "Bloomberg", "MarketWatch"}
    assert len(amzn_cluster.articles) == 3

    # TSLA cluster
    tsla_cluster = next(c for c in clusters if "Tesla" in c.canonical_article.title)
    assert tsla_cluster.corroboration_count == 1
    assert tsla_cluster.is_multi_source is False


def test_corroboration_boosts_credibility_score_and_status():
    corroborator = EventCorroborator()
    now = datetime.now(timezone.utc)

    cluster = EventCluster(
        canonical_article=RawArticle(
            title="Antitrust probe into Apple opens in EU",
            url="https://reuters.com/aapl",
            source_name="Reuters",
            published_at=now,
        ),
        sources={"Reuters", "Bloomberg", "Financial Times"},
    )

    initial_analysis = MarketImpactAnalysis(
        is_market_moving=False,
        confidence_score=6,
        impact_grade=ImpactGrade.AVERAGE,
        affected_tickers=["$AAPL"],
        affected_sectors=["Technology"],
        summary="Antitrust scrutiny in Europe.",
        catalyst_thesis="Potential fines.",
        status=EventStatus.WATCHLIST,
        directional_stance=DirectionalStance.BEARISH,
        hard_gates=HardGates(
            is_novel=True,
            is_verified=False,
            has_direct_exposure=True,
            passes_all_gates=False,
        ),
    )

    boosted = corroborator.corroborate_analysis(initial_analysis, cluster)

    assert boosted.corroboration_count == 3
    assert boosted.corroborating_sources == ["Bloomberg", "Financial Times", "Reuters"]
    # 3 sources gives +2 boost: 6 -> 8
    assert boosted.confidence_score == 8
    # Score 8 elevates to HIGH and is_market_moving=True
    assert boosted.impact_grade == ImpactGrade.HIGH
    assert boosted.is_market_moving is True
    # Verification hard gate satisfied
    assert boosted.hard_gates.is_verified is True
    assert boosted.hard_gates.passes_all_gates is True
    # Promoted to OPPORTUNITY_CANDIDATE
    assert boosted.status == EventStatus.OPPORTUNITY_CANDIDATE


def test_pipeline_multi_source_deduplication_and_single_alert():
    """Verify that when 3 sources report the same event in one cycle:
    1. Only ONE consolidated alert is dispatched to Telegram.
    2. The alert displays the corroborating sources.
    3. All articles are recorded in SQLite so none trigger duplicate alerts.
    """
    storage = SQLiteStorage(":memory:")
    notifier = MockNotifier()
    analyzer = MockAnalyzer(base_score=6, base_status=EventStatus.WATCHLIST)
    pipeline = Pipeline(
        storage=storage,
        analyzer=analyzer,
        notifier=notifier,
        min_score=7,
        min_impact_grade=ImpactGrade.HIGH,
    )

    now = datetime.now(timezone.utc)
    articles = [
        RawArticle(
            title="US Commerce Dept drafts strict export ban on $NVDA chips",
            url="https://reuters.com/nvda1",
            source_name="Reuters",
            published_at=now,
        ),
        RawArticle(
            title="Commerce Department drafting strict export ban on $NVDA chips",
            url="https://bloomberg.com/nvda2",
            source_name="Bloomberg",
            published_at=now,
        ),
    ]

    summary = pipeline.process_articles(articles)

    assert summary.fetched == 2
    assert summary.new_unseen == 2
    # Only 1 canonical article analyzed by LLM
    assert summary.analyzed == 1
    # Multi-source boost pushed score from 6 to 7 (HIGH), so it alerted!
    assert summary.alerted == 1

    # Exactly 1 Telegram alert dispatched
    assert len(notifier.sent_alerts) == 1
    sent_art, sent_analysis = notifier.sent_alerts[0]
    assert sent_analysis.corroboration_count == 2
    assert set(sent_analysis.corroborating_sources) == {"Reuters", "Bloomberg"}

    # Both articles are recorded in SQLite
    assert storage.is_article_seen(articles[0].content_hash) is True
    assert storage.is_article_seen(articles[1].content_hash) is True

    # Running another cycle with the exact same articles yields 0 new/unseen
    second_summary = pipeline.process_articles(articles)
    assert second_summary.new_unseen == 0
    assert second_summary.analyzed == 0
    assert second_summary.alerted == 0


def test_cross_newsroom_headline_clustering_delta():
    """Verify that distinct newsroom headlines about the same event cluster cleanly."""
    corroborator = EventCorroborator()
    now = datetime.now(timezone.utc)

    art1 = RawArticle(
        title="Delta cuts flight schedule due to fuel costs",
        url="https://wsj.com/delta-cuts-schedule",
        source_name="Wall Street Journal",
        published_at=now,
    )
    art2 = RawArticle(
        title="Delta trims capacity amid jet fuel spike",
        url="https://bloomberg.com/delta-trims-capacity",
        source_name="Bloomberg",
        published_at=now,
    )

    similarity = corroborator.compute_similarity(art1, art2)
    assert similarity >= corroborator.similarity_threshold
    assert corroborator.are_same_event(art1, art2) is True

    clusters = corroborator.cluster_articles([art1, art2])
    assert len(clusters) == 1
    assert clusters[0].is_multi_source is True
    assert clusters[0].sources == {"Wall Street Journal", "Bloomberg"}


def test_noise_article_detection():
    """Verify detection of zero-value syndicated noise releases like law firm solicitations and routine webcasts."""
    corroborator = EventCorroborator()
    noise1 = RawArticle(
        title="SHAREHOLDER ALERT: Pomerantz Law Firm Investigates Claims on Behalf of Investors",
        url="https://globenewswire.com/noise-1",
        source_name="GlobeNewswire",
    )
    noise2 = RawArticle(
        title="Company XYZ Scheduled to Present at Upcoming Investor Conference",
        url="https://prnewswire.com/noise-2",
        source_name="PR Newswire",
    )
    clean = RawArticle(
        title="Federal Reserve Cuts Interest Rates by 50 Basis Points in Emergency Move",
        url="https://reuters.com/fed-cuts",
        source_name="Reuters",
    )

    assert corroborator.is_noise_article(noise1) is True
    assert corroborator.is_noise_article(noise2) is True
    assert corroborator.is_noise_article(clean) is False

