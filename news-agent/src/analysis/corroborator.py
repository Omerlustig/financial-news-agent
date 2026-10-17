import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set
import difflib

from src.models import (
    EventStatus,
    HardGates,
    ImpactGrade,
    MarketImpactAnalysis,
    RawArticle,
)

STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "says", "said", "amid", "amidst", "new", "report",
    "reports", "first", "update", "exclusive", "breaking", "news"
}


@dataclass
class EventCluster:
    """A cluster of news articles representing the same real-world financial catalyst event."""
    canonical_article: RawArticle
    articles: List[RawArticle] = field(default_factory=list)
    sources: Set[str] = field(default_factory=set)

    @property
    def corroboration_count(self) -> int:
        """Total number of distinct publishers reporting this event."""
        return len(self.sources)

    @property
    def is_multi_source(self) -> bool:
        """True if corroborated by two or more distinct publishers."""
        return self.corroboration_count >= 2


class EventCorroborator:
    """Discovers, clusters, and corroborates multi-source breaking news events.

    Provides cross-source verification, credibility weighting boosts, and
    deduplication across distinct financial newsrooms.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.35,
        time_window_hours: float = 24.0,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.time_window_hours = time_window_hours

    def extract_tokens(self, text: str) -> Set[str]:
        """Normalize and tokenize text into informative keyword tokens."""
        if not text:
            return set()
        # Clean non-alphanumeric (keep '$' for tickers and '%' for percentages)
        cleaned = re.sub(r"[^\w\$\%\s]", " ", text.lower())
        tokens = cleaned.split()
        return {t for t in tokens if len(t) > 1 and t not in STOPWORDS}

    def compute_similarity(self, a: RawArticle, b: RawArticle) -> float:
        """Compute headline and content semantic similarity between two articles."""
        tokens_a = self.extract_tokens(a.title)
        tokens_b = self.extract_tokens(b.title)

        if not tokens_a or not tokens_b:
            return 0.0

        # 1. Jaccard similarity of informative headline tokens
        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b
        jaccard = len(intersection) / len(union) if union else 0.0

        # 2. Sequence matcher ratio on normalized titles for exact or phrase matches
        seq_ratio = difflib.SequenceMatcher(None, a.title.lower(), b.title.lower()).ratio()

        # Weighted combination: headline token overlap + title sequence similarity
        combined = 0.6 * jaccard + 0.4 * seq_ratio

        # Boost if both explicitly mention the exact same tradeable equity ticker ($NVDA, etc.)
        tickers_a = {t for t in tokens_a if t.startswith("$")}
        tickers_b = {t for t in tokens_b if t.startswith("$")}
        if tickers_a and tickers_b and (tickers_a & tickers_b):
            combined = min(1.0, combined + 0.15)

        return combined

    def are_same_event(self, a: RawArticle, b: RawArticle) -> bool:
        """Determine if two articles refer to the same event based on time and similarity."""
        # Check time window if publication dates are present
        if a.published_at and b.published_at:
            delta_seconds = abs((a.published_at - b.published_at).total_seconds())
            if delta_seconds > self.time_window_hours * 3600:
                return False

        sim = self.compute_similarity(a, b)
        return sim >= self.similarity_threshold

    def cluster_articles(self, articles: List[RawArticle]) -> List[EventCluster]:
        """Cluster a collection of articles into distinct event clusters.

        Articles reporting the same event are grouped together, capturing all
        corroborating publishers.
        """
        clusters: List[EventCluster] = []

        for article in articles:
            matched_cluster: Optional[EventCluster] = None
            best_sim = 0.0

            for cluster in clusters:
                sim = self.compute_similarity(article, cluster.canonical_article)
                if sim >= self.similarity_threshold and sim > best_sim:
                    # Check time window
                    if article.published_at and cluster.canonical_article.published_at:
                        delta_seconds = abs((article.published_at - cluster.canonical_article.published_at).total_seconds())
                        if delta_seconds > self.time_window_hours * 3600:
                            continue
                    best_sim = sim
                    matched_cluster = cluster

            if matched_cluster is not None:
                matched_cluster.articles.append(article)
                matched_cluster.sources.add(article.source_name)
                # If the new article has longer content or is from a primary wire, keep the richest
                if len(article.content or article.summary or "") > len(
                    matched_cluster.canonical_article.content or matched_cluster.canonical_article.summary or ""
                ):
                    matched_cluster.canonical_article = article
            else:
                clusters.append(
                    EventCluster(
                        canonical_article=article,
                        articles=[article],
                        sources={article.source_name},
                    )
                )

        return clusters

    def corroborate_analysis(
        self,
        analysis: MarketImpactAnalysis,
        cluster: EventCluster,
    ) -> MarketImpactAnalysis:
        """Apply multi-source corroboration weights, credibility boosts, and verification gates.

        If an event appears across multiple independent publishers:
        - Automatically corroborates and validates the event (hard gate `is_verified = True`).
        - Boosts the confidence score (+1 for 2 sources, +2 for 3+ sources).
        - Elevates workflow status from WATCHLIST/UNVERIFIED to OPPORTUNITY_CANDIDATE if all hard gates pass.
        - Populates corroborating_sources and corroboration_count.
        """
        corroborating_sources = sorted(list(cluster.sources))
        analysis.corroborating_sources = corroborating_sources
        analysis.corroboration_count = len(corroborating_sources)

        if cluster.is_multi_source:
            # 1. Multi-source independent corroboration satisfies the verification hard gate
            if analysis.hard_gates is not None:
                analysis.hard_gates.is_verified = True
                if analysis.hard_gates.is_novel and analysis.hard_gates.has_direct_exposure:
                    analysis.hard_gates.passes_all_gates = True

            # 2. Confidence score boost
            boost = 2 if analysis.corroboration_count >= 3 else 1
            new_score = min(10, analysis.confidence_score + boost)
            analysis.confidence_score = new_score

            # 3. Elevate impact grade if score crosses threshold
            if new_score >= 9:
                analysis.impact_grade = ImpactGrade.CRITICAL
                analysis.is_market_moving = True
            elif new_score >= 7 and analysis.impact_grade in (ImpactGrade.LOW, ImpactGrade.AVERAGE):
                analysis.impact_grade = ImpactGrade.HIGH
                analysis.is_market_moving = True

            # 4. Status promotion to OPPORTUNITY_CANDIDATE if hard gates pass
            if analysis.hard_gates is None or analysis.hard_gates.passes_all_gates:
                if analysis.status in (EventStatus.UNVERIFIED, EventStatus.WATCHLIST):
                    analysis.status = EventStatus.OPPORTUNITY_CANDIDATE

        return analysis
