import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set
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

# Common financial news synonym and action clusters
SYNONYM_MAP: Dict[str, str] = {
    # Cuts / Reductions
    "cuts": "token_cut", "cut": "token_cut", "cutting": "token_cut",
    "trims": "token_cut", "trim": "token_cut", "trimming": "token_cut",
    "slashes": "token_cut", "slash": "token_cut", "slashing": "token_cut",
    "reduces": "token_cut", "reduce": "token_cut", "reducing": "token_cut",
    "curtails": "token_cut", "curtail": "token_cut", "curtailing": "token_cut",
    # Raises / Hikes
    "hikes": "token_hike", "hike": "token_hike", "hiking": "token_hike",
    "raises": "token_hike", "raise": "token_hike", "raising": "token_hike",
    "boosts": "token_hike", "boost": "token_hike", "boosting": "token_hike",
    "lifts": "token_hike", "lift": "token_hike", "lifting": "token_hike",
    "increases": "token_hike", "increase": "token_hike", "increasing": "token_hike",
    # Drops / Declines
    "drops": "token_drop", "drop": "token_drop", "dropping": "token_drop",
    "falls": "token_drop", "fall": "token_drop", "falling": "token_drop",
    "plunges": "token_drop", "plunge": "token_drop", "plunging": "token_drop",
    "tumbles": "token_drop", "tumble": "token_drop", "tumbling": "token_drop",
    "slides": "token_drop", "slide": "token_drop", "sliding": "token_drop",
    "sinks": "token_drop", "sink": "token_drop", "sinking": "token_drop",
    "declines": "token_drop", "decline": "token_drop", "declining": "token_drop",
    # Spikes / Surges
    "spikes": "token_surge", "spike": "token_surge", "spiking": "token_surge",
    "surges": "token_surge", "surge": "token_surge", "surging": "token_surge",
    "jumps": "token_surge", "jump": "token_surge", "jumping": "token_surge",
    "soars": "token_surge", "soar": "token_surge", "soaring": "token_surge",
    "rallies": "token_surge", "rally": "token_surge", "rallying": "token_surge",
    # Aviation / Capacity / Schedules
    "flight": "token_aviation_capacity", "flights": "token_aviation_capacity",
    "capacity": "token_aviation_capacity",
    "schedule": "token_aviation_capacity", "schedules": "token_aviation_capacity",
    "routes": "token_aviation_capacity", "route": "token_aviation_capacity",
    # Fuel / Energy
    "fuel": "token_fuel", "fuels": "token_fuel",
    "oil": "token_fuel", "crude": "token_fuel",
    "kerosene": "token_fuel",
    # Costs / Expenditure
    "costs": "token_cost", "cost": "token_cost", "expenses": "token_cost",
}

# Mapping common company / entity names to tradeable tickers
COMPANY_TICKER_MAP: Dict[str, str] = {
    "delta": "$dal",
    "delta air lines": "$dal",
    "delta airlines": "$dal",
    "apple": "$aapl",
    "microsoft": "$msft",
    "google": "$googl",
    "alphabet": "$googl",
    "amazon": "$amzn",
    "aws": "$amzn",
    "nvidia": "$nvda",
    "tesla": "$tsla",
    "meta": "$meta",
    "facebook": "$meta",
    "boeing": "$ba",
    "intel": "$intc",
    "amd": "$amd",
    "starbucks": "$sbux",
    "exxon": "$xom",
    "exxonmobil": "$xom",
    "chevron": "$cvx",
    "jpmorgan": "$jpm",
    "chase": "$jpm",
    "goldman": "$gs",
    "goldman sachs": "$gs",
    "netflix": "$nflx",
    "palantir": "$pltr",
}

# Sector keywords for domain entity clustering
SECTOR_KEYWORDS: Set[str] = {
    "aviation", "airlines", "semiconductors", "chips", "defense", "energy",
    "banking", "technology", "pharmaceuticals", "biotech", "retail",
    "telecom", "automotive", "cloud", "artificial intelligence", "ai",
}

# Regex patterns for zero-value syndicated noise releases
NOISE_HEADLINE_PATTERNS = [
    r"\bshareholder\s+(?:alert|notice|lawsuit|action)\b",
    r"\bclass\s+action\s+(?:lawsuit|filed|alert|deadline)\b",
    r"\bdeadline\s+reminder\b",
    r"\binvestor\s+(?:notice|reminder|alert):\s*(?:the\s+)?(?:rosen|pomerantz|hagens|bronstein|schall|glancy|robbins|bragar|faruqi|kessler|bernstein)",
    r"\blaw\s+firm\s+(?:announces|reminds|investigates|files)\b",
    r"\bto\s+host\s+(?:earnings|quarterly|fourth|third|second|first\s+quarter)?\s*(?:conference\s+call|webcast)\b",
    r"\bannounces\s+(?:timing\s+of\s+)?(?:quarterly|first|second|third|fourth\s+quarter\s+)?(?:conference\s+call|webcast|earnings\s+release\s+date)\b",
    r"\bscheduled\s+to\s+present\s+at\s+upcoming\s+investor\s+conference\b",
    r"\bparticipate\s+in\s+(?:upcoming\s+)?investor\s+conferences?\b",
    r"\bdeclares\s+regular\s+quarterly\s+dividend\b",
]


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
        watchlist_settings: Optional[Any] = None,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.time_window_hours = time_window_hours
        self.watchlist_settings = watchlist_settings

    def is_noise_article(self, article: RawArticle) -> bool:
        """Detect and suppress zero-value syndicated noise releases (e.g. law firm solicitations, routine webcasts)."""
        if not article or not article.title:
            return True
        title_lower = article.title.lower()
        for pattern in NOISE_HEADLINE_PATTERNS:
            if re.search(pattern, title_lower):
                return True
        return False

    def extract_tokens(self, text: str) -> Set[str]:
        """Normalize and tokenize text into informative keyword tokens, company entities, tickers, and action concepts."""
        if not text:
            return set()

        tokens: Set[str] = set()

        # 1. Extract explicit tickers ($NVDA, $AAPL, etc.) from raw text
        raw_tickers = re.findall(r"\$[A-Za-z]{1,5}\b", text)
        for t in raw_tickers:
            tokens.add(t.lower())

        # 2. Extract capitalized multi-word entity names (e.g. 'Delta Air Lines', 'Jet Fuel', 'White House')
        multi_word_entities = re.findall(r"\b[A-Z][a-zA-Z0-9]+(?:\s+[A-Z][a-zA-Z0-9]+)+\b", text)
        for entity in multi_word_entities:
            ent_lower = entity.lower()
            if ent_lower not in STOPWORDS:
                tokens.add(ent_lower)
                # Check if multi-word entity maps to a known ticker
                if ent_lower in COMPANY_TICKER_MAP:
                    tokens.add(COMPANY_TICKER_MAP[ent_lower])

        # 3. Clean non-alphanumeric (keep '$' and '%' for tickers/rates)
        cleaned = re.sub(r"[^\w\$\%\s]", " ", text.lower())
        words = cleaned.split()

        # 4. Check for 2-word phrases in cleaned text (e.g. "jet fuel", "air lines")
        for i in range(len(words) - 1):
            bigram = f"{words[i]} {words[i+1]}"
            if bigram == "jet fuel":
                tokens.add("jet fuel")
                tokens.add("token_fuel")
            elif bigram in COMPANY_TICKER_MAP:
                tokens.add(COMPANY_TICKER_MAP[bigram])

        # 5. Process individual words, check company map, synonym clusters, and sector keywords
        for w in words:
            if len(w) <= 1 or w in STOPWORDS:
                continue
            tokens.add(w)

            # Map company name to ticker
            if w in COMPANY_TICKER_MAP:
                tokens.add(COMPANY_TICKER_MAP[w])

            # Map synonym / action verbs to canonical cluster token
            if w in SYNONYM_MAP:
                tokens.add(SYNONYM_MAP[w])

            # Sector keyword tag
            if w in SECTOR_KEYWORDS:
                tokens.add(f"sector_{w}")

        return tokens

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

        # Boost if both explicitly or implicitly mention the exact same tradeable equity ticker ($NVDA, $DAL, etc.)
        tickers_a = {t for t in tokens_a if t.startswith("$")}
        tickers_b = {t for t in tokens_b if t.startswith("$")}
        if tickers_a and tickers_b and (tickers_a & tickers_b):
            combined = min(1.0, combined + 0.15)
        else:
            # Boost if both share a common named company entity or action concept
            shared_canonical = {t for t in intersection if t.startswith("token_") or t in COMPANY_TICKER_MAP}
            if len(shared_canonical) >= 2:
                combined = min(1.0, combined + 0.15)
            elif len(shared_canonical) == 1:
                combined = min(1.0, combined + 0.10)

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
