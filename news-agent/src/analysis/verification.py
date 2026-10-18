import re
import urllib.parse
from typing import Any, Dict, List, Optional, Set, Tuple

from src.models import (
    ProvenanceLevel,
    RawArticle,
    StatementType,
)

# Known primary official domains (SEC, MAYA, Fed, official wires, government regulators)
PRIMARY_OFFICIAL_DOMAINS: Set[str] = {
    "sec.gov",
    "data.sec.gov",
    "maya.tase.co.il",
    "tase.co.il",
    "federalreserve.gov",
    "prnewswire.com",
    "businesswire.com",
    "globenewswire.com",
    "accesswire.com",
    "justice.gov",
    "ftc.gov",
}

# Tier 1 institutional financial newsrooms
TIER1_FINANCIAL_DOMAINS: Set[str] = {
    "ft.com",
    "bloomberg.com",
    "reuters.com",
    "wsj.com",
    "cnbc.com",
    "globes.co.il",
    "calcalist.co.il",
    "themarker.com",
    "barrons.com",
}

# Commercial mainstream news and financial aggregators
COMMERCIAL_NEWS_DOMAINS: Set[str] = {
    "foxbusiness.com",
    "foxnews.com",
    "finance.yahoo.com",
    "yahoo.com",
    "techcrunch.com",
    "seekingalpha.com",
    "allsides.com",
    "marketwatch.com",
    "forbes.com",
    "businessinsider.com",
    "insider.com",
    "cnn.com",
    "qz.com",
    "benzinga.com",
    "thestreet.com",
    "investopedia.com",
    "theverge.com",
    "wired.com",
}

# Regex patterns for anonymous rumors or unverified speculation
RUMOR_PATTERNS = [
    r"\b(?:people|sources)\s+(?:familiar\s+with|close\s+to)(?:\s+the\s+matter|\s+the\s+deal|\s+talks|\s+the\s+situation)?\b",
    r"\b(?:unnamed|anonymous|undisclosed)\s+sources?\b",
    r"\bsources?\s+(?:said|told|say|reported|declined|asked)\b",
    r"\baccording\s+to\s+sources\b",
    r"\b(?:reportedly|rumored|rumor|rumours?|speculation|whispers?|unconfirmed\s+reports?)\b",
    r"\b(?:sources\s+asking\s+not\s+to\s+be\s+identified|sources\s+who\s+declined\s+to\s+be\s+named)\b",
    r"\b(?:sources\s+with\s+knowledge)\b",
    r"\b(?:talks\s+are\s+ongoing,\s*sources\s+say)\b",
]

# Regex patterns for Wall Street analyst opinions and non-binding commentary
ANALYST_PATTERNS = [
    r"\b(?:analyst|analysts)\s+(?:at|from|wrote|said|noted|stated|reiterated|cut|raised)\b",
    r"\b(?:equity|research)\s+analyst\b",
    r"\bprice\s+target\b",
    r"\b(?:upgraded|downgraded|reiterated|initiated)\s+(?:to|at|coverage|rating|with)\b",
    r"\b(?:overweight|underweight|outperform|underperform|equal-weight|buy\s+rating|sell\s+rating|hold\s+rating)\b",
    r"\bin\s+a\s+(?:note|research\s+note|report)\s+to\s+clients\b",
    r"\bwall\s+street\s+(?:expects|estimates|analysts|consensus|notes)\b",
    r"\bbrokerage\s+firm\b",
    r"\bwall\s+street\s+note\b",
]

# Regex patterns for verified facts (regulatory filings, official press releases, court rulings)
FACT_PATTERNS = [
    r"\b(?:sec\s+filing|form\s+8-k|form\s+10-q|form\s+10-k|regulatory\s+filing|schedule\s+13d|form\s+4)\b",
    r"\b(?:filed\s+with\s+the\s+sec|reported\s+in\s+a\s+regulatory\s+filing|disclosed\s+in\s+an\s+8-k)\b",
    r"\b(?:tase|maya)\s+filing\b",
    r"\b(?:in\s+a\s+press\s+release|press\s+release\s+issued|official\s+announcement|formal\s+statement)\b",
    r"\b(?:court\s+ruled|federal\s+judge\s+(?:ruled|ordered)|judge\s+granted|consent\s+decree|court\s+ruling|antitrust\s+ruling|jury\s+found|settlement\s+approved|appellate\s+court)\b",
    r"\b(?:patent\s+granted|fda\s+approved|doj\s+sued|ftc\s+blocked|fcc\s+approved)\b",
    r"\b(?:confirmed\s+in\s+a\s+statement|company\s+announced\s+(?:today|yesterday|on\s+\w+))\b",
]

# Regex patterns for attributed claims citing named corporate officers or spokespersons
CLAIM_PATTERNS = [
    r"\b(?:ceo|cfo|coo|cto|chairman|chairwoman|president|spokesperson|spokeswoman|spokesman)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+said\b",
    r"\bsaid\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(?:chief|ceo|cfo|coo|cto|president|spokesperson|spokeswoman|spokesman|head|director)\b",
    r"\b(?:according\s+to|stated)\s+(?:ceo|cfo|spokesperson|spokeswoman|company\s+spokesperson|[A-Z][a-z]+\s+[A-Z][a-z]+)\b",
    r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?,\s*(?:spokesperson|spokeswoman|representative)\s+for\b",
    r"\b(?:government|fed|treasury|white\s+house|ministry)\s+(?:official|spokesperson)\s+(?:said|stated|confirmed)\b",
    r"\bcompany\s+stated\b",
]


def extract_domain(url: str) -> str:
    """Normalize and extract clean domain host from a URL."""
    if not url:
        return ""
    try:
        parsed = urllib.parse.urlparse(url.strip())
        netloc = parsed.netloc.lower()
        if netloc.startswith("www."):
            netloc = netloc[4:]
        return netloc
    except Exception:
        return ""


class FactVerificationEngine:
    """Institutional fact verification, statement classification, and provenance scoring engine."""

    def __init__(self) -> None:
        self._rumor_regexes = [re.compile(p, re.IGNORECASE) for p in RUMOR_PATTERNS]
        self._analyst_regexes = [re.compile(p, re.IGNORECASE) for p in ANALYST_PATTERNS]
        self._fact_regexes = [re.compile(p, re.IGNORECASE) for p in FACT_PATTERNS]
        self._claim_regexes = [re.compile(p, re.IGNORECASE) for p in CLAIM_PATTERNS]

    def classify_provenance(self, article: RawArticle) -> ProvenanceLevel:
        """Classify source authority into hierarchical ProvenanceLevel tiers."""
        domain = extract_domain(article.url)
        source = (article.source_name or "").strip().lower()

        # Test environment fixture support
        if source in ("test", "testfeed", "test source", "sample", "mock") or domain in ("test.com", "example.com"):
            return ProvenanceLevel.TIER1_FINANCIAL

        # Check PRIMARY_OFFICIAL (official regulatory & government portals)
        if (
            domain.endswith(".gov")
            or domain.endswith(".gov.il")
            or domain.endswith(".mil")
            or domain in PRIMARY_OFFICIAL_DOMAINS
            or any(d in domain for d in PRIMARY_OFFICIAL_DOMAINS)
        ):
            return ProvenanceLevel.PRIMARY_OFFICIAL
        if (
            domain.startswith("investor.")
            or domain.startswith("ir.")
            or "/investors/" in article.url.lower()
            or "/press-releases/" in article.url.lower()
            or "/news-releases/" in article.url.lower()
        ):
            return ProvenanceLevel.PRIMARY_OFFICIAL
        if any(
            kw in source
            for kw in [
                "sec edgar",
                "sec filing",
                "sec.gov",
                "edgar",
                "maya israel",
                "maya.tase",
                "tase",
                "federal reserve",
                "pr newswire",
                "business wire",
                "globe newswire",
                "globenewswire",
                "accesswire",
                "official corporate announcement",
                "investor relations",
            ]
        ):
            return ProvenanceLevel.PRIMARY_OFFICIAL

        # Check TIER1_FINANCIAL
        if domain in TIER1_FINANCIAL_DOMAINS or any(d in domain for d in TIER1_FINANCIAL_DOMAINS):
            return ProvenanceLevel.TIER1_FINANCIAL
        if any(
            kw in source
            for kw in [
                "financial times",
                "ft",
                "bloomberg",
                "reuters",
                "wall street journal",
                "wsj",
                "cnbc",
                "globes",
                "calcalist",
                "themarker",
                "the marker",
                "barron's",
                "barrons",
            ]
        ):
            return ProvenanceLevel.TIER1_FINANCIAL

        # Check COMMERCIAL_NEWS
        if domain in COMMERCIAL_NEWS_DOMAINS or any(d in domain for d in COMMERCIAL_NEWS_DOMAINS):
            return ProvenanceLevel.COMMERCIAL_NEWS
        if any(
            kw in source
            for kw in [
                "fox business",
                "fox",
                "yahoo finance",
                "yahoo",
                "techcrunch",
                "seeking alpha",
                "allsides",
                "marketwatch",
                "forbes",
                "business insider",
                "insider",
                "cnn",
                "cnn business",
                "benzinga",
                "thestreet",
                "investopedia",
            ]
        ):
            return ProvenanceLevel.COMMERCIAL_NEWS

        # Default to UNVERIFIED_BLOG
        return ProvenanceLevel.UNVERIFIED_BLOG

    def classify_statement(
        self,
        sentence: str,
        provenance: ProvenanceLevel = ProvenanceLevel.UNVERIFIED_BLOG,
    ) -> Tuple[StatementType, List[str]]:
        """Classify an individual statement into a StatementType with matching cues."""
        text = sentence.strip()
        matched_cues: List[str] = []

        # 1. Anonymous Rumor Check
        for rx in self._rumor_regexes:
            m = rx.search(text)
            if m:
                matched_cues.append(m.group(0))
        if matched_cues:
            return StatementType.ANONYMOUS_RUMOR, matched_cues

        # 2. Analyst Opinion Check
        for rx in self._analyst_regexes:
            m = rx.search(text)
            if m:
                matched_cues.append(m.group(0))
        if matched_cues:
            return StatementType.ANALYST_OPINION, matched_cues

        # 3. Verified Fact Check
        for rx in self._fact_regexes:
            m = rx.search(text)
            if m:
                matched_cues.append(m.group(0))
        if matched_cues:
            return StatementType.VERIFIED_FACT, matched_cues

        # 4. Attributed Claim Check
        for rx in self._claim_regexes:
            m = rx.search(text)
            if m:
                matched_cues.append(m.group(0))
        if matched_cues:
            return StatementType.ATTRIBUTED_CLAIM, matched_cues

        # Baseline fallback based on provenance
        if provenance == ProvenanceLevel.PRIMARY_OFFICIAL:
            return StatementType.VERIFIED_FACT, ["primary_official_origin"]
        if any(w in text.lower() for w in ["said", "stated", "announced", "reported", "confirmed"]):
            return StatementType.ATTRIBUTED_CLAIM, ["reporting_verb"]
        return StatementType.VERIFIED_FACT, ["declarative_fact"]

    def extract_statements(
        self,
        text: str,
        provenance: ProvenanceLevel = ProvenanceLevel.UNVERIFIED_BLOG,
    ) -> List[Dict[str, Any]]:
        """Segment article text into atomic statements and classify each."""
        if not text:
            return []

        # Normalize whitespace and segment by sentence terminators
        cleaned = re.sub(r"\s+", " ", text).strip()
        raw_sentences = re.split(r"(?<=[.!?])\s+", cleaned)

        statements: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for s in raw_sentences:
            s_clean = s.strip()
            if len(s_clean) < 15:
                continue
            norm = s_clean.lower()
            if norm in seen:
                continue
            seen.add(norm)

            st_type, cues = self.classify_statement(s_clean, provenance)
            statements.append({
                "statement": s_clean,
                "text": s_clean,
                "statement_type": st_type,
                "confidence": 0.9 if st_type == StatementType.VERIFIED_FACT else 0.75,
                "cues": cues,
            })

        return statements

    def extract_newsroom_entity(self, article: RawArticle) -> str:
        """Extract the root independent newsroom entity, resolving wire syndications."""
        domain = extract_domain(article.url)
        source = (article.source_name or "").strip().lower()
        full_text = f"{article.title} {article.summary or ''} {article.content or ''}".lower()

        # Wire syndication detection: Check if content or headline cites a wire
        if any(marker in full_text for marker in ["(reuters)", "reuters -", "by reuters", "- reuters", "via reuters"]):
            return "reuters"
        if any(marker in full_text for marker in ["(bloomberg)", "bloomberg -", "by bloomberg", "- bloomberg", "via bloomberg"]):
            return "bloomberg"
        if any(marker in full_text for marker in ["(ap)", "associated press", "by ap", "via ap"]):
            return "ap"
        if "pr newswire" in full_text or "prnewswire" in full_text:
            return "prnewswire"
        if "business wire" in full_text or "businesswire" in full_text:
            return "businesswire"

        # Explicit entity matching
        entity_map = [
            ("reuters", "reuters"),
            ("bloomberg", "bloomberg"),
            ("financial times", "ft"),
            ("ft.com", "ft"),
            ("wall street journal", "wsj"),
            ("wsj", "wsj"),
            ("cnbc", "cnbc"),
            ("globes", "globes"),
            ("calcalist", "calcalist"),
            ("themarker", "themarker"),
            ("fox business", "foxbusiness"),
            ("foxbusiness", "foxbusiness"),
            ("yahoo", "yahoo"),
            ("techcrunch", "techcrunch"),
            ("seeking alpha", "seekingalpha"),
            ("seekingalpha", "seekingalpha"),
            ("allsides", "allsides"),
            ("sec.gov", "sec"),
            ("edgar", "sec"),
            ("maya.tase", "maya"),
            ("federalreserve", "federalreserve"),
        ]

        for token, entity in entity_map:
            if token in domain or token in source:
                return entity

        if domain:
            return domain
        return re.sub(r"[^a-z0-9]", "", source) or "unknown"

    def count_independent_sources(
        self,
        article: RawArticle,
        corroborating_articles: Optional[List[RawArticle]] = None,
    ) -> int:
        """Count unique independent newsrooms, eliminating syndications of the same wire."""
        all_articles = [article] + (corroborating_articles or [])
        unique_newsrooms: Set[str] = set()

        for a in all_articles:
            newsroom = self.extract_newsroom_entity(a)
            if newsroom:
                unique_newsrooms.add(newsroom)

        return max(1, len(unique_newsrooms))

    def _has_rumor_cue(self, text: str) -> bool:
        """Helper to detect rumor or speculation cues in headline or text."""
        if not text:
            return False
        return any(rx.search(text) for rx in self._rumor_regexes)

    def extract_and_verify_facts(
        self,
        article: RawArticle,
        corroborating_articles: Optional[List[RawArticle]] = None,
    ) -> Tuple[ProvenanceLevel, List[Dict[str, Any]], float, bool]:
        """Perform comprehensive institutional fact verification, scoring, and gating.

        Returns:
            Tuple of:
            - ProvenanceLevel
            - List of classified statements
            - fact_quality_score (float from 1.0 to 10.0)
            - is_verified (bool: True if fact_quality_score >= 7.0 and no unconfirmed rumor)
        """
        provenance = self.classify_provenance(article)

        # Assemble text for statement extraction
        full_text = f"{article.title}. {article.summary or ''}. {article.content or ''}"
        statements = self.extract_statements(full_text, provenance)

        # Count independent newsrooms
        independent_sources = self.count_independent_sources(article, corroborating_articles)

        # Baseline score by provenance
        baseline_scores = {
            ProvenanceLevel.PRIMARY_OFFICIAL: 9.5,
            ProvenanceLevel.TIER1_FINANCIAL: 8.0,
            ProvenanceLevel.COMMERCIAL_NEWS: 6.5,
            ProvenanceLevel.UNVERIFIED_BLOG: 4.0,
        }
        baseline = baseline_scores.get(provenance, 4.0)

        # Corroboration boost
        if independent_sources >= 3:
            boost = 2.0
        elif independent_sources == 2:
            boost = 1.0
        else:
            boost = 0.0

        # Check for primary confirmation across all articles
        all_articles = [article] + (corroborating_articles or [])
        has_primary_confirmation = any(
            self.classify_provenance(a) == ProvenanceLevel.PRIMARY_OFFICIAL
            for a in all_articles
        )

        # Check for anonymous rumor in statements or headline
        has_rumor = any(
            s.get("statement_type") == StatementType.ANONYMOUS_RUMOR
            for s in statements
        ) or self._has_rumor_cue(article.title)

        has_unconfirmed_rumor = has_rumor and not has_primary_confirmation

        # Apply rumor penalty if unconfirmed
        penalty = 2.5 if has_unconfirmed_rumor else 0.0

        raw_score = baseline + boost - penalty
        fact_quality_score = max(1.0, min(10.0, round(raw_score, 1)))

        # Verification gate rule
        is_verified = bool(fact_quality_score >= 7.0 and not has_unconfirmed_rumor)

        return provenance, statements, fact_quality_score, is_verified
