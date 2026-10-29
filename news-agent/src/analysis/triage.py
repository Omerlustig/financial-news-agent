"""
Economic Relevance Triage Filter for Market News Agent.
High-speed pre-screening filter that discards non-economic general news
(sports, celebrity gossip, routine politics) from breaking RSS feeds with zero LLM token cost.
"""
from __future__ import annotations

import re
from typing import Set
from src.models import RawArticle

# Known financial feeds that never need filtering (always pass triage)
FINANCIAL_SOURCE_KEYWORDS: Set[str] = {
    "reuters", "bloomberg", "cnbc", "investing", "seeking alpha",
    "marketwatch", "calcalist", "themarker", "bizportal", "sponser",
    "globes", "sec edgar", "maya", "tase", "federal reserve",
    "financial times", "ft.com", "finurls", "yahoo finance", "benzinga", "eia",
}

# Multilingual keywords signaling potential macroeconomic, sector, or single-stock transmission
ECONOMIC_TRANSMISSION_KEYWORDS = [
    # Aviation & Transport / Logistics
    r"\bflights?\b", r"\bairlines?\b", r"\bairports?\b", r"\bairspace\b",
    r"\bcarriers?\b", r"\bcargo\b", r"\bshipping\b", r"\bstraits?\b",
    r"\bports?\b", r"\bvessels?\b", r"\bfreight\b",
    r"תעופה", r"טיס(ה|ות)", r"מטוס(ים)?", r"נתב\"?ג", r"נמל(ים)?", r"ספנות", r"אל[\s-]על",

    # Defense, Geopolitics & Military
    r"\bmissiles?\b", r"\bdrones?\b", r"\bstrikes?\b", r"\bdefense\b",
    r"\bpentagon\b", r"\bmilitary\b", r"\bweapons?\b", r"\bwar\b", r"\bsecurity\b",
    r"טיל(ים)?", r"כטב\"?מ", r"רחפן", r"תקיפ(ה|ות)", r"ביטחון", r"צה\"?ל", r"הרוגים",

    # Energy, Commodities & Chokepoints
    r"\boil\b", r"\bgas\b", r"\bcrude\b", r"\bopec\b", r"\bpetroleum\b",
    r"\brefiner(y|ies)\b", r"\bpipelines?\b", r"\benergy\b",
    r"נפט", r"גז", r"חבית", r"זיקוק", r"צינור", r"אופ\"?ק",

    # Trade, Tariffs, Sanctions & Sovereign Policy
    r"\btariffs?\b", r"\bsanctions?\b", r"\bembargo\b", r"\bexport\s+bans?\b",
    r"\bsubsid(y|ies)\b", r"\btrade\s+war\b", r"\bantitrust\b", r"\bregulat(ion|ory)\b",
    r"סנקצי(ה|ות)", r"אמברגו", r"מכס(ים)?", r"היטל", r"חרם",

    # Central Banks, Currencies & Rates
    r"\binflation\b", r"\binterest\s+rates?\b", r"\bcentral\s+banks?\b",
    r"\btreasur(y|ies)\b", r"\byields?\b", r"\bcurrency\b", r"\bdollars?\b", r"\bshekels?\b",
    r"אינפלציה", r"ריבית", r"בנק\s+ישראל", r"מדד", r"שקל", r"דולר",

    # Corporate Inflections & Labor Shocks
    r"\bstrikes?\b", r"\blabor\b", r"\bunions?\b", r"\bbankrupt(cy)?\b",
    r"\bdefaults?\b", r"\bacquisitions?\b", r"\bmergers?\b", r"\bearnings?\b",
    r"\brevenues?\b", r"\bguidance\b", r"\bcapex\b",
    r"שבית(ה|ות)", r"הסתדרות", r"מיזוג", r"רכיש(ה|ות)", r"פשיטת\s+רגל", r"בורסה", r"מניות"
]

RE_ECONOMIC_PATTERN = re.compile("|".join(ECONOMIC_TRANSMISSION_KEYWORDS), re.IGNORECASE)


def is_economically_relevant(article: RawArticle) -> bool:
    """Evaluate whether an article possesses tradeable economic or sector transmission potential.
    
    Dedicated financial sources always pass triage. General breaking news, politics,
    or world news feeds are checked against the multilingual economic transmission pattern.
    """
    source_lower = (article.source_name or "").lower()
    for fin_src in FINANCIAL_SOURCE_KEYWORDS:
        if fin_src in source_lower:
            return True

    # Combine title, summary, and content snippet for heuristic matching
    content = f"{article.title or ''} {article.summary or ''} {article.content or ''}"
    if not content.strip():
        return False

    return bool(RE_ECONOMIC_PATTERN.search(content))
