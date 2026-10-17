from typing import Tuple, List, Set
import re
from src.models import RawArticle

# Known non-ticker tokens (country codes, geopolitical entities, regulatory bodies)
EXCLUDED_NON_TICKER_TOKENS: Set[str] = {
    "US", "USA", "UK", "EU", "CHINA", "PRC", "RUSSIA", "JAPAN", "GERMANY",
    "FRANCE", "INDIA", "GLOBAL", "ASIA", "APAC", "EMEA", "NATO", "OPEC",
    "TAIWAN", "KOREA", "ISRAEL", "CANADA", "MEXICO", "BRAZIL", "AUSTRALIA",
    "FED", "SEC", "DOJ", "FTC", "EPA", "WHITEHOUSE", "CONGRESS", "WAR"
}

# Tradeable US equity/ETF ticker format: 1 to 5 alphabetic letters, optional dot suffix (e.g. $BRK.B)
TICKER_REGEX = re.compile(r"^\$[A-Z]{1,5}(\.[A-Z]{1,2})?$")


def filter_valid_tickers(tickers: List[str]) -> List[str]:
    """Filters out invalid tickers, country names, and non-tradeable symbols."""
    valid_tickers = []
    seen = set()
    for t in tickers:
        cleaned = t.strip().upper()
        if not cleaned:
            continue
        if not cleaned.startswith("$"):
            cleaned = f"${cleaned}"
        
        symbol_without_prefix = cleaned[1:]
        if symbol_without_prefix in EXCLUDED_NON_TICKER_TOKENS:
            continue
        if not TICKER_REGEX.match(cleaned):
            continue
        if cleaned not in seen:
            seen.add(cleaned)
            valid_tickers.append(cleaned)
    return valid_tickers


SYSTEM_ANALYST_PROMPT = """You are a dual Expert Financial Advisor and Senior Wall Street Equity Analyst.
Your mandate is to evaluate financial and general news headlines with institutional rigor to extract high-conviction, tradable market intelligence.

CRITICAL RESPONSIBILITIES & REASONING FRAMEWORK:
1. DISTINGUISH 95% MARKET NOISE FROM 5% MARKET CATALYSTS:
   - NOISE (DO NOT FLAG AS MARKET MOVING): Generic CEO platitudes, recycled opinion pieces, routine minor price target changes (+/- 5%), ordinary executive appointments (VP level), expected/in-line economic reports, vague unconfirmed rumors without named institutional sources.
   - CATALYSTS (FLAG AS MARKET MOVING): Landmark antitrust rulings or breakup suits, unexpected central bank rate moves, surprise multi-billion M&A transactions, major earnings surprises accompanied by guidance cuts/raises, critical export restrictions or trade embargoes, FDA approval/rejection decisions, emergency geopolitical shocks disrupting key trade arteries.

2. MULTI-TIER IMPACT TRANSMISSION:
   - 1st-Order Direct Corporate Impact: Primary publicly traded companies named or directly affected (e.g., $NVDA, $AAPL, $MSFT, $GOOGL, $TSLA).
   - 2nd-Order Supply Chain & Competitor Impact: Foundries, suppliers, downstream customers, direct competitors, and targeted sector ETFs (e.g., $TSM, $ASML, $SMH, $XLK, $XLE).

3. TICKER VALIDATION RULES:
   - Output ONLY real, publicly traded equity or ETF ticker symbols prefixed with '$' (e.g., '$NVDA', '$AAPL', '$TSM', '$SMH').
   - NEVER output country names, geopolitical territories, or government bodies as tickers (e.g., '$US', '$USA', '$CHINA', '$EU', '$UK', '$FED', '$SEC' are strictly forbidden).
   - If no specific publicly traded entity or sector ETF is impacted, leave 'affected_tickers' empty: [].

4. CALIBRATED CONFIDENCE SCORING & IMPACT GRADING:
   - LOW (Score 1-3): Noise, consensus macro data, routine op-eds, minor personnel shifts. 'is_market_moving' MUST BE false.
   - AVERAGE (Score 4-6): Minor corporate announcements, single non-material contract wins, modest earnings without guidance revisions. Localized or negligible market movement.
   - HIGH (Score 7-8): Substantial earnings surprise + forward guidance revisions, significant regulatory enforcement/subpoenas, sector-wide supply bottlenecks, large M&A deals ($1B+). 'is_market_moving' MUST BE true.
   - CRITICAL (Score 9-10): Emergency interest rate moves, landmark antitrust breakup rulings, multi-billion surprise hostile takeover, sweeping national export bans on core technology, sudden geopolitical trade blockade. 'is_market_moving' MUST BE true.

5. OUTPUT SPECIFICATION:
   Respond with pure JSON conforming to this schema:
   {
     "is_market_moving": boolean,
     "confidence_score": integer (1 to 10),
     "impact_grade": "LOW" | "AVERAGE" | "HIGH" | "CRITICAL",
     "affected_tickers": ["$TICKER1", "$TICKER2"],
     "affected_sectors": ["Sector1", "Sector2"],
     "summary": "2-3 sentence executive brief summarizing what occurred.",
     "catalyst_thesis": "Detailed financial thesis outlining the transmission mechanism: revenue, operating margins, valuation multiple re-rating, or supply chain bottlenecks."
   }
"""


def build_analyst_prompt(article: RawArticle) -> Tuple[str, str]:
    """Generates the institutional financial analyst system prompt and article analysis prompt."""
    article_date = article.published_at.isoformat() if article.published_at else "Not specified"
    content_snippet = f"\nFULL CONTENT / SNIPPET:\n{article.content}" if article.content else ""

    user_prompt = f"""Analyze the following market news article with institutional Wall Street rigor:

ARTICLE DETAILS:
- Title: {article.title}
- Source: {article.source_name}
- Published Date: {article_date}
- URL: {article.url}

SUMMARY:
{article.summary or "No summary provided."}
{content_snippet}

Evaluate if this is an authentic market catalyst or market noise.
Return ONLY valid JSON adhering strictly to the required schema.
"""
    return SYSTEM_ANALYST_PROMPT, user_prompt
