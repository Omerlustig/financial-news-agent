import re
from typing import Optional, Tuple

from src.models import (
    EventLineageRecord,
    NoveltyState,
    RawArticle,
)

# Revision catalysts that overturn or materially alter an existing tracked event
REVISION_PATTERNS = [
    r"\brevis(?:es|ed|ing|ion)\b",
    r"\bdeni(?:es|ed|al|ying)\b",
    r"\bcancel(?:s|led|ed|lation|ing)\b",
    r"\bapprov(?:es|ed|al|ing)\b",
    r"\bhalt(?:s|ed|ing)\b",
    r"\b(?:sec\s+probe|sec\s+investigation|doj\s+probe|doj\s+investigation|subpoena|probe\s+opened)\b",
    r"\brevers(?:es|ed|al|ing)\b",
    r"\bterminat(?:es|ed|ion|ing)\b",
    r"\breject(?:s|ed|ion|ing)\b",
    r"\bblock(?:s|ed|ing)\b",
]

# Genuine shocks and unexpected deviations from consensus expectations
SURPRISE_PATTERNS = [
    r"\bunexpected(?:ly)?\b",
    r"\bsurprise(?:d|s|ing)?\b",
    r"\bsudden(?:ly)?\b",
    r"\bterminat(?:es|ed|ing)\s+(?:deal|merger|agreement|acquisition)\b",
    r"\bscrap(?:s|ped|ping)\s+(?:merger|deal|acquisition)\b",
    r"\bwalks?\s+away\s+from\s+deal\b",
    r"\bslash(?:es|ed|ing)\s+guidance\b",
    r"\bcuts?\s+(?:guidance|outlook|forecast)\s+(?:unexpectedly|abruptly)?\b",
    r"\blowers?\s+forecast\s+unexpectedly\b",
    r"\bwithdraws?\s+guidance\b",
    r"\bhik(?:es|ed|ing)\s+dividend\b",
    r"\bsurprise\s+profit\b",
    r"\bbeats?\s+by\s+wide\s+margin\b",
    r"\bunexpected\s+loss\b",
    r"\bmassive\s+(?:beat|miss)\b",
    r"\brestatement\b",
    r"\baccounting\s+(?:restatement|irregularities|probe|fraud)\b",
    r"\brestates?\s+(?:financials|earnings|results)\b",
    r"\bshock(?:ing|s|ed)?\b",
    r"\babrupt(?:ly)?\b",
    r"\bunannounced\b",
    r"\bunforeseen\b",
    r"\bemergency\s+(?:meeting|rate\s+cut|rate\s+hike)\b",
    r"\b(?:ousted|fires?\s+ceo|ceo\s+abruptly\s+resigns?|immediate\s+resignation)\b",
    r"\bantitrust\s+(?:lawsuit|probe|ruling|block)\b",
    r"\bexport\s+ban\b",
]

# Priced-in, stale, or recycled reporting cues
PRICED_IN_PATTERNS = [
    r"\bmonths\s+after\b",
    r"\bweeks\s+after\b",
    r"\bpreviously\s+announced\b",
    r"\bpreviously\s+reported\b",
    r"\bas\s+reported\s+earlier\b",
    r"\bas\s+noted\s+previously\b",
    r"\balready\s+announced\b",
    r"\balready\s+priced\s+in\b",
    r"\brecap(?:s)?\b",
    r"\breminder\b",
    r"\brecycled\b",
]

# In-line consensus validation cues
IN_LINE_PATTERNS = [
    r"\bas\s+expected\b",
    r"\bin\s+line\s+with\s+expectations\b",
    r"\bin[- ]line\b",
    r"\bwidely\s+expected\b",
    r"\bmeets?\s+expectations\b",
    r"\bmeets?\s+consensus\b",
    r"\breiterat(?:es|ed|ing)\b",
    r"\breaffirm(?:s|ed|ing)\b",
    r"\bmaintains?\s+(?:guidance|outlook)\b",
]


class NoveltyEngine:
    """Evaluates catalyst expectation gaps, genuine surprise vs priced-in rehash, and event lineage."""

    def __init__(self) -> None:
        self._revision_regexes = [re.compile(p, re.IGNORECASE) for p in REVISION_PATTERNS]
        self._surprise_regexes = [re.compile(p, re.IGNORECASE) for p in SURPRISE_PATTERNS]
        self._priced_in_regexes = [re.compile(p, re.IGNORECASE) for p in PRICED_IN_PATTERNS]
        self._in_line_regexes = [re.compile(p, re.IGNORECASE) for p in IN_LINE_PATTERNS]

    def detect_revision_catalyst(self, text: str) -> Optional[str]:
        """Detect if text contains a revision or reversal catalyst."""
        if not text:
            return None
        for rx in self._revision_regexes:
            m = rx.search(text)
            if m:
                return m.group(0)
        return None

    def detect_surprise(self, text: str) -> Optional[str]:
        """Detect genuine market shock or unexpected consensus deviation cues."""
        if not text:
            return None
        for rx in self._surprise_regexes:
            m = rx.search(text)
            if m:
                return m.group(0)
        return None

    def detect_priced_in(self, text: str) -> Optional[str]:
        """Detect stale, recycled, or previously announced cues."""
        if not text:
            return None
        for rx in self._priced_in_regexes:
            m = rx.search(text)
            if m:
                return m.group(0)
        return None

    def detect_in_line(self, text: str) -> Optional[str]:
        """Detect in-line or consensus-conforming validation cues."""
        if not text:
            return None
        for rx in self._in_line_regexes:
            m = rx.search(text)
            if m:
                return m.group(0)
        return None

    def assess_novelty(
        self,
        article: RawArticle,
        existing_lineage: Optional[EventLineageRecord] = None,
    ) -> Tuple[NoveltyState, float, str]:
        """Evaluate catalyst novelty state, score (1.0 to 10.0), and analyst rationale.

        Args:
            article: Ingested news item to evaluate.
            existing_lineage: Prior tracked event record if previously alerted or recorded.

        Returns:
            Tuple of (NoveltyState, novelty_score, rationale)
        """
        full_text = f"{article.title} {article.summary or ''} {article.content or ''}".strip()

        # 1. Existing lineage branch
        if existing_lineage is not None:
            revision_trigger = self.detect_revision_catalyst(full_text)
            if revision_trigger:
                return (
                    NoveltyState.GENUINE_SURPRISE,
                    8.5,
                    f"Material revision or status change for tracked event: '{revision_trigger}'",
                )
            return (
                NoveltyState.PRICED_IN_OR_RECYCLED,
                3.0,
                "Follow-up reporting on previously alerted event",
            )

        # 2. Standalone evaluation: Check unexpected shocks / genuine surprise
        surprise_cue = self.detect_surprise(full_text)
        if surprise_cue:
            return (
                NoveltyState.GENUINE_SURPRISE,
                9.0,
                f"Genuine market surprise catalyst detected: '{surprise_cue}'",
            )

        # 3. Check priced-in / recycled cues
        priced_in_cue = self.detect_priced_in(full_text)
        if priced_in_cue:
            return (
                NoveltyState.PRICED_IN_OR_RECYCLED,
                3.0,
                f"Priced-in or recycled catalyst: '{priced_in_cue}'",
            )

        # 4. Check in-line / consensus confirmation cues
        in_line_cue = self.detect_in_line(full_text)
        if in_line_cue:
            return (
                NoveltyState.IN_LINE_CONFIRMATION,
                5.5,
                f"In-line confirmation meeting consensus expectations: '{in_line_cue}'",
            )

        # 5. Default fresh event without explicit surprise or expectation deviation cues
        return (
            NoveltyState.IN_LINE_CONFIRMATION,
            6.0,
            "New standalone event without explicit expectation deviation or surprise cues",
        )
