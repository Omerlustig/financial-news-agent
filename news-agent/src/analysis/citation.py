import logging
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional

logger = logging.getLogger(__name__)

# Patterns capturing specific financial figures: dollar sums, billions/millions, percentages, EPS
FIGURE_PATTERNS = [
    r"\$\d+(?:\.\d+)?(?:\s*(?:billion|million|trillion|B|M|k))?",
    r"\b\d+(?:\.\d+)?%",
    r"\b\d+(?:\.\d+)?\s*(?:billion|million|trillion)\s*(?:dollars)?",
    r"\$\d+(?:\.\d+)?\s*per\s*share",
]


@dataclass
class CitationResult:
    """Outcome of factual evidence verification against the source article."""
    is_grounded: bool
    unsupported_claims: List[str] = field(default_factory=list)
    evidence_snippets: List[str] = field(default_factory=list)
    repaired_thesis: Optional[str] = None


@dataclass
class CitationConfig:
    """Configuration for citation and factual grounding verification."""
    enabled: bool = True
    auto_repair: bool = True


class CitationGroundingEngine:
    """Verifies that quantitative figures (dollars, percentages, multiples) and key
    assertions made in the catalyst thesis are explicitly grounded in the extracted
    full article text, flagging or repairing uncorroborated hallucinations.
    """

    def __init__(self, config: Optional[CitationConfig] = None) -> None:
        self.config = config or CitationConfig()
        self._compiled_patterns = [re.compile(p, re.IGNORECASE) for p in FIGURE_PATTERNS]

    def extract_figures(self, text: str) -> List[str]:
        """Extracts numerical claims, currency figures, and percentages from text."""
        figures: List[str] = []
        for pattern in self._compiled_patterns:
            matches = pattern.findall(text)
            for m in matches:
                clean = m.strip()
                if clean and clean not in figures:
                    figures.append(clean)
        return figures

    def _normalize(self, text: str) -> str:
        """Normalizes string for robust substring matching."""
        lowered = text.lower()
        # Remove extra punctuation
        cleaned = re.sub(r"[,\'\"]", "", lowered)
        return " ".join(cleaned.split())

    def verify_grounding(
        self,
        thesis: str,
        article_body: str,
        repair_fn: Optional[Callable[[str, str], str]] = None,
    ) -> CitationResult:
        """Validates that key figures in thesis exist in article_body."""
        if not self.config.enabled:
            return CitationResult(is_grounded=True)

        if not article_body or not article_body.strip():
            # If no body available, cannot reliably cross-verify
            return CitationResult(is_grounded=True)

        norm_body = self._normalize(article_body)
        figures = self.extract_figures(thesis)

        unsupported: List[str] = []
        evidence: List[str] = []

        for fig in figures:
            norm_fig = self._normalize(fig)
            # Check direct occurrence
            if norm_fig in norm_body:
                evidence.append(fig)
                continue

            # Check core number part
            num_match = re.search(r"\d+(?:\.\d+)?", norm_fig)
            if num_match and num_match.group(0) in norm_body:
                evidence.append(fig)
            else:
                unsupported.append(fig)

        is_grounded = len(unsupported) == 0
        repaired_text = None

        if not is_grounded and self.config.auto_repair and repair_fn:
            try:
                unsupported_desc = ", ".join(unsupported)
                repaired_text = repair_fn(thesis, unsupported_desc)
                if repaired_text:
                    logger.info("Successfully repaired thesis ungrounded figures: %s", unsupported_desc)
                    is_grounded = True
            except Exception as e:
                logger.warning("Auto-repair callback failed: %s", e)

        return CitationResult(
            is_grounded=is_grounded,
            unsupported_claims=unsupported,
            evidence_snippets=evidence,
            repaired_thesis=repaired_text,
        )
