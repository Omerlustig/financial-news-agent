import json
import logging
import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from src.models import RawArticle
from src.analysis.triage import is_economically_relevant

logger = logging.getLogger(__name__)


@dataclass
class IntegerTriageConfig:
    """Settings for integer-index candidate triage."""
    batch_size: int = 20
    enabled: bool = True
    fallback_to_heuristic: bool = True


class IntegerIndexTriageEngine:
    """Performs hallucination-proof article triage by prompting an LLM with
    enumerated integer indices [0..N-1]. The LLM returns ONLY integer indices
    of market-moving catalyst articles, preventing any title or URL hallucinations.
    """

    def __init__(
        self,
        llm_callable: Optional[Callable[[str], str]] = None,
        config: Optional[IntegerTriageConfig] = None,
    ) -> None:
        self.llm_callable = llm_callable
        self.config = config or IntegerTriageConfig()

    def format_batch_prompt(self, articles: Sequence[RawArticle]) -> str:
        """Formats articles into a zero-based numbered list for the LLM."""
        lines = [
            "You are an institutional financial news filter. Review the following numbered headlines.",
            "Identify ONLY articles containing genuine, actionable corporate or macroeconomic market-moving catalysts (e.g. earnings surprises, M&A, FDA decisions, major contracts, sovereign policy, executive departures).",
            "Discard routine market noise, generic listicles ('stocks to buy'), sponsored marketing, and sports/celebrity news.",
            "",
            "HEADLINES:",
        ]
        for idx, art in enumerate(articles):
            src = art.source_name or "Unknown"
            lines.append(f"[{idx}] [{src}] {art.title}")

        lines.extend([
            "",
            "RESPONSE INSTRUCTION:",
            "Return ONLY a JSON array of the integer indices of the selected market catalyst articles (e.g. [0, 2]).",
            "If no articles meet the threshold, return [].",
            "Do not output any explanation or extra text.",
        ])
        return "\n".join(lines)

    def extract_indices(self, raw_response: str, max_index: int) -> List[int]:
        """Parses the LLM output into validated integer indices."""
        text = raw_response.strip()
        # Try direct JSON parse
        try:
            data = json.loads(text)
            if isinstance(data, list):
                valid = [int(x) for x in data if isinstance(x, (int, float)) and 0 <= int(x) < max_index]
                return sorted(list(set(valid)))
            elif isinstance(data, dict) and "selected" in data and isinstance(data["selected"], list):
                valid = [int(x) for x in data["selected"] if isinstance(x, (int, float)) and 0 <= int(x) < max_index]
                return sorted(list(set(valid)))
        except Exception:
            pass

        # Regex fallback to extract bracketed numbers: [0, 2]
        match = re.search(r"\[([\d\s,]+)\]", text)
        if match:
            numbers_str = match.group(1)
            parsed: List[int] = []
            for item in numbers_str.split(","):
                cleaned = item.strip()
                if cleaned.isdigit():
                    num = int(cleaned)
                    if 0 <= num < max_index:
                        parsed.append(num)
            return sorted(list(set(parsed)))

        raise ValueError(f"Failed to parse valid integer indices from response: '{raw_response[:100]}'")

    def triage_batch(self, articles: List[RawArticle]) -> List[RawArticle]:
        """Filters a batch of raw articles, returning only viable catalyst candidates."""
        if not articles:
            return []
        if not self.config.enabled:
            return list(articles)

        # If no LLM callable provided, use heuristic triage
        if not self.llm_callable:
            return [a for a in articles if is_economically_relevant(a)]

        selected_articles: List[RawArticle] = []

        # Process in chunks up to batch_size
        batch_size = max(1, self.config.batch_size)
        for i in range(0, len(articles), batch_size):
            chunk = articles[i : i + batch_size]
            prompt = self.format_batch_prompt(chunk)
            try:
                raw_response = self.llm_callable(prompt)
                indices = self.extract_indices(raw_response, max_index=len(chunk))
                for idx in indices:
                    selected_articles.append(chunk[idx])
            except Exception as e:
                logger.warning("LLM triage failed for chunk; falling back to heuristic: %s", e)
                if self.config.fallback_to_heuristic:
                    selected_articles.extend([a for a in chunk if is_economically_relevant(a)])
                else:
                    selected_articles.extend(chunk)

        return selected_articles
