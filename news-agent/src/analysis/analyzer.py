import json
import logging
import re
import time
from typing import Any, Dict, Optional, Union

import litellm

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.config import LLMSettings
from src.analysis.prompts import build_analyst_prompt, filter_valid_tickers

logger = logging.getLogger(__name__)


class AnalysisError(Exception):
    """Raised when market impact analysis fails due to LLM errors or unparseable output."""
    pass


def clean_json_markdown(text: str) -> str:
    """Robustly extracts and strips JSON from markdown code blocks or surrounding text."""
    s = text.strip()
    
    # 1. Check for markdown code fences (```json ... ``` or ``` ... ```)
    code_block_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", s, re.IGNORECASE)
    if code_block_match:
        block_content = code_block_match.group(1).strip()
        # Find outermost { and } within the matched block
        start = block_content.find("{")
        end = block_content.rfind("}")
        if start != -1 and end != -1 and end > start:
            return block_content[start:end + 1].strip()
        return block_content

    # 2. Check for outermost { and } in raw text with potential preamble / postamble
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1].strip()

    return s


class MarketImpactAnalyzer:
    """Institutional LLM analysis engine for evaluating financial news catalysts."""

    def __init__(
        self,
        model: Union[str, LLMSettings] = "gemini/gemini-1.5-flash",
        temperature: float = 0.1,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        backoff_factor: float = 2.0,
        fallback_on_error: bool = False,
    ):
        if isinstance(model, LLMSettings):
            self.model = model.model
            self.temperature = model.temperature
            self.max_retries = model.max_retries
        else:
            self.model = model
            self.temperature = temperature
            self.max_retries = max_retries

        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.fallback_on_error = fallback_on_error

    def analyze(self, article: RawArticle) -> MarketImpactAnalysis:
        """Analyzes an article using litellm with exponential backoff retries on API & parse errors."""
        system_prompt, user_prompt = build_analyst_prompt(article)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        last_error: Optional[Exception] = None
        current_delay = self.retry_delay

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "Submitting LLM analysis request (attempt %d/%d) for article: %s",
                    attempt,
                    self.max_retries,
                    article.title,
                )
                response = litellm.completion(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    response_format={"type": "json_object"},
                )
                raw_content = self._extract_content(response)
                return self._parse_and_validate(raw_content, article)
            except Exception as e:
                last_error = e
                logger.warning(
                    "LLM analysis attempt %d/%d failed (API or parse error): %s",
                    attempt,
                    self.max_retries,
                    e,
                )
                if attempt < self.max_retries:
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor

        # All retries failed
        error_msg = f"LLM analysis failed after {self.max_retries} retries (unparseable JSON or API error): {last_error}"
        logger.error(error_msg)
        if self.fallback_on_error:
            return self._build_fallback(article, error_msg)
        raise AnalysisError(error_msg) from last_error

    def _extract_content(self, response: Any) -> str:
        """Extracts text content from litellm response object or dict."""
        if hasattr(response, "choices"):
            choice = response.choices[0]
            if hasattr(choice, "message"):
                content = choice.message.content
            else:
                content = choice["message"]["content"]
        elif isinstance(response, dict) and "choices" in response:
            content = response["choices"][0]["message"]["content"]
        else:
            raise ValueError(f"Unrecognized litellm response structure: {type(response)}")

        if content is None:
            raise ValueError("LLM returned empty content.")
        return content

    def _parse_and_validate(
        self,
        content: Union[str, Dict[str, Any]],
        article: Optional[RawArticle] = None,
    ) -> MarketImpactAnalysis:
        """Cleans, parses, validates JSON, and sanitizes tradeable tickers."""
        if isinstance(content, dict):
            data = dict(content)
        else:
            cleaned_str = clean_json_markdown(str(content))
            data = json.loads(cleaned_str)

        # Sanitize tickers
        raw_tickers = data.get("affected_tickers", [])
        data["affected_tickers"] = filter_valid_tickers(raw_tickers)

        return MarketImpactAnalysis.model_validate(data)

    def _build_fallback(self, article: RawArticle, reason: str) -> MarketImpactAnalysis:
        """Produces a conservative, structured non-market-moving fallback analysis."""
        return MarketImpactAnalysis(
            is_market_moving=False,
            confidence_score=1,
            impact_grade=ImpactGrade.LOW,
            affected_tickers=[],
            affected_sectors=[],
            summary=article.title[:200],
            catalyst_thesis=f"Automated analysis fallback due to: {reason}",
        )
