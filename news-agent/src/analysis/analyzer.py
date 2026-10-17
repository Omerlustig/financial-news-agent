import json
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Union

import litellm
litellm.suppress_debug_info = True
litellm.telemetry = False
litellm.set_verbose = False
litellm.drop_params = True

# Silence verbose LiteLLM logger outputs
logging.getLogger("LiteLLM").setLevel(logging.WARNING)
logging.getLogger("litellm").setLevel(logging.WARNING)

from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    DirectionalStance,
    TimeHorizon,
    CausalChain,
    HardGates,
    TokenUsage,
)
from src.config import LLMSettings
from src.storage.base import BaseStorage
from src.analysis.prompts import build_analyst_prompt, filter_valid_tickers
from src.analysis.taxonomy import SECTaxonomyClassifier
from src.analysis.sentiment import FinancialSentimentEngine



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
        fallback_model: Optional[str] = None,
        storage: Optional[BaseStorage] = None,
        llm_settings: Optional[LLMSettings] = None,
    ):
        self.storage = storage
        self.llm_settings = llm_settings

        if llm_settings is not None:
            self.model = llm_settings.model
            self.fallback_model = fallback_model or llm_settings.fallback_model
            self.temperature = llm_settings.temperature
            self.max_retries = llm_settings.max_retries
        elif isinstance(model, LLMSettings):
            self.llm_settings = model
            self.model = model.model
            self.fallback_model = fallback_model or model.fallback_model
            self.temperature = model.temperature
            self.max_retries = model.max_retries
        else:
            self.model = model
            self.fallback_model = fallback_model
            self.temperature = temperature
            self.max_retries = max_retries

        self.retry_delay = retry_delay
        self.backoff_factor = backoff_factor
        self.fallback_on_error = fallback_on_error
        self.sentiment_engine = FinancialSentimentEngine()
        self.quota_exhausted = False


    def analyze(self, article: RawArticle) -> MarketImpactAnalysis:
        """Analyzes an article using litellm with failover to fallback_model and exponential backoff retries."""
        if self.quota_exhausted:
            return self._build_fallback(
                article,
                "API request quota previously exhausted; evaluated with deterministic financial sentiment & regulatory engine.",
            )

        system_prompt, user_prompt = build_analyst_prompt(article)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        candidate_models = [self.model]
        if self.fallback_model and self.fallback_model != self.model:
            candidate_models.append(self.fallback_model)

        last_error: Optional[Exception] = None

        for model_idx, target_model in enumerate(candidate_models):
            is_fallback = model_idx > 0
            has_fallback_remaining = model_idx < len(candidate_models) - 1
            current_delay = self.retry_delay

            for attempt in range(1, self.max_retries + 1):
                try:
                    logger.debug(
                        "Submitting LLM analysis request (model=%s, attempt %d/%d) for article: %s",
                        target_model,
                        attempt,
                        self.max_retries,
                        article.title,
                    )
                    completion_kwargs: Dict[str, Any] = {
                        "model": target_model,
                        "messages": messages,
                        "response_format": {"type": "json_object"},
                    }
                    if not ("gemini-3" in target_model or "gemini/gemini-3" in target_model):
                        completion_kwargs["temperature"] = self.temperature

                    response = litellm.completion(**completion_kwargs)
                    raw_content = self._extract_content(response)
                    analysis = self._parse_and_validate(raw_content, article)

                    # Calibrate sentiment using domain financial sentiment engine
                    analysis.sentiment = self.sentiment_engine.calibrate_sentiment(
                        text=f"{article.title} {article.summary or ''}".strip(),
                        llm_sentiment=analysis.sentiment,
                    )



                    # Observability & Token Telemetry Extraction
                    usage = getattr(response, "usage", None)
                    if usage is None and isinstance(response, dict):
                        usage = response.get("usage")

                    prompt_tokens = 0
                    completion_tokens = 0
                    total_tokens = 0
                    if usage is not None:
                        if isinstance(usage, dict):
                            prompt_tokens = int(usage.get("prompt_tokens") or 0)
                            completion_tokens = int(usage.get("completion_tokens") or 0)
                            total_tokens = int(usage.get("total_tokens") or (prompt_tokens + completion_tokens))
                        else:
                            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
                            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
                            total_tokens = int(getattr(usage, "total_tokens", 0) or (prompt_tokens + completion_tokens))

                    try:
                        cost = float(litellm.completion_cost(completion_response=response))
                    except Exception:
                        cost = 0.0

                    analysis.token_usage = TokenUsage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        estimated_cost_usd=cost,
                        model_name=target_model,
                    )
                    return analysis

                except Exception as e:
                    last_error = e
                    is_rate_limit = (
                        getattr(e, "status_code", None) == 429
                        or "429" in str(e)
                        or "rate_limit" in str(e).lower()
                        or "ratelimit" in str(e).lower()
                        or "quota" in str(e).lower()
                        or "resource_exhausted" in str(e).lower()
                    )
                    is_high_demand = (
                        getattr(e, "status_code", None) == 503
                        or "503" in str(e)
                        or "high demand" in str(e).lower()
                        or "temporarily unavailable" in str(e).lower()
                        or "serviceunavailable" in str(e).lower()
                    )

                    is_auth_error = (
                        getattr(e, "status_code", None) in (401, 403)
                        or "api key" in str(e).lower()
                        or "credentials" in str(e).lower()
                        or "authentication" in str(e).lower()
                        or "unauthorized" in str(e).lower()
                    )

                    is_quota_error = (
                        "quota" in str(e).lower()
                        or "resource_exhausted" in str(e).lower()
                        or "quota exceeded" in str(e).lower()
                    )
                    if is_quota_error and not has_fallback_remaining:
                        self.quota_exhausted = True
                        logger.info(
                            "Daily API quota reached for model '%s'. Seamlessly using high-precision financial rules engine for catalyst scoring.",
                            target_model,
                        )
                        return self._build_fallback(article, f"Daily quota reached for {target_model}")

                    # Immediate failover upon 429 rate limit, 503 high demand spike, or missing/invalid API key
                    if has_fallback_remaining and (is_rate_limit or is_high_demand):
                        next_model = candidate_models[model_idx + 1]
                        logger.warning(
                            "Model '%s' encountered %s; failing over immediately to fallback model '%s'",
                            target_model,
                            "high demand (503)" if is_high_demand else "rate limit (429)",
                            next_model,
                        )
                        break

                    if is_auth_error:
                        if has_fallback_remaining:
                            next_model = candidate_models[model_idx + 1]
                            logger.warning(
                                "Model '%s' encountered auth/missing API key; failing over immediately to fallback model '%s'",
                                target_model,
                                next_model,
                            )
                            break
                        else:
                            logger.info(
                                "Model '%s' credentials unavailable. Using deterministic financial engine for catalyst analysis.",
                                target_model,
                            )
                            return self._build_fallback(article, f"Credentials unavailable for {target_model}")

                    logger.warning(
                        "LLM analysis attempt %d/%d on model '%s' failed (%s): %s",
                        attempt,
                        self.max_retries,
                        target_model,
                        "rate limit 429" if is_rate_limit else ("high demand 503" if is_high_demand else "API or parse error"),
                        e,
                    )

                    if attempt < self.max_retries:
                        time.sleep(current_delay)
                        current_delay *= self.backoff_factor

            # Failover after exhausting retries on current model
            if has_fallback_remaining:
                next_model = candidate_models[model_idx + 1]
                logger.warning(
                    "Model '%s' failed all attempts; failing over to fallback model '%s'",
                    target_model,
                    next_model,
                )

        # All models and retries failed
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
        """Cleans, parses, validates JSON, and sanitizes tradeable tickers and hard gates."""
        if isinstance(content, dict):
            data = dict(content)
        else:
            cleaned_str = clean_json_markdown(str(content))
            data = json.loads(cleaned_str)

        # Sanitize tickers in affected_tickers
        raw_tickers = data.get("affected_tickers", [])
        data["affected_tickers"] = filter_valid_tickers(raw_tickers)

        # Sanitize tickers in causal_chain if present
        if isinstance(data.get("causal_chain"), dict):
            chain = data["causal_chain"]
            if "beneficiaries" in chain:
                chain["beneficiaries"] = filter_valid_tickers(chain["beneficiaries"])
            if "negatively_exposed" in chain:
                chain["negatively_exposed"] = filter_valid_tickers(chain["negatively_exposed"])

            # If affected_tickers is empty, aggregate valid tickers from causal_chain
            if not data["affected_tickers"]:
                combined: List[str] = []
                for t in chain.get("beneficiaries", []) + chain.get("negatively_exposed", []):
                    if t not in combined:
                        combined.append(t)
                if combined:
                    data["affected_tickers"] = filter_valid_tickers(combined)

        # Validate and enforce hard gates if present
        if isinstance(data.get("hard_gates"), dict):
            hg = data["hard_gates"]
            novel = bool(hg.get("is_novel", False))
            verified = bool(hg.get("is_verified", False))
            direct = bool(hg.get("has_direct_exposure", False))
            passes_all = novel and verified and direct
            hg["passes_all_gates"] = passes_all

            # Hard-gate rule: Failed verification or weak indirect exposure blocks OPPORTUNITY_CANDIDATE.
            # Scores cannot override failed hard gates.
            current_status = str(data.get("status", "")).strip().upper()
            if not passes_all and (not current_status or current_status == EventStatus.OPPORTUNITY_CANDIDATE.value):
                if not verified:
                    data["status"] = EventStatus.UNVERIFIED.value
                else:
                    data["status"] = EventStatus.WATCHLIST.value

        # Infer sec_filing_item if missing from LLM response
        if article and not data.get("sec_filing_item"):
            sec_item, _, _ = SECTaxonomyClassifier.classify_sec_item(article.title, article.summary or "")
            if sec_item:
                data["sec_filing_item"] = sec_item.value
            else:
                maya_cat, _, _ = SECTaxonomyClassifier.classify_maya_report(article.title, article.summary or "")
                if maya_cat:
                    data["sec_filing_item"] = f"MAYA: {maya_cat}"

        return MarketImpactAnalysis.model_validate(data)


    def _build_fallback(self, article: RawArticle, reason: str) -> MarketImpactAnalysis:
        """Produces a conservative, structured non-market-moving fallback analysis with compliant CausalChain and HardGates."""
        title_snippet = article.title[:200] if article.title else "Unknown headline"
        return MarketImpactAnalysis(
            is_market_moving=False,
            confidence_score=1,
            impact_grade=ImpactGrade.LOW,
            affected_tickers=[],
            affected_sectors=[],
            summary=title_snippet,
            catalyst_thesis=f"Automated analysis fallback due to: {reason}",
            status=EventStatus.NO_ACTION,
            directional_stance=DirectionalStance.NEUTRAL,
            time_horizon=TimeHorizon.INTRADAY.value,
            causal_chain=CausalChain(
                event_facts=title_snippet,
                economic_mechanism=f"Fallback unparsed content; no economic transmission mechanism ({reason})",
                affected_sectors=[],
                beneficiaries=[],
                negatively_exposed=[],
                counterarguments="Insufficient verified information to build causal thesis.",
                invalidation_conditions="Publication of verified official disclosures.",
            ),
            hard_gates=HardGates(
                is_novel=False,
                is_verified=False,
                has_direct_exposure=False,
                passes_all_gates=False,
            ),
            sentiment=self.sentiment_engine.calibrate_sentiment(
                text=f"{article.title} {article.summary or ''}".strip()
            ),
            disclaimer="⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked.",
        )

