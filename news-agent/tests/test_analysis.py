from unittest.mock import patch, MagicMock
import pytest
from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    DirectionalStance,
    TimeHorizon,
    CausalChain,
    HardGates,
)
from src.analysis.prompts import build_analyst_prompt, SYSTEM_ANALYST_PROMPT
from src.analysis.analyzer import MarketImpactAnalyzer, AnalysisError

MOCK_LLM_JSON = """{
  "is_market_moving": true,
  "confidence_score": 9,
  "impact_grade": "CRITICAL",
  "affected_tickers": ["$NVDA", "$AMD", "$TSM"],
  "affected_sectors": ["Semiconductors"],
  "summary": "U.S. widens restrictions on advanced AI chip exports.",
  "catalyst_thesis": "Immediate high-margin revenue impairment in foreign markets."
}"""

MOCK_MARKDOWN_JSON = f"""```json
{MOCK_LLM_JSON}
```"""


def test_build_analyst_prompt_content():
    article = RawArticle(
        title="Federal Reserve Cuts Benchmark Rate by 50 bps in Surprise Move",
        url="https://wsj.com/fed-cut-surprise",
        source_name="Wall Street Journal",
        published_at=None,
        summary="Fed initiates emergency rate easing cycle to avert liquidity freeze.",
        content="The Federal Open Market Committee voted unexpectedly to lower rates."
    )
    system_prompt, user_prompt = build_analyst_prompt(article)

    # System prompt assertions: Wall Street rubric & institutional intelligence
    assert "Wall Street" in system_prompt or "Financial Analyst" in system_prompt
    assert "catalyst" in system_prompt.lower()
    assert "noise" in system_prompt.lower()
    assert "1st-order" in system_prompt.lower() or "first-order" in system_prompt.lower()
    assert "2nd-order" in system_prompt.lower() or "second-order" in system_prompt.lower()
    assert "CRITICAL" in system_prompt and "HIGH" in system_prompt and "LOW" in system_prompt
    assert "$" in system_prompt  # Ticker prefix requirement

    # User prompt assertions: article components
    assert "Federal Reserve Cuts Benchmark Rate" in user_prompt
    assert "Wall Street Journal" in user_prompt
    assert "Fed initiates emergency rate easing" in user_prompt
    assert "The Federal Open Market Committee voted" in user_prompt


def test_analyzer_parses_llm_response():
    analyzer = MarketImpactAnalyzer(model="gemini/gemini-1.5-flash")
    article = RawArticle(
        title="U.S. expands export bans on AI chips",
        url="https://reuters.com/1",
        source_name="Reuters",
        published_at=None,
        summary="New restrictions announced by Commerce Dept."
    )
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": MOCK_LLM_JSON}}]
        }
        result = analyzer.analyze(article)
        assert isinstance(result, MarketImpactAnalysis)
        assert result.is_market_moving is True
        assert result.confidence_score == 9
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert "$NVDA" in result.affected_tickers
        assert "$TSM" in result.affected_tickers
        assert "Semiconductors" in result.affected_sectors


def test_analyzer_parses_markdown_wrapped_json():
    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="U.S. expands export bans on AI chips",
        url="https://reuters.com/1",
        source_name="Reuters",
        published_at=None,
        summary="New restrictions announced by Commerce Dept."
    )
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": MOCK_MARKDOWN_JSON}}]
        }
        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert "$NVDA" in result.affected_tickers


def test_analyzer_filters_invalid_tickers():
    raw_with_invalid_tickers = """{
      "is_market_moving": true,
      "confidence_score": 8,
      "impact_grade": "HIGH",
      "affected_tickers": ["$NVDA", "$US", "$USA", "$CHINA", "$AAPL", "$EU"],
      "affected_sectors": ["Technology"],
      "summary": "Trade policy update affects major tech firms.",
      "catalyst_thesis": "Tariffs impact margin structures of US multinationals."
    }"""
    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Trade Policy Shift",
        url="https://bloomberg.com/trade",
        source_name="Bloomberg",
        published_at=None,
        summary="Tariffs announced."
    )
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": raw_with_invalid_tickers}}]
        }
        result = analyzer.analyze(article)
        # Verify country tokens are filtered out and real tickers are retained
        assert "$NVDA" in result.affected_tickers
        assert "$AAPL" in result.affected_tickers
        assert "$US" not in result.affected_tickers
        assert "$USA" not in result.affected_tickers
        assert "$CHINA" not in result.affected_tickers
        assert "$EU" not in result.affected_tickers


def test_analyzer_retries_with_exponential_backoff_on_rate_limit():
    analyzer = MarketImpactAnalyzer(max_retries=3)
    article = RawArticle(
        title="Test Article",
        url="https://example.com/test",
        source_name="Test Source",
        published_at=None,
        summary="Test summary"
    )

    with patch("litellm.completion") as mock_completion, \
         patch("time.sleep") as mock_sleep:
        # Fail twice with exception, succeed on 3rd call
        mock_completion.side_effect = [
            Exception("Rate limit exceeded 429"),
            Exception("Rate limit exceeded 429"),
            {"choices": [{"message": {"content": MOCK_LLM_JSON}}]}
        ]

        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert mock_completion.call_count == 3
        assert mock_sleep.call_count == 2
        # Verify backoff delay increases exponentially
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert delays[1] > delays[0]


def test_analyzer_exceeds_max_retries_raises_analysis_error():
    analyzer = MarketImpactAnalyzer(max_retries=2)
    article = RawArticle(
        title="Test Article",
        url="https://example.com/test",
        source_name="Test Source",
        published_at=None,
        summary="Test summary"
    )

    with patch("litellm.completion", side_effect=Exception("API Connection Refused")), \
         patch("time.sleep"):
        with pytest.raises(AnalysisError) as exc_info:
            analyzer.analyze(article)
        assert "retries" in str(exc_info.value).lower() or "failed" in str(exc_info.value).lower()


def test_analyzer_unparseable_output_raises_analysis_error():
    analyzer = MarketImpactAnalyzer(max_retries=1)
    article = RawArticle(
        title="Test Article",
        url="https://example.com/test",
        source_name="Test Source",
        published_at=None,
        summary="Test summary"
    )

    with patch("litellm.completion") as mock_completion, \
         patch("time.sleep"):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "I am an AI and cannot evaluate this right now."}}]
        }
        with pytest.raises(AnalysisError) as exc_info:
            analyzer.analyze(article)
        assert "json" in str(exc_info.value).lower() or "unparseable" in str(exc_info.value).lower() or "validation" in str(exc_info.value).lower()


def test_analyzer_unparseable_output_with_structured_fallback():
    analyzer = MarketImpactAnalyzer(max_retries=1, fallback_on_error=True)
    article = RawArticle(
        title="Test Article",
        url="https://example.com/test",
        source_name="Test Source",
        published_at=None,
        summary="Test summary"
    )

    with patch("litellm.completion") as mock_completion, \
         patch("time.sleep"):
        mock_completion.return_value = {
            "choices": [{"message": {"content": "Garbage response that cannot be parsed as JSON"}}]
        }
        result = analyzer.analyze(article)
        assert isinstance(result, MarketImpactAnalysis)
        assert result.is_market_moving is False
        assert result.confidence_score == 1
        assert result.impact_grade == ImpactGrade.LOW
        assert "fallback" in result.catalyst_thesis.lower() or "failed" in result.catalyst_thesis.lower()


def test_analyzer_init_with_llm_settings():
    from src.config import LLMSettings
    settings = LLMSettings(model="openai/gpt-4o", temperature=0.2, max_retries=5)
    analyzer = MarketImpactAnalyzer(model=settings)
    assert analyzer.model == "openai/gpt-4o"
    assert analyzer.temperature == 0.2
    assert analyzer.max_retries == 5


def test_analyzer_handles_object_style_response():
    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Test Object Style",
        url="https://example.com/test-obj",
        source_name="Bloomberg",
        published_at=None,
        summary="Brief summary"
    )

    class MockMessage:
        content = MOCK_LLM_JSON

    class MockChoice:
        message = MockMessage()

    class MockResponse:
        choices = [MockChoice()]

    with patch("litellm.completion", return_value=MockResponse()):
        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert "$NVDA" in result.affected_tickers


def test_analyzer_fallback_on_network_failure():
    analyzer = MarketImpactAnalyzer(max_retries=2, fallback_on_error=True)
    article = RawArticle(
        title="Network Failure Article",
        url="https://example.com/net-fail",
        source_name="Reuters",
        published_at=None,
        summary="Some summary"
    )

    with patch("litellm.completion", side_effect=Exception("Total network outage")), \
         patch("time.sleep"):
        result = analyzer.analyze(article)
        assert isinstance(result, MarketImpactAnalysis)
        assert result.is_market_moving is False
        assert result.confidence_score == 1
        assert result.impact_grade == ImpactGrade.LOW
        assert "fallback" in result.catalyst_thesis.lower()


def test_clean_json_markdown():
    from src.analysis.analyzer import clean_json_markdown
    assert clean_json_markdown('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_json_markdown('```\n{"a": 1}\n```') == '{"a": 1}'
    assert clean_json_markdown('{"a": 1}') == '{"a": 1}'
    assert clean_json_markdown('  \n```json\n{"a": 1}\n```\n  ') == '{"a": 1}'
    assert clean_json_markdown('Here is analysis:\n```json\n{"a": {"b": 2}}\n```\nRegards') == '{"a": {"b": 2}}'
    assert clean_json_markdown('Here is data: {"a": 1} Thanks!') == '{"a": 1}'


def test_filter_valid_tickers():
    from src.analysis.prompts import filter_valid_tickers
    raw = ["nvda", "$aapl", "$US", "$CHINA", "$BRK.B", "123", "$1234", "$", "$SMH", "NVDA"]
    filtered = filter_valid_tickers(raw)
    assert filtered == ["$NVDA", "$AAPL", "$BRK.B", "$SMH"]


def test_analyzer_retries_on_parse_failure_and_succeeds():
    analyzer = MarketImpactAnalyzer(max_retries=3)
    article = RawArticle(
        title="Parse Failure Then Success",
        url="https://example.com/retry-parse",
        source_name="Bloomberg",
        published_at=None,
        summary="Summary test"
    )

    with patch("litellm.completion") as mock_completion, \
         patch("time.sleep") as mock_sleep:
        # 1st attempt: invalid JSON (parse error)
        # 2nd attempt: valid JSON
        mock_completion.side_effect = [
            {"choices": [{"message": {"content": "Not valid JSON output"}}]},
            {"choices": [{"message": {"content": MOCK_LLM_JSON}}]}
        ]

        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert mock_completion.call_count == 2
        assert mock_sleep.call_count == 1


def test_analyzer_parses_markdown_json_with_preamble_and_postamble():
    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Preamble and Postamble Test",
        url="https://example.com/preamble-test",
        source_name="Reuters",
        published_at=None,
        summary="Summary"
    )

    llm_output_with_text = f"""Here is the market impact analysis for the headline:
```json
{MOCK_LLM_JSON}
```
Please let me know if you would like deeper channel checks on the supply chain.
"""

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": llm_output_with_text}}]
        }
        result = analyzer.analyze(article)
        assert result.is_market_moving is True
        assert result.confidence_score == 9
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert "$NVDA" in result.affected_tickers


def test_prompt_construction_institutional_skills_and_debate():
    article = RawArticle(
        title="NVIDIA Unveils Next-Gen Blackwell Ultra Architecture",
        url="https://reuters.com/nvidia-blackwell-ultra",
        source_name="Reuters",
        published_at=None,
        summary="NVIDIA reveals next-generation Blackwell architecture with 4x AI training throughput.",
        content="CEO Jensen Huang announced the Blackwell Ultra platform during the GTC keynote."
    )
    system_prompt, user_prompt = build_analyst_prompt(article)

    # 1. 5 Deterministic skills in prompt
    assert "extract_financial_event" in system_prompt
    assert "verify_event" in system_prompt
    assert "analyze_financial_impact" in system_prompt
    assert "evaluate_alert_gates" in system_prompt
    assert "syndicated" in system_prompt.lower()

    # 2. Hard-gate rule blocking OPPORTUNITY_CANDIDATE
    assert "passes_all_gates" in system_prompt
    assert "failed verification or weak indirect exposure blocks" in system_prompt.lower() or "blocks `opportunity_candidate`" in system_prompt.lower()

    # 3. Balanced analysis: debate logic, counterarguments, invalidation conditions
    assert "beneficiaries" in system_prompt
    assert "negatively_exposed" in system_prompt
    assert "counterarguments" in system_prompt
    assert "invalidation_conditions" in system_prompt

    # 4. Multi-language and US + Israel support
    assert "Israel" in system_prompt
    assert "Hebrew" in system_prompt or "Hebrew" in user_prompt
    assert "Globes" in system_prompt or "Globes" in user_prompt
    assert "Calcalist" in system_prompt or "Calcalist" in user_prompt
    assert "MAYA" in system_prompt or "MAYA" in user_prompt

    # 5. Schema alignment
    assert "causal_chain" in system_prompt
    assert "hard_gates" in system_prompt
    assert "directional_stance" in system_prompt
    assert "time_horizon" in system_prompt

    # User prompt verification
    assert "extract_financial_event" in user_prompt
    assert "verify_event" in user_prompt
    assert "analyze_financial_impact" in user_prompt
    assert "evaluate_alert_gates" in user_prompt
    assert "NVIDIA Unveils Next-Gen Blackwell Ultra" in user_prompt


def test_analyzer_parses_full_causal_response_with_debate_and_hard_gates():
    full_causal_json = """{
      "is_market_moving": true,
      "confidence_score": 9,
      "impact_grade": "CRITICAL",
      "affected_tickers": ["$NVDA", "$TSM"],
      "affected_sectors": ["Semiconductors", "Cloud Infrastructure"],
      "summary": "U.S. Commerce Dept introduces tighter restrictions on advanced compute accelerators.",
      "catalyst_thesis": "Substantial enterprise pricing power compression for legacy products while accelerating sovereign AI foundry demand.",
      "status": "OPPORTUNITY_CANDIDATE",
      "directional_stance": "BULLISH",
      "time_horizon": "SHORT_TERM",
      "causal_chain": {
        "event_facts": "U.S. Commerce Department issues updated export controls on high-bandwidth AI hardware.",
        "economic_mechanism": "Forces hyperscalers to expedite custom ASIC deployments and boosts advanced packaging bookings at TSMC.",
        "affected_sectors": ["Semiconductors", "Technology"],
        "beneficiaries": ["$NVDA", "$TSM"],
        "negatively_exposed": ["$INTC"],
        "counterarguments": "Near-term loss of APAC datacenter revenue could weigh on Q3 gross margin guidance.",
        "invalidation_conditions": "Commerce Department grants sweeping export license waivers to foreign cloud providers."
      },
      "hard_gates": {
        "is_novel": true,
        "is_verified": true,
        "has_direct_exposure": true,
        "passes_all_gates": true
      },
      "disclaimer": "⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked."
    }"""

    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="U.S. Tightens Advanced AI Chip Controls",
        url="https://wsj.com/us-ai-chip-controls",
        source_name="Wall Street Journal",
        published_at=None,
        summary="Commerce department expands semiconductor rules."
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": full_causal_json}}]
        }
        result = analyzer.analyze(article)

        assert isinstance(result, MarketImpactAnalysis)
        assert result.is_market_moving is True
        assert result.confidence_score == 9
        assert result.impact_grade == ImpactGrade.CRITICAL
        assert result.status == EventStatus.OPPORTUNITY_CANDIDATE
        assert result.directional_stance == DirectionalStance.BULLISH
        assert result.time_horizon == "SHORT_TERM"

        # CausalChain assertions
        assert result.causal_chain is not None
        assert "$NVDA" in result.causal_chain.beneficiaries
        assert "$TSM" in result.causal_chain.beneficiaries
        assert "$INTC" in result.causal_chain.negatively_exposed
        assert "APAC datacenter revenue" in result.causal_chain.counterarguments
        assert "export license waivers" in result.causal_chain.invalidation_conditions

        # HardGates assertions
        assert result.hard_gates is not None
        assert result.hard_gates.is_novel is True
        assert result.hard_gates.is_verified is True
        assert result.hard_gates.has_direct_exposure is True
        assert result.hard_gates.passes_all_gates is True


def test_analyzer_hebrew_headline_input_parsing():
    hebrew_article = RawArticle(
        title="בנק ישראל מעלה את הריבית ב-0.5% בעקבות התחזקות האינפלציה",
        url="https://www.globes.co.il/news/article.aspx?did=1001478952",
        source_name="גלובס Globes",
        published_at=None,
        summary="הוועדה המוניטרית של בנק ישראל החליטה להעלות את שער הריבית ב-50 נקודות בסיס לרמה של 4.75%."
    )

    hebrew_llm_json = """{
      "is_market_moving": true,
      "confidence_score": 8,
      "impact_grade": "HIGH",
      "affected_tickers": ["$LUMI", "$POLI"],
      "affected_sectors": ["Banking", "Financial Services"],
      "summary": "Bank of Israel unexpectedly raises benchmark interest rate by 50 bps to 4.75% to counter persistent inflation.",
      "catalyst_thesis": "Immediate expansion of net interest margins on variable-rate prime credit books across tier-1 Israeli lenders.",
      "status": "OPPORTUNITY_CANDIDATE",
      "directional_stance": "BULLISH",
      "time_horizon": "SHORT_TERM",
      "causal_chain": {
        "event_facts": "Bank of Israel Monetary Committee raises rate by 50 bps to 4.75%.",
        "economic_mechanism": "Broad lending rate reset increases net interest margin (NIM) on prime mortgage and corporate portfolios.",
        "affected_sectors": ["Banking", "Financial Services"],
        "beneficiaries": ["$LUMI", "$POLI"],
        "negatively_exposed": ["$BLDG"],
        "counterarguments": "Mortgage affordability deterioration could increase loan-loss provisions in construction loans.",
        "invalidation_conditions": "Emergency liquidity injections or immediate rate cuts following sovereign debt turbulence."
      },
      "hard_gates": {
        "is_novel": true,
        "is_verified": true,
        "has_direct_exposure": true,
        "passes_all_gates": true
      },
      "disclaimer": "⚠️ News-derived hypothesis. Market pricing, valuation, and entry points were not checked."
    }"""

    analyzer = MarketImpactAnalyzer()
    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": hebrew_llm_json}}]
        }
        result = analyzer.analyze(hebrew_article)

        assert result.is_market_moving is True
        assert result.status == EventStatus.OPPORTUNITY_CANDIDATE
        assert "$LUMI" in result.affected_tickers
        assert "$POLI" in result.affected_tickers
        assert result.causal_chain is not None
        assert "$LUMI" in result.causal_chain.beneficiaries
        assert "$BLDG" in result.causal_chain.negatively_exposed
        assert result.hard_gates is not None
        assert result.hard_gates.passes_all_gates is True


def test_unverified_claim_blocks_opportunity_candidate():
    unverified_llm_json = """{
      "is_market_moving": true,
      "confidence_score": 8,
      "impact_grade": "HIGH",
      "affected_tickers": ["$NVDA"],
      "affected_sectors": ["Technology"],
      "summary": "Anonymous blog reports rumored regulatory probe into cloud contracts.",
      "catalyst_thesis": "Potential antitrust scrutiny over enterprise sales practices.",
      "status": "OPPORTUNITY_CANDIDATE",
      "directional_stance": "BEARISH",
      "time_horizon": "MEDIUM_TERM",
      "causal_chain": {
        "event_facts": "Unconfirmed blog rumor alleging regulatory review.",
        "economic_mechanism": "Threat of potential fines or forced sales renegotiations.",
        "affected_sectors": ["Technology"],
        "beneficiaries": [],
        "negatively_exposed": ["$NVDA"],
        "counterarguments": "No official regulatory filings or court documents exist.",
        "invalidation_conditions": "Official regulatory agency denial or lack of formal notice."
      },
      "hard_gates": {
        "is_novel": true,
        "is_verified": false,
        "has_direct_exposure": true,
        "passes_all_gates": true
      }
    }"""

    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Rumored Probe into AI Cloud Contracts",
        url="https://blog.example.com/rumor",
        source_name="Blog Leaks",
        published_at=None,
        summary="Unconfirmed rumors circulating online."
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": unverified_llm_json}}]
        }
        result = analyzer.analyze(article)

        # Hard-gate rule enforcement: unverified claims CANNOT be OPPORTUNITY_CANDIDATE
        assert result.hard_gates is not None
        assert result.hard_gates.is_verified is False
        assert result.hard_gates.passes_all_gates is False
        assert result.status != EventStatus.OPPORTUNITY_CANDIDATE
        assert result.status == EventStatus.UNVERIFIED


def test_failed_direct_exposure_blocks_opportunity_candidate():
    indirect_exposure_json = """{
      "is_market_moving": true,
      "confidence_score": 7,
      "impact_grade": "HIGH",
      "affected_tickers": ["$AAPL"],
      "affected_sectors": ["Consumer Electronics"],
      "summary": "Rare earth mining quotas tightened by regional exporter.",
      "catalyst_thesis": "Generic raw material supply constraints might eventually affect electronic assembly lines.",
      "status": "OPPORTUNITY_CANDIDATE",
      "directional_stance": "BEARISH",
      "time_horizon": "LONG_TERM",
      "causal_chain": {
        "event_facts": "Regional quotas reduced by 10% on raw mineral exports.",
        "economic_mechanism": "Broad upstream commodity cost pressure with indirect downstream exposure.",
        "affected_sectors": ["Materials", "Consumer Electronics"],
        "beneficiaries": [],
        "negatively_exposed": ["$AAPL"],
        "counterarguments": "Company maintains 18-month stockpiles and alternate component suppliers.",
        "invalidation_conditions": "Commodity substitution agreements announced."
      },
      "hard_gates": {
        "is_novel": true,
        "is_verified": true,
        "has_direct_exposure": false,
        "passes_all_gates": true
      }
    }"""

    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Rare Earth Quotas Shift Upstream",
        url="https://metals.example.com/quotas",
        source_name="Mining Wire",
        published_at=None,
        summary="Upstream mining quotas adjusted."
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": indirect_exposure_json}}]
        }
        result = analyzer.analyze(article)

        assert result.hard_gates is not None
        assert result.hard_gates.has_direct_exposure is False
        assert result.hard_gates.passes_all_gates is False
        assert result.status != EventStatus.OPPORTUNITY_CANDIDATE
        assert result.status == EventStatus.WATCHLIST


def test_stale_news_blocks_opportunity_candidate():
    stale_json = """{
      "is_market_moving": true,
      "confidence_score": 8,
      "impact_grade": "HIGH",
      "affected_tickers": ["$NVDA"],
      "affected_sectors": ["Semiconductors"],
      "summary": "Recapped summary of last month's earnings release.",
      "catalyst_thesis": "Past earnings figures reiterated in weekly retrospective.",
      "status": "OPPORTUNITY_CANDIDATE",
      "directional_stance": "NEUTRAL",
      "time_horizon": "LONG_TERM",
      "hard_gates": {
        "is_novel": false,
        "is_verified": true,
        "has_direct_exposure": true,
        "passes_all_gates": false
      }
    }"""

    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Weekly Market Retrospective",
        url="https://example.com/recap",
        source_name="Weekend Review",
        published_at=None,
        summary="A look back at last month's events."
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": stale_json}}]
        }
        result = analyzer.analyze(article)

        assert result.hard_gates is not None
        assert result.hard_gates.is_novel is False
        assert result.hard_gates.passes_all_gates is False
        assert result.status != EventStatus.OPPORTUNITY_CANDIDATE


def test_analyzer_fallback_builds_compliant_causal_chain_and_hard_gates():
    analyzer = MarketImpactAnalyzer(max_retries=1, fallback_on_error=True)
    article = RawArticle(
        title="Garbage Feed Entry",
        url="https://example.com/bad",
        source_name="Bad Feed",
        published_at=None,
        summary="Unparseable content."
    )

    with patch("litellm.completion", side_effect=Exception("LLM Context Length Exceeded")):
        result = analyzer.analyze(article)

        assert isinstance(result, MarketImpactAnalysis)
        assert result.is_market_moving is False
        assert result.status == EventStatus.NO_ACTION
        assert result.confidence_score == 1
        assert result.impact_grade == ImpactGrade.LOW

        # Fallback CausalChain compliance
        assert result.causal_chain is not None
        assert isinstance(result.causal_chain, CausalChain)
        assert "fallback" in result.causal_chain.economic_mechanism.lower()
        assert result.causal_chain.affected_sectors == []
        assert result.causal_chain.beneficiaries == []
        assert result.causal_chain.negatively_exposed == []

        # Fallback HardGates compliance
        assert result.hard_gates is not None
        assert isinstance(result.hard_gates, HardGates)
        assert result.hard_gates.is_novel is False
        assert result.hard_gates.is_verified is False
        assert result.hard_gates.has_direct_exposure is False
        assert result.hard_gates.passes_all_gates is False


def test_analyzer_causal_chain_ticker_sanitization_and_synthesis():
    json_with_chain_tickers = """{
      "is_market_moving": true,
      "confidence_score": 8,
      "impact_grade": "HIGH",
      "affected_tickers": [],
      "affected_sectors": ["Semiconductors"],
      "summary": "Major semiconductor trade policy update.",
      "catalyst_thesis": "Foundry capex expansion benefits advanced equipment makers.",
      "status": "WATCHLIST",
      "directional_stance": "BULLISH",
      "time_horizon": "MEDIUM_TERM",
      "causal_chain": {
        "event_facts": "Government announces chip equipment incentive subsidies.",
        "economic_mechanism": "Direct subsidies lower capital equipment acquisition cost.",
        "affected_sectors": ["Semiconductors"],
        "beneficiaries": ["$asml", "tsm", "$US", "$CHINA"],
        "negatively_exposed": ["intc", "$FED"],
        "counterarguments": "Subsidy disbursements face legislative review.",
        "invalidation_conditions": "Court injunction on subsidy budget."
      },
      "hard_gates": {
        "is_novel": true,
        "is_verified": true,
        "has_direct_exposure": true,
        "passes_all_gates": true
      }
    }"""

    analyzer = MarketImpactAnalyzer()
    article = RawArticle(
        title="Chip Subsidy Announcement",
        url="https://example.com/chip-subsidies",
        source_name="Semi Wire",
        published_at=None,
        summary="Government unveils incentive package."
    )

    with patch("litellm.completion") as mock_completion:
        mock_completion.return_value = {
            "choices": [{"message": {"content": json_with_chain_tickers}}]
        }
        result = analyzer.analyze(article)

        # Verify invalid country and agency tokens were filtered out from causal_chain
        assert result.causal_chain is not None
        assert result.causal_chain.beneficiaries == ["$ASML", "$TSM"]
        assert result.causal_chain.negatively_exposed == ["$INTC"]

        # Verify affected_tickers was synthesized from causal_chain since it was empty
        assert "$ASML" in result.affected_tickers
        assert "$TSM" in result.affected_tickers
        assert "$INTC" in result.affected_tickers
        assert "$US" not in result.affected_tickers
        assert "$FED" not in result.affected_tickers

