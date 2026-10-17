from unittest.mock import patch, MagicMock
import pytest
from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade
from src.analysis.prompts import build_analyst_prompt
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


def test_filter_valid_tickers():
    from src.analysis.prompts import filter_valid_tickers
    raw = ["nvda", "$aapl", "$US", "$CHINA", "$BRK.B", "123", "$1234", "$", "$SMH", "NVDA"]
    filtered = filter_valid_tickers(raw)
    assert filtered == ["$NVDA", "$AAPL", "$BRK.B", "$SMH"]
