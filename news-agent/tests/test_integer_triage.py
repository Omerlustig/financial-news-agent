import json
from datetime import datetime, timezone
from typing import List
import pytest

from src.models import RawArticle
from src.analysis.integer_triage import IntegerIndexTriageEngine, IntegerTriageConfig


def make_article(title: str, source: str = "Reuters") -> RawArticle:
    return RawArticle(
        title=title,
        url=f"https://example.com/{abs(hash(title))}",
        source_name=source,
        published_at=datetime.now(timezone.utc),
        summary="Sample description for testing.",
    )


def test_format_batch_prompt() -> None:
    engine = IntegerIndexTriageEngine()
    articles = [
        make_article("Nvidia releases Blackwell Ultra B200", source="Reuters"),
        make_article("Local soccer tournament ends in draw", source="General Breaking"),
        make_article("FDA approves Pfizer new oncology treatment", source="CNBC"),
    ]
    prompt = engine.format_batch_prompt(articles)
    assert "[0] [Reuters] Nvidia releases Blackwell Ultra B200" in prompt
    assert "[1] [General Breaking] Local soccer tournament ends in draw" in prompt
    assert "[2] [CNBC] FDA approves Pfizer new oncology treatment" in prompt
    assert "integer indices" in prompt.lower()


def test_triage_batch_with_llm_json_response() -> None:
    def mock_llm(prompt: str) -> str:
        return json.dumps([0, 2])

    engine = IntegerIndexTriageEngine(llm_callable=mock_llm)
    articles = [
        make_article("Nvidia releases Blackwell Ultra B200", source="Reuters"),
        make_article("Celebrity wedding in Venice attracts paparazzi", source="General Breaking"),
        make_article("FDA approves Pfizer new oncology treatment", source="CNBC"),
    ]
    selected = engine.triage_batch(articles)
    assert len(selected) == 2
    assert selected[0].title == "Nvidia releases Blackwell Ultra B200"
    assert selected[1].title == "FDA approves Pfizer new oncology treatment"


def test_triage_batch_discards_out_of_bounds_indices() -> None:
    def mock_llm_oob(prompt: str) -> str:
        # 99 and -1 are invalid out-of-range indices
        return "[1, 99, -1]"

    engine = IntegerIndexTriageEngine(llm_callable=mock_llm_oob)
    articles = [
        make_article("Routine article 0"),
        make_article("Crucial economic report on inflation"),
    ]
    selected = engine.triage_batch(articles)
    assert len(selected) == 1
    assert selected[0].title == "Crucial economic report on inflation"


def test_triage_batch_fallback_on_malformed_llm_response() -> None:
    def mock_broken_llm(prompt: str) -> str:
        return "I am unable to analyze this batch because of system load."

    engine = IntegerIndexTriageEngine(llm_callable=mock_broken_llm)
    articles = [
        make_article("Central bank raises benchmark interest rate by 50 bps", source="General Breaking"),
        make_article("Local cat rescued from tree by firefighters", source="General Breaking"),
    ]
    selected = engine.triage_batch(articles)
    # Fallback to economic keyword relevance
    assert len(selected) == 1
    assert "interest rate" in selected[0].title.lower()
