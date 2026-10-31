import pytest
from typing import Optional
from src.analysis.citation import CitationGroundingEngine, CitationResult, CitationConfig


def test_verify_grounding_passes_when_figures_match_body() -> None:
    engine = CitationGroundingEngine()
    thesis = "Nvidia reported quarterly revenue of $26.04 billion, beating expectations by 18%."
    body = (
        "Nvidia Corp on Wednesday reported stellar financial results for its fiscal first quarter, "
        "posting record quarterly revenue of $26.04 billion, an 18% surge above consensus forecasts."
    )
    result = engine.verify_grounding(thesis, body)
    assert result.is_grounded is True
    assert len(result.unsupported_claims) == 0
    assert len(result.evidence_snippets) > 0


def test_verify_grounding_flags_fabricated_numerical_claims() -> None:
    engine = CitationGroundingEngine()
    thesis = "Nvidia signed a secret $950 million contract with the Department of Energy."
    body = (
        "Nvidia discussed government AI partnerships yesterday. The Department of Energy "
        "confirmed ongoing pilot evaluations, but no financial terms or figures were disclosed."
    )
    result = engine.verify_grounding(thesis, body)
    assert result.is_grounded is False
    assert any("$950 million" in claim or "950" in claim for claim in result.unsupported_claims)


def test_verify_grounding_auto_repair_callback() -> None:
    def mock_repair(original_thesis: str, unsupported: str) -> str:
        # Repairs thesis by stripping the ungrounded numerical assertion
        return "Nvidia discussed government AI partnerships with the Department of Energy."

    engine = CitationGroundingEngine(config=CitationConfig(auto_repair=True))
    thesis = "Nvidia signed a secret $950 million contract with the Department of Energy."
    body = (
        "Nvidia discussed government AI partnerships yesterday with the Department of Energy."
    )
    result = engine.verify_grounding(thesis, body, repair_fn=mock_repair)
    assert result.is_grounded is True
    assert result.repaired_thesis is not None
    assert "$950 million" not in result.repaired_thesis
