import pytest
from src.analysis.novelty import NoveltyEngine
from src.models import (
    EventLineageRecord,
    NoveltyState,
    RawArticle,
)


@pytest.fixture
def engine():
    return NoveltyEngine()


def test_genuine_surprise_catalysts(engine):
    # Unexpected guidance slash
    slash_art = RawArticle(
        title="Intel slashes guidance unexpectedly amid data center slowdown",
        url="https://www.bloomberg.com/news/articles/2026-10-18/intel-guidance",
        source_name="Bloomberg",
        summary="Intel Corp abruptly lowered its quarterly revenue forecast by 18%.",
    )
    state, score, rationale = engine.assess_novelty(slash_art)
    assert state == NoveltyState.GENUINE_SURPRISE
    assert score >= 8.5
    assert "surprise" in rationale.lower() or "slashes guidance" in rationale.lower()

    # Deal termination shock
    term_art = RawArticle(
        title="Boeing terminates deal to acquire aerospace supplier Spirit",
        url="https://www.reuters.com/business/aerospace/boeing-terminates-deal-2026-10-18/",
        source_name="Reuters",
    )
    state, score, rationale = engine.assess_novelty(term_art)
    assert state == NoveltyState.GENUINE_SURPRISE
    assert score >= 8.5

    # Accounting restatement
    restate_art = RawArticle(
        title="Super Micro announces accounting restatement for prior fiscal years",
        url="https://www.wsj.com/finance/accounting/smci-restatement",
        source_name="Wall Street Journal",
    )
    state, score, rationale = engine.assess_novelty(restate_art)
    assert state == NoveltyState.GENUINE_SURPRISE
    assert score >= 8.5

    # Dividend hike shock
    dividend_art = RawArticle(
        title="Exxon hikes dividend by unexpected 15% following record upstream cash flow",
        url="https://www.cnbc.com/2026/10/18/exxon-dividend-hike.html",
        source_name="CNBC",
    )
    state, score, rationale = engine.assess_novelty(dividend_art)
    assert state == NoveltyState.GENUINE_SURPRISE
    assert score >= 8.5


def test_priced_in_or_recycled(engine):
    # Recycled / stale reporting
    recycled_art = RawArticle(
        title="Microsoft completes acquisition months after previously announced terms",
        url="https://finance.yahoo.com/news/msft-deal-close.html",
        source_name="Yahoo Finance",
        summary="As reported earlier, the closing comes weeks after regulatory clearance was secured.",
    )
    state, score, rationale = engine.assess_novelty(recycled_art)
    assert state == NoveltyState.PRICED_IN_OR_RECYCLED
    assert score <= 3.5
    assert "priced-in" in rationale.lower() or "recycled" in rationale.lower()


def test_in_line_confirmation(engine):
    # In-line expectations validation
    in_line_art = RawArticle(
        title="Apple reports quarterly revenue in line with expectations and reiterates guidance",
        url="https://www.reuters.com/technology/apple-quarterly-results-2026-10-18/",
        source_name="Reuters",
        summary="Apple reported fiscal Q4 results as expected, reaffirming previous revenue targets.",
    )
    state, score, rationale = engine.assess_novelty(in_line_art)
    assert state == NoveltyState.IN_LINE_CONFIRMATION
    assert 5.0 <= score <= 6.0
    assert "in-line" in rationale.lower() or "consensus" in rationale.lower()


def test_existing_lineage_suppression(engine):
    # Prior tracked event in lineage memory
    prior_lineage = EventLineageRecord(
        event_id="evt_aapl_ai_partnership_20261018",
        title="Apple enters partnership with OpenAI for on-device Siri models",
        primary_ticker="$AAPL",
        all_sources=["Reuters", "Bloomberg"],
    )

    # Routine follow-up article without any revision catalyst
    followup_art = RawArticle(
        title="Apple and OpenAI integration: What users can expect next month",
        url="https://www.cnbc.com/2026/10/18/apple-openai-details.html",
        source_name="CNBC",
        summary="Details on how Apple devices will incorporate the previously announced OpenAI features.",
    )

    state, score, rationale = engine.assess_novelty(followup_art, existing_lineage=prior_lineage)
    assert state == NoveltyState.PRICED_IN_OR_RECYCLED
    assert score == 3.0
    assert rationale == "Follow-up reporting on previously alerted event"


def test_existing_lineage_with_revision_catalysts(engine):
    prior_lineage = EventLineageRecord(
        event_id="evt_merger_xyz_20261018",
        title="Synopsys plans acquisition of Ansys",
        primary_ticker="$SNPS",
        all_sources=["Bloomberg"],
    )

    revision_cases = [
        ("Company revises merger terms after shareholder pushback", "revises"),
        ("CEO denies acquisition talks reported yesterday", "denies"),
        ("Regulator cancels hearing as parties halt merger review", "cancels"),
        ("European Commission approves revised divestiture package", "approves"),
        ("FTC halts proposed acquisition pending court hearing", "halts"),
        ("SEC probe opened into executive trading ahead of announcement", "sec probe"),
    ]

    for title, expected_trigger in revision_cases:
        art = RawArticle(
            title=title,
            url="https://www.reuters.com/business/tech-deal-update",
            source_name="Reuters",
        )
        state, score, rationale = engine.assess_novelty(art, existing_lineage=prior_lineage)
        assert state == NoveltyState.GENUINE_SURPRISE, f"Failed for title: '{title}'"
        assert score >= 8.5
        assert "material revision" in rationale.lower()
