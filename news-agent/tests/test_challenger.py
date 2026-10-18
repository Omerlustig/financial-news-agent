import pytest
from src.analysis.challenger import AdversarialChallenger
from src.models import AdversarialReview


@pytest.fixture
def challenger():
    return AdversarialChallenger()


def test_challenge_bullish_thesis(challenger):
    result = challenger.challenge_thesis(
        title="NVIDIA Unveils Next-Gen AI Architecture with Record Pre-Orders",
        summary="Demand reaches unprecedented levels with enterprise backlog extending into 2027.",
        stance="BULLISH",
        affected_tickers=["$NVDA", "$TSM"],
    )
    assert isinstance(result, AdversarialReview)
    assert "$NVDA" in result.bearish_counter_thesis or "TSM" in result.bearish_counter_thesis
    assert "margin compression" in result.bearish_counter_thesis.lower() or "delays" in result.bearish_counter_thesis.lower()
    assert "sell the news" in result.priced_in_risk.lower() or "discounted" in result.priced_in_risk.lower()
    assert "beta" in result.macro_confounding_risk.lower() or "macro" in result.macro_confounding_risk.lower()
    assert "eps" in result.falsification_threshold.lower() or "margin" in result.falsification_threshold.lower()
    assert result.is_thesis_robust is True


def test_challenge_bearish_thesis(challenger):
    result = challenger.challenge_thesis(
        title="Intel Delays Next-Gen Fab Buildout Following Enterprise Cancellation",
        summary="Supplier faces severe capacity overhang and revenue loss.",
        stance="BEARISH",
        affected_tickers=["$INTC"],
    )
    assert isinstance(result, AdversarialReview)
    assert "$INTC" in result.bearish_counter_thesis
    assert "overreact" in result.bearish_counter_thesis.lower() or "temporary" in result.bearish_counter_thesis.lower() or "cushion" in result.bearish_counter_thesis.lower()
    assert "discounted" in result.priced_in_risk.lower() or "priced into" in result.priced_in_risk.lower()
    assert "macro" in result.macro_confounding_risk.lower() or "risk-off" in result.macro_confounding_risk.lower()
    assert "eps" in result.falsification_threshold.lower() or "approval" in result.falsification_threshold.lower()
    assert result.is_thesis_robust is True


def test_challenge_volatile_stance(challenger):
    result = challenger.challenge_thesis(
        title="Biotech Firm Awaits FDA Advisory Committee Vote on Lead Drug",
        summary="High binary risk ahead of panel deliberation.",
        stance="VOLATILE",
        affected_tickers=["$BIIB"],
    )
    assert isinstance(result, AdversarialReview)
    assert "whipsaw" in result.bearish_counter_thesis.lower() or "swings" in result.bearish_counter_thesis.lower()
    assert "volatility" in result.priced_in_risk.lower()
    assert "volatility" in result.falsification_threshold.lower()
    assert result.is_thesis_robust is True


def test_challenge_neutral_stance(challenger):
    result = challenger.challenge_thesis(
        title="Company Announces Routine Annual Shareholder Meeting Date",
        summary="Annual meeting scheduled for next quarter.",
        stance="NEUTRAL",
        affected_tickers=["$AAPL"],
    )
    assert isinstance(result, AdversarialReview)
    assert "drag" in result.bearish_counter_thesis.lower() or "inertia" in result.bearish_counter_thesis.lower()
    assert "routine" in result.priced_in_risk.lower() or "consensus" in result.priced_in_risk.lower()
    assert result.is_thesis_robust is True


def test_challenge_unverified_rumor_fails_robustness(challenger):
    result = challenger.challenge_thesis(
        title="Rumor: Unconfirmed Speculation of Mega Buyout Circulates on Social Media",
        summary="Anonymous sources say deal might be under discussion.",
        stance="BULLISH",
        affected_tickers=["$XYZ"],
    )
    assert isinstance(result, AdversarialReview)
    assert result.is_thesis_robust is False


def test_challenge_llm_payload_parsing(challenger):
    payload = {
        "bearish_counter_thesis": "Custom contrarian thesis from LLM reasoning.",
        "priced_in_risk": "Custom priced in risk evaluation.",
        "macro_confounding_risk": "Custom macro beta risk.",
        "falsification_threshold": "Next quarter EPS below $1.00 disproves thesis.",
        "is_thesis_robust": True,
    }
    result = challenger.challenge_thesis(
        title="Headline",
        llm_payload=payload,
    )
    assert result.bearish_counter_thesis == "Custom contrarian thesis from LLM reasoning."
    assert result.priced_in_risk == "Custom priced in risk evaluation."
    assert result.macro_confounding_risk == "Custom macro beta risk."
    assert result.falsification_threshold == "Next quarter EPS below $1.00 disproves thesis."
    assert result.is_thesis_robust is True


def test_challenge_llm_payload_nested(challenger):
    payload = {
        "adversarial_review": {
            "bearish_counter_thesis": "Nested counter thesis.",
            "priced_in_risk": "Nested priced in.",
            "macro_confounding_risk": "Nested macro.",
            "falsification_threshold": "Nested falsification.",
            "is_thesis_robust": False,
        }
    }
    result = challenger.challenge_thesis(
        title="Headline",
        llm_payload=payload,
    )
    assert result.bearish_counter_thesis == "Nested counter thesis."
    assert result.is_thesis_robust is False
