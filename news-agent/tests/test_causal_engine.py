import pytest
from src.models import (
    CatalystStage,
    TimeHorizon,
    MarketExchange,
    CausalTransmissionAnalysis,
    MarketImpactAnalysis,
)
from src.analysis.prompts import filter_valid_tickers

def test_causal_transmission_model_serialization():
    causal = CausalTransmissionAnalysis(
        catalyst_stage=CatalystStage.PREDICTIVE_ANTICIPATORY,
        trigger_event="Foreign airlines suspend Tel Aviv routes",
        transmission_mechanism="Removes 40% seat capacity; grants monopoly pricing power to flag carrier",
        time_horizon=TimeHorizon.TACTICAL_1_3_DAYS,
        market_exchange=MarketExchange.TASE,
        predicted_beneficiaries=["$ELAL"],
        unpriced_edge_thesis="Anticipatory window before traffic report discounts yield surge",
    )
    data = causal.model_dump()
    assert data["catalyst_stage"] == "PREDICTIVE_ANTICIPATORY"
    assert data["market_exchange"] == "TASE"
    assert "$ELAL" in data["predicted_beneficiaries"]

def test_tase_ticker_filtering_support():
    tickers = ["$ELAL", "$ESLT", "$LUMI", "ICL", "$NVDA", "$USA"]
    valid = filter_valid_tickers(tickers)
    assert "$ELAL" in valid
    assert "$ESLT" in valid
    assert "$LUMI" in valid
    assert "$ICL" in valid
    assert "$NVDA" in valid
    assert "$USA" not in valid  # Sovereign entity dropped
