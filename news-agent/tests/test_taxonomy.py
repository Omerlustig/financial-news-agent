"""Comprehensive test suite for SEC 8-K and Israeli MAYA corporate disclosure taxonomy."""

import pytest

from src.models import (
    DirectionalStance,
    ImpactGrade,
    RawArticle,
    SECFilingItem,
)
from src.analysis.taxonomy import (
    MayaReportCategory,
    SECTaxonomyClassifier,
)
from src.analysis.prompts import build_analyst_prompt, SYSTEM_ANALYST_PROMPT
from src.analysis.analyzer import MarketImpactAnalyzer


class TestSECTaxonomyClassification:
    """Test suite for SEC 8-K items (1.01, 1.02, 2.02, 4.02, 5.02, General)."""

    def test_classify_item_1_01_explicit_item_number(self):
        title = "Acme Corp (ACM) Form 8-K: Item 1.01 Entry into a Material Definitive Agreement"
        summary = "Company signs definitive merger agreement to acquire competitor for $1.5B."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT
        assert grade == ImpactGrade.HIGH
        assert score == 8
        assert SECTaxonomyClassifier.get_sec_directional_bias(item) == DirectionalStance.BULLISH

    def test_classify_item_1_01_keywords_without_item_number(self):
        title = "NVIDIA Enters into Material Definitive Agreement for Global Foundry Expansion"
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title)

        assert item == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT
        assert grade == ImpactGrade.HIGH
        assert score == 8

    def test_classify_item_1_02_termination(self):
        title = "Form 8-K Current Report: Item 1.02 Termination of a Material Definitive Agreement"
        summary = "Major cloud customer terminates multi-year distribution agreement."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_1_02_TERMINATION_AGREEMENT
        assert grade == ImpactGrade.HIGH
        assert score == 8
        assert SECTaxonomyClassifier.get_sec_directional_bias(item) == DirectionalStance.BEARISH

    def test_classify_item_1_02_keywords_without_item_number(self):
        title = "Supplier Announces Termination of Material Agreement with Lead Automaker"
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title)

        assert item == SECFilingItem.ITEM_1_02_TERMINATION_AGREEMENT
        assert grade == ImpactGrade.HIGH
        assert score == 8
        assert SECTaxonomyClassifier.get_sec_directional_bias(item) == DirectionalStance.BEARISH

    def test_classify_item_2_02_results_of_operations(self):
        title = "Form 8-K: Item 2.02 Results of Operations and Financial Condition"
        summary = "Q3 financial results press release furnished under Item 2.02."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_2_02_EARNINGS_OPERATIONS
        assert grade in (ImpactGrade.HIGH, ImpactGrade.AVERAGE)
        assert score == 7

    def test_classify_item_4_02_accounting_restatement(self):
        title = "8-K Item 4.02 Non-Reliance on Previously Issued Financial Statements"
        summary = "Audit committee determined previous 10-K and 10-Q statements should no longer be relied upon."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_4_02_ACCOUNTING_RESTATEMENT
        assert grade == ImpactGrade.CRITICAL
        assert score == 9
        assert SECTaxonomyClassifier.get_sec_directional_bias(item) == DirectionalStance.BEARISH

    def test_classify_item_4_02_keywords_without_item_number(self):
        title = "Company Discloses Accounting Restatement of Prior Fiscal Year Audited Statements"
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title)

        assert item == SECFilingItem.ITEM_4_02_ACCOUNTING_RESTATEMENT
        assert grade == ImpactGrade.CRITICAL
        assert score == 9

    def test_classify_item_5_02_ceo_cfo_departure(self):
        title = "Item 5.02: Departure of Chief Executive Officer; Appointment of Interim CEO"
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title)

        assert item == SECFilingItem.ITEM_5_02_EXECUTIVE_CHANGE
        assert grade == ImpactGrade.HIGH
        assert score == 7

    def test_classify_item_5_02_routine_director_change(self):
        title = "Form 8-K Item 5.02 Departure of Directors or Certain Officers; Election of Directors"
        summary = "Board of directors adds independent member to audit committee."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_5_02_EXECUTIVE_CHANGE
        assert grade == ImpactGrade.AVERAGE
        assert score == 6

    def test_classify_sec_general_corporate_disclosure(self):
        title = "MicroStrategy Files Current Report on Form 8-K with the SEC"
        summary = "Update regarding corporate treasury management."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item == SECFilingItem.ITEM_GENERAL
        assert grade == ImpactGrade.AVERAGE
        assert score == 5

    def test_classify_sec_non_regulatory_news(self):
        title = "Crude Oil Climbs 2% as Inventory Draw Exceeds Expectations"
        summary = "EIA reports a 3.5 million barrel decline in stockpiles."
        item, grade, score = SECTaxonomyClassifier.classify_sec_item(title, summary)

        assert item is None
        assert grade is None
        assert score is None


class TestMayaTaxonomyClassification:
    """Test suite for Israeli MAYA / TASE corporate reports."""

    def test_classify_maya_profit_warning(self):
        title = "דוח מיידי - אזהרת רווח והרעה מהותית בתוצאות לרבעון השלישי"
        summary = "החברה צופה הפסד תפעולי של 50 מיליון ש\"ח לעומת תחזית קודמת."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.PROFIT_WARNING.value
        assert grade in (ImpactGrade.HIGH, ImpactGrade.CRITICAL)
        assert score in (8, 9)
        assert SECTaxonomyClassifier.get_maya_directional_bias(cat) == DirectionalStance.BEARISH

    def test_classify_maya_merger_and_tender_offer(self):
        title = "דוח מיידי: חתימה על הסכם מיזוג מחייב ורכישת מלוא מניות החברה"
        summary = "העסקה בוצעה לפי שווי של 450 מיליון דולר במזומן."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.MERGER_ACQUISITION.value
        assert grade == ImpactGrade.HIGH
        assert score == 8
        assert SECTaxonomyClassifier.get_maya_directional_bias(cat) == DirectionalStance.BULLISH

    def test_classify_maya_tender_offer(self):
        title = "מאיה: דיווח על הצעת רכש מלאה לרכישת מניות חברת הבת"
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title)

        assert cat == MayaReportCategory.MERGER_ACQUISITION.value
        assert grade == ImpactGrade.HIGH
        assert score == 8

    def test_classify_maya_material_agreement(self):
        title = "דוח מיידי - כניסה להסכם מהותי לאספקת מערכות ביטחוניות מתקדמות"
        summary = "היקף ההתקשרות מוערך בכ-120 מיליון ש\"ח על פני שלוש שנים."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.MATERIAL_AGREEMENT.value
        assert grade == ImpactGrade.HIGH
        assert score == 8
        assert SECTaxonomyClassifier.get_maya_directional_bias(cat) == DirectionalStance.BULLISH

    def test_classify_maya_capital_raise(self):
        title = "דוח מיידי: גיוס הון בדרך של הנפקת מניות ואגרות חוב בבורסה בתל אביב"
        summary = "החברה פרסמה הצעת מדף לגיוס עד 100 מיליון ש\"ח."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.CAPITAL_RAISE.value
        assert grade in (ImpactGrade.HIGH, ImpactGrade.AVERAGE)
        assert score in (6, 7)

    def test_classify_maya_shareholder_meeting(self):
        title = "דוח מיידי בדבר זימון אסיפה כללית מיוחדת של בעלי המניות"
        summary = "על סדר היום: אישור מדיניות תגמול לנושאי משרה."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.SHAREHOLDER_MEETING.value
        assert grade == ImpactGrade.AVERAGE
        assert score == 5

    def test_classify_maya_immediate_report_general(self):
        title = "דוח מיידי - עדכון סטטוס פעילות תפעולית"
        summary = "החברה מדווחת על המשך שגרת העסקים הרגילה."
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title, summary)

        assert cat == MayaReportCategory.IMMEDIATE_REPORT.value
        assert grade == ImpactGrade.AVERAGE
        assert score == 5

    def test_classify_maya_non_israeli_disclosure(self):
        title = "European Central Bank leaves deposit facility rate unchanged at 3.75%"
        cat, grade, score = SECTaxonomyClassifier.classify_maya_report(title)

        assert cat is None
        assert grade is None
        assert score is None


class TestUnifiedClassifierAndPromptGeneration:
    """Test unified classification dictionary and prompt injection."""

    def test_unified_classify_helper(self):
        title = "Apple Inc. Form 8-K: Item 1.01 Entry into Material Definitive Agreement"
        summary = "Strategic acquisition of specialized photonics lab."
        result = SECTaxonomyClassifier.classify(title, summary)

        assert result["sec_filing_item"] == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT
        assert result["sec_impact_grade"] == ImpactGrade.HIGH
        assert result["sec_baseline_score"] == 8
        assert result["sec_directional_bias"] == DirectionalStance.BULLISH
        assert result["maya_category"] is None

    def test_prompt_generation_with_sec_filing_disclosure(self):
        article = RawArticle(
            title="Tesla Files Form 8-K Item 4.02 Non-Reliance on Previously Issued Financial Statements",
            url="https://sec.gov/edgar/data/123/tesla-8k",
            source_name="SEC EDGAR",
            summary="Audit committee determines previous quarterly revenue calculations require restatement.",
            content="Restatement impacts Q1 and Q2 recognition.",
        )
        system_prompt, user_prompt = build_analyst_prompt(article)

        # Prompt must include regulatory taxonomy in system prompt
        assert "REGULATORY DISCLOSURES & TAXONOMY" in system_prompt
        assert "Item 4.02" in system_prompt
        assert "Item 1.01" in system_prompt

        # User prompt must include detected filing item and baseline metrics
        assert "REGULATORY DISCLOSURE CONTEXT" in user_prompt
        assert "Item 4.02" in user_prompt
        assert "CRITICAL" in user_prompt
        assert "Baseline Confidence Score: 9" in user_prompt

    def test_prompt_generation_with_maya_disclosure(self):
        article = RawArticle(
            title="דוח מיידי: אזהרת רווח לרבעון השלישי לשנת 2026",
            url="https://maya.tase.co.il/reports/details/12345",
            source_name="בורסת תל אביב - מאיה",
            summary="החברה מודיעה על הרעה מהותית בתוצאות והפסד תפעולי.",
        )
        system_prompt, user_prompt = build_analyst_prompt(article)

        assert "REGULATORY DISCLOSURE CONTEXT" in user_prompt
        assert "Profit Warning" in user_prompt
        assert "Regulatory Taxonomy" in user_prompt

    def test_prompt_generation_without_regulatory_disclosure(self):
        article = RawArticle(
            title="US Consumer Sentiment Ticks Up in October Survey",
            url="https://reuters.com/us-consumer-sentiment",
            source_name="Reuters",
            summary="University of Michigan index rises slightly.",
        )
        _, user_prompt = build_analyst_prompt(article)

        assert "REGULATORY DISCLOSURE CONTEXT" not in user_prompt

    def test_analyzer_infers_sec_item_when_llm_omits_field(self):
        """Validates that analyzer automatically supplements missing sec_filing_item."""
        analyzer = MarketImpactAnalyzer()
        article = RawArticle(
            title="Acme Corp Form 8-K Item 1.01 Entry into a Material Definitive Agreement",
            url="https://sec.gov/edgar/acme",
            source_name="SEC EDGAR",
        )
        fake_llm_output = {
            "is_market_moving": True,
            "confidence_score": 8,
            "impact_grade": "HIGH",
            "affected_tickers": ["$ACM"],
            "affected_sectors": ["Technology"],
            "summary": "Acme signs material agreement.",
            "catalyst_thesis": "Substantial revenue expansion.",
            "status": "OPPORTUNITY_CANDIDATE",
            "directional_stance": "BULLISH",
            "time_horizon": "MEDIUM_TERM",
            "causal_chain": {
                "event_facts": "Acme executed material agreement.",
                "economic_mechanism": "Expands market reach.",
                "affected_sectors": ["Technology"],
                "beneficiaries": ["$ACM"],
                "negatively_exposed": [],
                "counterarguments": "Integration hurdles.",
                "invalidation_conditions": "Termination of agreement.",
            },
            "hard_gates": {
                "is_novel": True,
                "is_verified": True,
                "has_direct_exposure": True,
                "passes_all_gates": True,
            },
            "sec_filing_item": None,  # Omitted by LLM
        }

        result = analyzer._parse_and_validate(fake_llm_output, article=article)
        assert result.sec_filing_item == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT.value
