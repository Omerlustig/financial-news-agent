"""SEC 8-K & Israeli Corporate Disclosure (MAYA/TASE) Taxonomy Classifier.

Provides deterministic baseline impact severities, confidence scores, and directional
biases for regulatory filings pursuant to institutional Wall Street and TASE standards.
"""

import re
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union

from src.models import DirectionalStance, ImpactGrade, SECFilingItem


class MayaReportCategory(str, Enum):
    """Standardized reporting categories for Israeli MAYA / TASE corporate disclosures."""
    PROFIT_WARNING = "Profit Warning"
    MERGER_ACQUISITION = "M&A / Merger"
    MATERIAL_AGREEMENT = "Material Agreement"
    CAPITAL_RAISE = "Capital Raise"
    SHAREHOLDER_MEETING = "Shareholder Meeting"
    IMMEDIATE_REPORT = "Immediate Report"
    GENERAL = "General MAYA Report"

    def __str__(self) -> str:
        return self.value


# Precompiled regex patterns for SEC 8-K Items
RE_ITEM_1_01 = re.compile(
    r"\b(?:Item|ITEM)\s*1[\.\-_]0?1\b|"
    r"\b(?:entry\s+into\s+(?:a\s+)?material\s+definitive\s+agreement|"
    r"material\s+definitive\s+agreement|"
    r"entered\s+into\s+(?:a\s+)?material\s+(?:definitive\s+)?agreement|"
    r"enters\s+into\s+(?:a\s+)?material\s+(?:definitive\s+)?agreement|"
    r"definitive\s+merger\s+agreement|"
    r"major\s+partnership\s+agreement)\b",
    re.IGNORECASE,
)

RE_ITEM_1_02 = re.compile(
    r"\b(?:Item|ITEM)\s*1[\.\-_]0?2\b|"
    r"\b(?:termination\s+of\s+(?:a\s+)?material\s+definitive\s+agreement|"
    r"termination\s+of\s+(?:a\s+)?material\s+agreement|"
    r"terminated\s+material\s+agreement|"
    r"terminates\s+material\s+agreement|"
    r"termination\s+of\s+definitive\s+agreement)\b",
    re.IGNORECASE,
)

RE_ITEM_2_02 = re.compile(
    r"\b(?:Item|ITEM)\s*2[\.\-_]0?2\b|"
    r"\b(?:results\s+of\s+operations\s+and\s+financial\s+condition|"
    r"financial\s+condition\s+and\s+results\s+of\s+operations|"
    r"operating\s+results\s+and\s+financial\s+condition|"
    r"quarterly\s+results\s+8-k|"
    r"earnings\s+release\s+8-k)\b",
    re.IGNORECASE,
)

RE_ITEM_4_02 = re.compile(
    r"\b(?:Item|ITEM)\s*4[\.\-_]0?2\b|"
    r"\b(?:non-?reliance\s+on\s+previously\s+issued\s+financial\s+statements|"
    r"accounting\s+restatement|"
    r"financial\s+statement\s+restatement|"
    r"restatement\s+of\s+financial\s+statements|"
    r"restatement\s+of\s+previously\s+issued\s+financial\s+statements)\b",
    re.IGNORECASE,
)

RE_ITEM_5_02 = re.compile(
    r"\b(?:Item|ITEM)\s*5[\.\-_]0?2\b|"
    r"\b(?:departure\s+of\s+directors\s+or\s+(?:certain|principal)\s+officers|"
    r"election\s+of\s+directors(?:;\s*appointment\s+of\s+(?:certain|principal)\s+officers)?|"
    r"appointment\s+of\s+(?:certain|principal)\s+officers|"
    r"(?:resignation|departure|appointment|transition)\s+of\s+(?:the\s+)?(?:ceo|cfo|coo|chief\s+executive|chief\s+financial|president)|"
    r"(?:ceo|cfo|coo)\s+(?:resignation|departure|appointment|transition))\b",
    re.IGNORECASE,
)

RE_SEC_GENERAL = re.compile(
    r"\b(?:form\s*8-k|8-k\b|form\s*6-k|6-k\b|current\s+report\b|sec\s+(?:edgar\s+)?filing\b|filed\s+(?:a\s+)?form\s+8-k\b|filed\s+with\s+(?:the\s+)?sec\b)",
    re.IGNORECASE,
)

RE_OFFICER_KEY = re.compile(
    r"\b(ceo|cfo|chief\s+executive|chief\s+financial|coo|chief\s+operating|president)\b",
    re.IGNORECASE,
)

# Precompiled regex patterns for Israeli MAYA / TASE disclosures
RE_MAYA_SOURCE = re.compile(
    r"(?:מאיה|בורסה\s+לניירות\s+ערך|בורסת\s+תל\s*אביב|tase|maya\.tase|דוח\s+מיידי|דיווח\s+מיידי|הודעה\s+מיידית)",
    re.IGNORECASE,
)

RE_MAYA_PROFIT_WARNING = re.compile(
    r"(?:אזהרת\s+רווח|אזהרת\s+הכנסות|הרעה\s+מהותית\s+בתוצאות|ירידה\s+מהותית\s+ברווח|הפסד\s+צפוי|profit\s+warning)",
    re.IGNORECASE,
)

RE_MAYA_MERGER = re.compile(
    r"(?:הסכם\s+מיזוג|עסקת\s+מיזוג|מיזוג\s+חברות|מיזוג|הצעת\s+רכש\s+מלאה|הצעת\s+רכש|רכישת\s+פעילות|רכישת\s+מניות|עסקת\s+רכישה|merger|tender\s+offer)",
    re.IGNORECASE,
)

RE_MAYA_MATERIAL_AGREEMENT = re.compile(
    r"(?:כניסה\s+להסכם\s+מהותי|התקשרות\s+בהסכם\s+מהותי|הסכם\s+מהותי|הסכם\s+אסטרטגי|התקשרות\s+אסטרטגית|material\s+(?:definitive\s+)?agreement)",
    re.IGNORECASE,
)

RE_MAYA_CAPITAL_RAISE = re.compile(
    r"(?:גיוס\s+הון|הנפקת\s+מניות|הנפקת\s+אג\"?ח|הנפקת\s+אגרות\s+חוב|הקצאה\s+פרטית|הצעת\s+מדף|גיוס\s+חוב|הנפקה\s+לציבור|capital\s+raise|rights\s+offering|private\s+placement|public\s+offering)",
    re.IGNORECASE,
)

RE_MAYA_SHAREHOLDER_MEETING = re.compile(
    r"(?:זימון\s+אסיפה\s+כללית|זימון\s+אסיפה\s+מיוחדת|זימון\s+אסיפה|אסיפת\s+בעלי\s+מניות|shareholder\s+meeting|general\s+meeting)",
    re.IGNORECASE,
)

RE_MAYA_IMMEDIATE = re.compile(
    r"(?:דוח\s+מיידי|דיווח\s+מיידי|הודעה\s+מיידית|immediate\s+report)",
    re.IGNORECASE,
)


class SECTaxonomyClassifier:
    """Institutional classification engine for SEC EDGAR 8-K filings and Israeli MAYA reports."""

    @classmethod
    def classify_sec_item(
        cls, title: str, summary: str = ""
    ) -> Tuple[Optional[SECFilingItem], Optional[ImpactGrade], Optional[int]]:
        """Classifies SEC 8-K filing item, impact grade, and baseline confidence score.

        Returns:
            Tuple of (SECFilingItem, ImpactGrade, baseline_score), or (None, None, None)
            if not identified as an SEC filing / corporate disclosure.
        """
        text = f"{title} {summary}".strip()
        if not text:
            return None, None, None

        # 1. Item 4.02: Accounting Restatement / Non-Reliance on Financial Statements
        # Critical severity (Grade CRITICAL, baseline score 9, Bearish)
        if RE_ITEM_4_02.search(text):
            return SECFilingItem.ITEM_4_02_ACCOUNTING_RESTATEMENT, ImpactGrade.CRITICAL, 9

        # 2. Item 1.02: Termination of Material Definitive Agreement
        # High severity (Grade HIGH, baseline score 8, Bearish)
        if RE_ITEM_1_02.search(text):
            return SECFilingItem.ITEM_1_02_TERMINATION_AGREEMENT, ImpactGrade.HIGH, 8

        # 3. Item 1.01: Entry into Material Definitive Agreement (M&A / Partnership)
        # High severity (Grade HIGH, baseline score 8, Bullish/Neutral)
        if RE_ITEM_1_01.search(text):
            return SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT, ImpactGrade.HIGH, 8

        # 4. Item 2.02: Results of Operations and Financial Condition (Earnings)
        # High severity (Grade HIGH, baseline score 7)
        if RE_ITEM_2_02.search(text):
            return SECFilingItem.ITEM_2_02_EARNINGS_OPERATIONS, ImpactGrade.HIGH, 7

        # 5. Item 5.02: Departure / Election of Key Officers or Directors
        # Grade HIGH (score 7) for CEO/CFO/Key Executive; Grade AVERAGE (score 6) for general officers/directors
        if RE_ITEM_5_02.search(text):
            if RE_OFFICER_KEY.search(text):
                return SECFilingItem.ITEM_5_02_EXECUTIVE_CHANGE, ImpactGrade.HIGH, 7
            return SECFilingItem.ITEM_5_02_EXECUTIVE_CHANGE, ImpactGrade.AVERAGE, 6

        # 6. General SEC Filing / 8-K
        if RE_SEC_GENERAL.search(text):
            return SECFilingItem.ITEM_GENERAL, ImpactGrade.AVERAGE, 5

        return None, None, None

    @classmethod
    def classify_maya_report(
        cls, title: str, summary: str = ""
    ) -> Tuple[Optional[str], Optional[ImpactGrade], Optional[int]]:
        """Classifies Israeli MAYA / TASE corporate disclosure type, impact grade, and baseline score.

        Identifies:
        - Immediate Report (דוח מיידי / זימון אסיפה / הצעת רכש / מיזוג / כניסה להסכם מהותי):
          detects M&A, capital raise, profit warning.

        Returns:
            Tuple of (category_string, ImpactGrade, baseline_score), or (None, None, None)
            if not identified as an Israeli MAYA/TASE corporate disclosure.
        """
        text = f"{title} {summary}".strip()
        if not text:
            return None, None, None

        # An Israeli MAYA report requires Hebrew text or explicit MAYA / TASE context
        has_hebrew = bool(re.search(r"[\u0590-\u05FF]", text))
        has_maya_tase = bool(
            re.search(r"\b(?:maya|tase|tel\s*aviv\s*(?:stock\s*exchange)?)\b", text, re.IGNORECASE)
            or "maya.tase" in text.lower()
        )

        if not (has_hebrew or has_maya_tase):
            return None, None, None


        # 1. Profit Warning (אזהרת רווח / הרעה מהותית בתוצאות)
        if RE_MAYA_PROFIT_WARNING.search(text):
            return MayaReportCategory.PROFIT_WARNING.value, ImpactGrade.HIGH, 8

        # 2. M&A / Tender Offer (מיזוג / הצעת רכש / רכישת פעילות)
        if RE_MAYA_MERGER.search(text):
            return MayaReportCategory.MERGER_ACQUISITION.value, ImpactGrade.HIGH, 8

        # 3. Material Agreement (כניסה להסכם מהותי / התקשרות מהותית)
        if RE_MAYA_MATERIAL_AGREEMENT.search(text):
            return MayaReportCategory.MATERIAL_AGREEMENT.value, ImpactGrade.HIGH, 8

        # 4. Capital Raise / Offering (גיוס הון / הנפקת מניות / אג"ח / הקצאה פרטית)
        if RE_MAYA_CAPITAL_RAISE.search(text):
            return MayaReportCategory.CAPITAL_RAISE.value, ImpactGrade.HIGH, 7

        # 5. Shareholder Meeting / Proxy (זימון אסיפה כללית / מיוחדת)
        if RE_MAYA_SHAREHOLDER_MEETING.search(text):
            return MayaReportCategory.SHAREHOLDER_MEETING.value, ImpactGrade.AVERAGE, 5

        # 6. Immediate Report general fallback
        if RE_MAYA_IMMEDIATE.search(text):
            return MayaReportCategory.IMMEDIATE_REPORT.value, ImpactGrade.AVERAGE, 5

        # 7. General MAYA Report
        return MayaReportCategory.GENERAL.value, ImpactGrade.AVERAGE, 5

    @classmethod
    def get_sec_directional_bias(cls, item: SECFilingItem) -> DirectionalStance:
        """Determines expected directional market bias for a given SEC filing item."""
        if item in (
            SECFilingItem.ITEM_1_02_TERMINATION_AGREEMENT,
            SECFilingItem.ITEM_4_02_ACCOUNTING_RESTATEMENT,
        ):
            return DirectionalStance.BEARISH
        if item == SECFilingItem.ITEM_1_01_MATERIAL_AGREEMENT:
            return DirectionalStance.BULLISH
        return DirectionalStance.NEUTRAL

    @classmethod
    def get_maya_directional_bias(cls, category: Union[str, MayaReportCategory]) -> DirectionalStance:
        """Determines expected directional market bias for a given MAYA report category."""
        cat_str = category.value if hasattr(category, "value") else str(category)
        if MayaReportCategory.PROFIT_WARNING.value in cat_str:
            return DirectionalStance.BEARISH
        if (
            MayaReportCategory.MERGER_ACQUISITION.value in cat_str
            or MayaReportCategory.MATERIAL_AGREEMENT.value in cat_str
        ):
            return DirectionalStance.BULLISH
        return DirectionalStance.NEUTRAL

    @classmethod
    def classify(cls, title: str, summary: str = "") -> Dict[str, Any]:
        """Runs both SEC and MAYA taxonomy classifications across title and summary."""
        sec_item, sec_grade, sec_score = cls.classify_sec_item(title, summary)
        maya_cat, maya_grade, maya_score = cls.classify_maya_report(title, summary)
        return {
            "sec_filing_item": sec_item,
            "sec_impact_grade": sec_grade,
            "sec_baseline_score": sec_score,
            "sec_directional_bias": cls.get_sec_directional_bias(sec_item) if sec_item else None,
            "maya_category": maya_cat,
            "maya_impact_grade": maya_grade,
            "maya_baseline_score": maya_score,
            "maya_directional_bias": cls.get_maya_directional_bias(maya_cat) if maya_cat else None,
        }
