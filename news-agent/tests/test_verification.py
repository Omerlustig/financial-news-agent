import pytest
from src.analysis.verification import FactVerificationEngine, extract_domain
from src.models import (
    ProvenanceLevel,
    RawArticle,
    StatementType,
)


@pytest.fixture
def engine():
    return FactVerificationEngine()


def test_extract_domain():
    assert extract_domain("https://www.sec.gov/edgar/data/12345/filing.htm") == "sec.gov"
    assert extract_domain("https://finance.yahoo.com/news/article.html") == "finance.yahoo.com"
    assert extract_domain("http://bloomberg.com/news/1") == "bloomberg.com"
    assert extract_domain("") == ""


def test_provenance_classification_primary_official(engine):
    # SEC EDGAR URL
    sec_art = RawArticle(
        title="Form 8-K Current Report",
        url="https://www.sec.gov/edgar/data/320193/000032019324000106/a8-k.htm",
        source_name="SEC EDGAR",
    )
    assert engine.classify_provenance(sec_art) == ProvenanceLevel.PRIMARY_OFFICIAL

    # MAYA Israel
    maya_art = RawArticle(
        title="Immediate Report on Corporate Event",
        url="https://maya.tase.co.il/reports/details/1623456",
        source_name="MAYA Israel",
    )
    assert engine.classify_provenance(maya_art) == ProvenanceLevel.PRIMARY_OFFICIAL

    # Federal Reserve
    fed_art = RawArticle(
        title="Federal Reserve Issues FOMC Statement",
        url="https://www.federalreserve.gov/newsevents/pressreleases/monetary20261018a.htm",
        source_name="Federal Reserve",
    )
    assert engine.classify_provenance(fed_art) == ProvenanceLevel.PRIMARY_OFFICIAL

    # PR Newswire
    prn_art = RawArticle(
        title="Nvidia Announces Expanded Partnership",
        url="https://www.prnewswire.com/news-releases/nvidia-announces-30211234.html",
        source_name="PR Newswire",
    )
    assert engine.classify_provenance(prn_art) == ProvenanceLevel.PRIMARY_OFFICIAL

    # Business Wire
    bw_art = RawArticle(
        title="Apple Reports Q4 Results",
        url="https://www.businesswire.com/news/home/20261018005001/en/",
        source_name="Business Wire",
    )
    assert engine.classify_provenance(bw_art) == ProvenanceLevel.PRIMARY_OFFICIAL

    # Corporate Investor Relations URL
    ir_art = RawArticle(
        title="Microsoft Investor Relations Press Release",
        url="https://investor.microsoft.com/press-releases/default.aspx",
        source_name="Microsoft IR",
    )
    assert engine.classify_provenance(ir_art) == ProvenanceLevel.PRIMARY_OFFICIAL


def test_provenance_classification_tier1_financial(engine):
    tier1_sources = [
        ("Financial Times", "https://www.ft.com/content/12345"),
        ("Bloomberg", "https://www.bloomberg.com/news/articles/2026-10-18/us-yields"),
        ("Reuters", "https://www.reuters.com/markets/asia/stocks-rally-2026-10-18/"),
        ("Wall Street Journal", "https://www.wsj.com/finance/investing/tech-earnings"),
        ("CNBC", "https://www.cnbc.com/2026/10/18/semiconductor-stocks.html"),
        ("Globes", "https://www.globes.co.il/news/article.aspx?did=1001487654"),
        ("Calcalist", "https://www.calcalist.co.il/market/article/bj12345"),
        ("TheMarker", "https://www.themarker.com/markets/2026-10-18/ty-article/000001"),
    ]

    for source_name, url in tier1_sources:
        art = RawArticle(title="Financial update", url=url, source_name=source_name)
        assert engine.classify_provenance(art) == ProvenanceLevel.TIER1_FINANCIAL, f"Failed for {source_name}"


def test_provenance_classification_commercial_news(engine):
    commercial_sources = [
        ("Fox Business", "https://www.foxbusiness.com/markets/wall-street-open"),
        ("Yahoo Finance", "https://finance.yahoo.com/news/apple-guidance-beat-1234.html"),
        ("TechCrunch", "https://techcrunch.com/2026/10/18/ai-startup-funding/"),
        ("Seeking Alpha", "https://seekingalpha.com/news/4123456-nvidia-earnings"),
        ("AllSides", "https://www.allsides.com/news/2026-10-18/headline"),
    ]

    for source_name, url in commercial_sources:
        art = RawArticle(title="Commercial market news", url=url, source_name=source_name)
        assert engine.classify_provenance(art) == ProvenanceLevel.COMMERCIAL_NEWS, f"Failed for {source_name}"


def test_provenance_classification_unverified_blog(engine):
    blog_sources = [
        ("Tech Speculator Blog", "https://speculative-tech.substack.com/p/rumors"),
        ("Market Whispers Medium", "https://medium.com/@trader123/my-thesis"),
        ("Unknown Forum", "https://some-crypto-blog.xyz/post/123"),
    ]

    for source_name, url in blog_sources:
        art = RawArticle(title="Speculative commentary", url=url, source_name=source_name)
        assert engine.classify_provenance(art) == ProvenanceLevel.UNVERIFIED_BLOG, f"Failed for {source_name}"


def test_statement_classification(engine):
    # 1. Verified Fact
    fact_stmt, cues = engine.classify_statement(
        "In a Form 8-K filed with the SEC, the company disclosed a definitive merger agreement."
    )
    assert fact_stmt == StatementType.VERIFIED_FACT
    assert len(cues) > 0

    court_stmt, cues = engine.classify_statement(
        "A federal judge ruled that the proposed acquisition violates antitrust laws."
    )
    assert court_stmt == StatementType.VERIFIED_FACT

    # 2. Attributed Claim
    claim_stmt, cues = engine.classify_statement(
        "CEO Jensen Huang said that demand for Blackwell chips continues to exceed expectations."
    )
    assert claim_stmt == StatementType.ATTRIBUTED_CLAIM
    assert len(cues) > 0

    claim_spokesperson, cues = engine.classify_statement(
        "According to company spokesperson Sarah Connor, operations will resume on Monday."
    )
    assert claim_spokesperson == StatementType.ATTRIBUTED_CLAIM

    # 3. Analyst Opinion
    analyst_stmt, cues = engine.classify_statement(
        "Analysts at Goldman Sachs upgraded the stock to Overweight with a price target of $160."
    )
    assert analyst_stmt == StatementType.ANALYST_OPINION
    assert len(cues) > 0

    note_stmt, cues = engine.classify_statement(
        "In a research note to clients, Morgan Stanley cut its price target to $120."
    )
    assert note_stmt == StatementType.ANALYST_OPINION

    # 4. Anonymous Rumor
    rumor_stmt, cues = engine.classify_statement(
        "The company is reportedly considering a sale of its cloud unit, according to sources familiar with the matter."
    )
    assert rumor_stmt == StatementType.ANONYMOUS_RUMOR
    assert len(cues) > 0

    unnamed_stmt, cues = engine.classify_statement(
        "Unnamed sources said talks are ongoing between the two chipmakers."
    )
    assert unnamed_stmt == StatementType.ANONYMOUS_RUMOR


def test_rumor_penalty_suppresses_verification(engine):
    # Tier 1 source reporting anonymous rumor without official filing
    rumor_art = RawArticle(
        title="Apple in talks to acquire AI startup, sources familiar with the matter say",
        url="https://www.bloomberg.com/news/articles/2026-10-18/apple-ai-talks",
        source_name="Bloomberg",
        summary="Apple is reportedly discussing an acquisition, according to people familiar with the deal.",
    )

    prov, stmts, score, is_verified = engine.extract_and_verify_facts(rumor_art)
    assert prov == ProvenanceLevel.TIER1_FINANCIAL
    # Baseline for Tier 1 is 8.0, rumor penalty of -2.5 reduces score to 5.5
    assert score == 5.5
    # Verification gate fails due to unconfirmed rumor
    assert is_verified is False


def test_rumor_with_primary_confirmation_clears_penalty(engine):
    # Tier 1 reporting rumor, but corroborated by official SEC 8-K filing
    news_art = RawArticle(
        title="Apple reportedly negotiating AI acquisition, sources say",
        url="https://www.bloomberg.com/news/articles/2026-10-18/apple-ai",
        source_name="Bloomberg",
    )
    primary_art = RawArticle(
        title="Apple Inc Form 8-K Material Agreement Disclosed",
        url="https://www.sec.gov/edgar/data/320193/apple8k.htm",
        source_name="SEC EDGAR",
    )

    prov, stmts, score, is_verified = engine.extract_and_verify_facts(
        news_art, corroborating_articles=[primary_art]
    )
    # Primary confirmation exists across corroboration set
    # Baseline 8.0 + Corroboration boost +1.0 (2 independent sources) = 9.0, no unconfirmed rumor penalty
    assert score >= 8.0
    assert is_verified is True


def test_corroboration_boost_and_thresholds(engine):
    # 1. Commercial news single source: baseline 6.5 -> not verified (needs >= 7.0)
    single_comm = RawArticle(
        title="Target announces new distribution hub expansion",
        url="https://finance.yahoo.com/news/target-expansion.html",
        source_name="Yahoo Finance",
        summary="Target confirmed in a statement that it opened a new Midwest distribution center.",
    )
    prov, stmts, score, is_verified = engine.extract_and_verify_facts(single_comm)
    assert prov == ProvenanceLevel.COMMERCIAL_NEWS
    assert score == 6.5
    assert is_verified is False

    # 2. Commercial news with 2 independent newsrooms: 6.5 + 1.0 = 7.5 -> verified!
    corrob_fox = RawArticle(
        title="Target opens Midwest hub",
        url="https://www.foxbusiness.com/markets/target-midwest",
        source_name="Fox Business",
        summary="Target confirmed new facility opening today.",
    )
    prov, stmts, score, is_verified = engine.extract_and_verify_facts(
        single_comm, corroborating_articles=[corrob_fox]
    )
    assert score == 7.5
    assert is_verified is True

    # 3. Commercial news with 3+ independent newsrooms: 6.5 + 2.0 = 8.5 -> verified!
    corrob_tc = RawArticle(
        title="Target logistics expansion tech",
        url="https://techcrunch.com/2026/10/18/target-automation",
        source_name="TechCrunch",
    )
    prov, stmts, score, is_verified = engine.extract_and_verify_facts(
        single_comm, corroborating_articles=[corrob_fox, corrob_tc]
    )
    assert score == 8.5
    assert is_verified is True


def test_syndicated_wires_not_counted_as_independent(engine):
    # Both articles are republishing the same Reuters wire
    art1 = RawArticle(
        title="Delta Cuts Flights Amid Rising Fuel Expenses - Reuters",
        url="https://finance.yahoo.com/news/delta-cuts-flights.html",
        source_name="Yahoo Finance",
        content="NEW YORK (Reuters) - Delta Air Lines announced it will trim capacity by 2%.",
    )
    art2 = RawArticle(
        title="Delta Air Lines cuts capacity (Reuters)",
        url="https://www.foxbusiness.com/markets/delta-capacity-reuters",
        source_name="Fox Business",
        content="By Reuters - Delta said rising jet fuel expenses forced capacity reductions.",
    )

    count = engine.count_independent_sources(art1, [art2])
    # Both resolve to the 'reuters' originating wire entity
    assert count == 1

    # Verify score does not receive independent boost
    prov, stmts, score, is_verified = engine.extract_and_verify_facts(art1, [art2])
    # Commercial baseline 6.5, boost 0.0 -> score 6.5, is_verified False
    assert score == 6.5
    assert is_verified is False


def test_primary_official_is_verified_standalone(engine):
    sec_art = RawArticle(
        title="Broadcom Inc. Form 8-K Item 1.01 Material Definitive Agreement",
        url="https://www.sec.gov/edgar/data/1730168/000173016824000050/avgo-8k.htm",
        source_name="SEC EDGAR",
        summary="Broadcom entered into an amendment to its credit facility.",
    )
    prov, stmts, score, is_verified = engine.extract_and_verify_facts(sec_art)
    assert prov == ProvenanceLevel.PRIMARY_OFFICIAL
    assert score == 9.5
    assert is_verified is True


def test_unverified_blog_cannot_pass_verification_even_with_corroboration(engine):
    blog1 = RawArticle(
        title="New speculative leak on GPU release",
        url="https://gpu-rumors.substack.com/p/leak1",
        source_name="Substack",
        summary="A new blog post claims next gen GPUs will launch early.",
    )
    blog2 = RawArticle(
        title="More details on GPU release leak",
        url="https://tech-whispers.medium.com/post-456",
        source_name="Medium",
    )
    blog3 = RawArticle(
        title="Speculative discussion on GPU launch",
        url="https://some-hardware-blog.org/2026/gpu",
        source_name="Hardware Blog",
    )

    prov, stmts, score, is_verified = engine.extract_and_verify_facts(
        blog1, corroborating_articles=[blog2, blog3]
    )
    assert prov == ProvenanceLevel.UNVERIFIED_BLOG
    # Baseline 4.0 + 2.0 (3 sources) = 6.0 (< 7.0)
    assert score == 6.0
    assert is_verified is False
