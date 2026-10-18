import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from src.models import (
    RawArticle,
    MarketImpactAnalysis,
    ImpactGrade,
    EventStatus,
    EventLineageRecord,
    ProvenanceLevel,
    NoveltyState,
)
from src.config import Settings, SourceConfig, WatchlistSettings, load_settings, load_sources
from src.ingestion.rss import fetch_all_sources
from src.storage.base import BaseStorage
from src.storage.sqlite import SQLiteStorage
from src.analysis.analyzer import MarketImpactAnalyzer
from src.analysis.corroborator import EventCorroborator
from src.analysis.surge import SurgeVelocityEngine
from src.analysis.trend import TrendEngine
from src.analysis.technical import TradingViewEngine
from src.analysis.verification import FactVerificationEngine
from src.analysis.novelty import NoveltyEngine
from src.analysis.decision_gate import TwoLayerGateEngine
from src.notification.base import BaseNotifier
from src.notification.telegram import TelegramNotifier
from src.notification.console import ConsoleNotifier


logger = logging.getLogger(__name__)

GRADE_SEVERITY: Dict[ImpactGrade, int] = {
    ImpactGrade.LOW: 1,
    ImpactGrade.AVERAGE: 2,
    ImpactGrade.HIGH: 3,
    ImpactGrade.CRITICAL: 4,
}


@dataclass
class CycleSummary:
    """Execution summary of a news ingestion and analysis processing cycle."""
    fetched: int = 0
    new_unseen: int = 0
    analyzed: int = 0
    alerted: int = 0
    errors: int = 0
    top_catalysts: List[Any] = field(default_factory=list)

    @property
    def seen(self) -> int:
        """Alias for new_unseen articles evaluated during the cycle."""
        return self.new_unseen

    def __getitem__(self, item: str) -> Any:
        if item == "seen":
            return self.new_unseen
        try:
            return getattr(self, item)
        except AttributeError:
            raise KeyError(item)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "fetched": self.fetched,
            "new_unseen": self.new_unseen,
            "seen": self.seen,
            "analyzed": self.analyzed,
            "alerted": self.alerted,
            "errors": self.errors,
        }


class Pipeline:
    """Coordinates end-to-end ingestion, deduplication, LLM impact analysis,
    alert notification dispatch, and article persistence.
    """

    def __init__(
        self,
        storage: Optional[BaseStorage] = None,
        analyzer: Optional[MarketImpactAnalyzer] = None,
        notifier: Optional[BaseNotifier] = None,
        corroborator: Optional[EventCorroborator] = None,
        surge_engine: Optional[SurgeVelocityEngine] = None,
        trend_engine: Optional[TrendEngine] = None,
        fact_verification_engine: Optional[FactVerificationEngine] = None,
        novelty_engine: Optional[NoveltyEngine] = None,
        gate_engine: Optional[TwoLayerGateEngine] = None,
        technical_engine: Optional[TradingViewEngine] = None,
        sources: Optional[List[SourceConfig]] = None,
        fetcher: Optional[Callable[[], List[RawArticle]]] = None,
        min_score: Optional[int] = None,
        min_confidence_score: Optional[int] = None,
        min_impact_grade: Optional[Union[ImpactGrade, str]] = None,
        alert_on_all: Optional[bool] = None,
        force: bool = False,
        settings: Optional[Settings] = None,
        watchlist: Optional[WatchlistSettings] = None,
        fetch_timeout: float = 15.0,
        batch_size_per_source: Optional[int] = None,
        max_article_age_minutes: Optional[int] = None,
    ) -> None:
        self.settings = settings
        self.force = force
        self.fetch_timeout = fetch_timeout
        self.fetcher = fetcher

        # Watchlist resolution
        if watchlist is not None:
            self.watchlist = watchlist
        elif settings is not None and getattr(settings, "watchlist", None) is not None:
            self.watchlist = settings.watchlist
        else:
            self.watchlist = None

        self.corroborator = (
            corroborator
            if corroborator is not None
            else EventCorroborator(watchlist_settings=self.watchlist)
        )
        self.surge_engine = surge_engine if surge_engine is not None else SurgeVelocityEngine()
        self.trend_engine = trend_engine if trend_engine is not None else TrendEngine()

        # Batch size per source resolution (default to settings or 15)
        if batch_size_per_source is not None:
            self.batch_size_per_source = batch_size_per_source
        elif settings is not None and getattr(settings, "agent", None) is not None:
            self.batch_size_per_source = getattr(settings.agent, "batch_size_per_source", 15)
        else:
            self.batch_size_per_source = 15

        # Recency filter resolution (default to 15 minutes unless overridden or disabled)
        if max_article_age_minutes is not None:
            self.max_article_age_minutes = max_article_age_minutes
        elif settings is not None and getattr(settings, "agent", None) is not None:
            self.max_article_age_minutes = getattr(settings.agent, "max_article_age_minutes", 15)
        else:
            self.max_article_age_minutes = 15

        # Threshold resolution: min_score takes precedence over min_confidence_score
        if min_score is not None:
            self.min_confidence_score = min_score
        elif min_confidence_score is not None:
            self.min_confidence_score = min_confidence_score
        elif settings is not None:
            self.min_confidence_score = settings.filter.min_confidence_score
        else:
            self.min_confidence_score = 6

        # Severity threshold resolution
        if min_impact_grade is not None:
            if isinstance(min_impact_grade, str):
                self.min_impact_grade = ImpactGrade(min_impact_grade.upper())
            else:
                self.min_impact_grade = min_impact_grade
        elif settings is not None:
            self.min_impact_grade = settings.filter.min_impact_grade
        else:
            self.min_impact_grade = None

        # Alert bypass flag
        if alert_on_all is not None:
            self.alert_on_all = alert_on_all
        elif settings is not None:
            self.alert_on_all = settings.filter.alert_on_all
        else:
            self.alert_on_all = False

        # Ingestion sources
        if sources is not None:
            self.sources = sources
        elif settings is not None:
            self.sources = load_sources()
        else:
            self.sources = []

        # Storage subsystem
        if storage is not None:
            self.storage = storage
        elif settings is not None:
            self.storage = SQLiteStorage(settings.agent.database_path)
        else:
            self.storage = SQLiteStorage()

        # Analyzer subsystem
        if analyzer is not None:
            self.analyzer = analyzer
        elif settings is not None:
            self.analyzer = MarketImpactAnalyzer(
                settings.llm,
                fallback_on_error=True,
                watchlist_settings=self.watchlist,
            )
        else:
            self.analyzer = MarketImpactAnalyzer(
                fallback_on_error=True,
                watchlist_settings=self.watchlist,
            )

        # Notification subsystem
        if notifier is not None:
            self.notifier = notifier
        elif settings is not None:
            if not getattr(settings.telegram, "is_configured", False):
                logger.info("[LOCAL MODE] Telegram not configured - using ConsoleNotifier (alerts will print to terminal).")
                self.notifier = ConsoleNotifier()
            else:
                self.notifier = TelegramNotifier(
                    bot_token=settings.telegram.bot_token,
                    chat_id=settings.telegram.chat_id,
                )
        else:
            self.notifier = ConsoleNotifier()

        # Technical Analysis subsystem (TradingView confluence)
        self.technical_engine = technical_engine if technical_engine is not None else TradingViewEngine()

        # Institutional Intelligence engines
        self.fact_verification_engine = (
            fact_verification_engine if fact_verification_engine is not None else FactVerificationEngine()
        )
        self.novelty_engine = novelty_engine if novelty_engine is not None else NoveltyEngine()
        self.gate_engine = gate_engine if gate_engine is not None else TwoLayerGateEngine()


    @classmethod
    def from_settings(
        cls,
        settings: Optional[Settings] = None,
        sources: Optional[List[SourceConfig]] = None,
        config_path: Optional[str] = None,
        sources_path: Optional[str] = None,
        **overrides: Any,
    ) -> "Pipeline":
        """Factory initializing a fully configured pipeline using file settings."""
        if settings is None:
            settings = load_settings(config_path)
        if sources is None:
            sources = load_sources(sources_path)
        return cls(
            settings=settings,
            sources=sources,
            **overrides,
        )

    def should_alert(self, analysis: MarketImpactAnalysis) -> bool:
        """Evaluate whether an analyzed article meets notification criteria.

        Enforces the Institutional Hard Gate Decision Engine:
        - Telegram notification is dispatched ONLY when:
          * analysis.status == EventStatus.OPPORTUNITY_CANDIDATE
          * (analysis.hard_gates is None or analysis.hard_gates.passes_all_gates)
          * analysis.confidence_score >= min_confidence_score (or alert_on_all)
          * analysis.impact_grade >= min_impact_grade (or alert_on_all)
        - Non-qualifying events (WATCHLIST, INFORMATIONAL, UNVERIFIED, NO_ACTION,
          or failed hard gates) are suppressed, recorded with alerted=False, and audited.
        """
        # 1. Two-Layer Gate: Check gate_verdict if present
        if getattr(analysis, "two_layer_gate", None) is not None:
            if analysis.two_layer_gate.gate_verdict != "ALERT" and not self.alert_on_all:
                logger.info(
                    "Alert suppressed: two_layer_gate verdict is '%s' (audit reason: %s)",
                    analysis.two_layer_gate.gate_verdict,
                    analysis.two_layer_gate.rejection_reason,
                )
                return False

        # 2. Hard Gate: Workflow Status
        if analysis.status != EventStatus.OPPORTUNITY_CANDIDATE:
            logger.info(
                "Alert suppressed: status '%s' is not %s (audit reason: non-qualifying event status)",
                analysis.status,
                EventStatus.OPPORTUNITY_CANDIDATE,
            )
            return False

        # 2. Hard Gate: Quality gating verification, novelty, and direct exposure
        if analysis.hard_gates is not None and not analysis.hard_gates.passes_all_gates:
            logger.info(
                "Alert suppressed: hard gates failed [novel=%s, verified=%s, direct_exposure=%s] "
                "(audit reason: quality gating criteria failed)",
                analysis.hard_gates.is_novel,
                analysis.hard_gates.is_verified,
                analysis.hard_gates.has_direct_exposure,
            )
            return False

        # 3. CLI Testing Override (bypasses threshold/materiality for valid OPPORTUNITY_CANDIDATE events)
        if self.alert_on_all:
            return True

        # 4. Materiality / Market moving requirement
        if not analysis.is_market_moving:
            logger.info("Alert suppressed: article evaluated as not market moving (audit reason: is_market_moving=False)")
            return False

        # 5. Confidence score threshold
        if self.min_confidence_score is not None:
            if analysis.confidence_score < self.min_confidence_score:
                logger.info(
                    "Alert suppressed: confidence score %d < threshold %d (audit reason: below min_confidence_score)",
                    analysis.confidence_score,
                    self.min_confidence_score,
                )
                return False

        # 6. Severity / impact grade threshold
        if self.min_impact_grade is not None:
            analysis_rank = GRADE_SEVERITY.get(analysis.impact_grade, 0)
            min_rank = GRADE_SEVERITY.get(self.min_impact_grade, 0)
            if analysis_rank < min_rank:
                logger.info(
                    "Alert suppressed: impact grade %s < threshold %s (audit reason: below min_impact_grade)",
                    analysis.impact_grade,
                    self.min_impact_grade,
                )
                return False

        return True

    def process_articles(self, articles: List[RawArticle]) -> CycleSummary:
        """Process a list of ingested articles through deduplication, analysis,
        alert dispatch, and persistence.
        """
        fetched_count = len(articles)
        if not articles:
            return CycleSummary(fetched=0, new_unseen=0, analyzed=0, alerted=0, errors=0)

        # 1. Recency Filtering: discard articles older than max_article_age_minutes
        if self.max_article_age_minutes and self.max_article_age_minutes > 0:
            now_utc = datetime.now(timezone.utc)
            fresh_articles: List[RawArticle] = []
            stale_articles: List[RawArticle] = []
            for art in articles:
                if art.published_at is not None:
                    pub = (
                        art.published_at
                        if art.published_at.tzinfo is not None
                        else art.published_at.replace(tzinfo=timezone.utc)
                    )
                    age_min = (now_utc - pub).total_seconds() / 60.0
                    if age_min > self.max_article_age_minutes:
                        stale_articles.append(art)
                        continue
                fresh_articles.append(art)

            # Persist stale articles into SQLite so subsequent cycles skip them immediately
            if stale_articles and self.storage is not None:
                for stale in stale_articles:
                    try:
                        if not self.storage.is_article_seen(stale.content_hash):
                            self.storage.record_article(stale, alerted=False)
                    except Exception:
                        pass

            if stale_articles:
                logger.info(
                    "Filtered %d stale articles older than %d minutes; %d fresh articles remain.",
                    len(stale_articles),
                    self.max_article_age_minutes,
                    len(fresh_articles),
                )
            articles = fresh_articles

        # 2. Deduplication via SQLite
        if self.force:
            logger.info("Force mode active: evaluating all %d articles in the window (bypassing SQLite deduplication).", len(articles))
            unseen_articles = articles
        else:
            unseen_articles = self.storage.filter_unseen(articles)
        new_unseen_count = len(unseen_articles)

        analyzed_count = 0
        alerted_count = 0
        errors_count = 0
        top_catalysts: List[Tuple[RawArticle, MarketImpactAnalysis]] = []

        # 3. Multi-Source Event Clustering
        clusters = self.corroborator.cluster_articles(unseen_articles)

        # 4. Iterate through candidate event clusters
        for cluster in clusters:
            canonical = cluster.canonical_article

            # Lineage search: extract informative title tokens and primary ticker candidate
            title_tokens = [w.lower() for w in re.findall(r"\b[A-Za-z]{3,}\b", canonical.title)]
            ticker_matches = re.findall(r"\$([A-Z]{1,5})\b", canonical.title)
            primary_ticker_cand = f"${ticker_matches[0]}" if ticker_matches else ""

            existing_lineage: Optional[EventLineageRecord] = None
            if hasattr(self.storage, "find_existing_event") and callable(self.storage.find_existing_event):
                try:
                    res = self.storage.find_existing_event(primary_ticker_cand, title_tokens)
                    if isinstance(res, EventLineageRecord):
                        existing_lineage = res
                except Exception as exc:
                    logger.debug("Error checking existing lineage for '%s': %s", canonical.title, exc)

            # Novelty Assessment
            novelty_state, nov_score, nov_rationale = self.novelty_engine.assess_novelty(
                canonical, existing_lineage
            )

            # Fact Verification & Source Provenance
            provenance, statements, fact_score, is_verified = self.fact_verification_engine.extract_and_verify_facts(
                canonical, cluster.articles
            )

            try:
                analysis = self.analyzer.analyze(canonical)
            except Exception as exc:
                logger.error(
                    "Error analyzing article '%s' (%s): %s",
                    canonical.title,
                    canonical.url,
                    exc,
                    exc_info=True,
                )
                errors_count += 1
                continue

            # Populate institutional intelligence fields
            analysis.novelty_state = novelty_state
            analysis.provenance_level = provenance
            if existing_lineage and hasattr(existing_lineage, "event_id") and existing_lineage.event_id:
                analysis.event_id = existing_lineage.event_id
            else:
                analysis.event_id = f"evt_{abs(hash(canonical.title)) % 100000000}_{int(time.time())}"

            # Audit hard-gates against verification and novelty
            if analysis.hard_gates is not None:
                if not is_verified:
                    analysis.hard_gates.is_verified = False
                    analysis.hard_gates.passes_all_gates = False
                if novelty_state == NoveltyState.PRICED_IN_OR_RECYCLED:
                    analysis.hard_gates.is_novel = False
                    analysis.hard_gates.passes_all_gates = False

            # If article was not analyzed by LLM (rate limit exhausted or credentials unavailable),
            # do not persist to DB/cache so it will be retried in the next polling cycle.
            if not getattr(analysis, "is_llm_analyzed", True) and not self.alert_on_all:
                logger.warning(
                    "Article '%s' (%s) was not analyzed by LLM (rate-limited / error); skipping storage persistence so it will be retried in the next cycle.",
                    canonical.title,
                    canonical.url,
                )
                continue

            analyzed_count += 1

            # Apply multi-source corroboration weights, validation boost & source tracking
            analysis = self.corroborator.corroborate_analysis(analysis, cluster)

            # Apply surge velocity detection
            surge_level = self.surge_engine.detect_surge(cluster)
            analysis = self.surge_engine.apply_surge_metadata(analysis, surge_level)

            # Enrich analysis with market trend & buzz metrics
            if self.storage is not None and self.trend_engine is not None:
                try:
                    analysis = self.trend_engine.enrich_analysis_with_trend(self.storage, analysis)
                except Exception as exc:
                    logger.warning(
                        "Error enriching analysis with trend metrics for '%s': %s",
                        canonical.title,
                        exc,
                    )

            # Record LLM observability and token usage telemetry
            if analysis.token_usage is not None and self.storage is not None:
                try:
                    self.storage.record_llm_metric(
                        content_hash=canonical.content_hash,
                        model_name=analysis.token_usage.model_name or "unknown",
                        prompt_tokens=analysis.token_usage.prompt_tokens,
                        completion_tokens=analysis.token_usage.completion_tokens,
                        total_tokens=analysis.token_usage.total_tokens,
                        cost_usd=analysis.token_usage.estimated_cost_usd,
                    )
                except Exception as exc:
                    logger.error(
                        "Error recording LLM telemetry for article '%s': %s",
                        canonical.title,
                        exc,
                        exc_info=True,
                    )

            # Noise filtering: detect and suppress zero-value syndicated noise releases
            if self.corroborator.is_noise_article(canonical):
                logger.info(
                    "Article '%s' identified as syndicated noise release; suppressing alert.",
                    canonical.title,
                )
                analysis.status = EventStatus.NO_ACTION
                analysis.is_market_moving = False

            # Watchlist matching & priority scoring boost
            if self.watchlist is not None and getattr(self.watchlist, "enabled", True):
                matched_items: List[str] = []
                watched_tickers = {
                    (t.strip().upper() if t.strip().upper().startswith("$") else f"${t.strip().upper()}")
                    for t in getattr(self.watchlist, "tickers", [])
                    if t and t.strip()
                }
                watched_sectors = {
                    s.strip().lower()
                    for s in getattr(self.watchlist, "sectors", [])
                    if s and s.strip()
                }

                # Candidate tickers
                candidate_tickers = list(analysis.affected_tickers or [])
                if analysis.causal_chain:
                    candidate_tickers.extend(analysis.causal_chain.beneficiaries or [])
                    candidate_tickers.extend(analysis.causal_chain.negatively_exposed or [])

                for ticker in candidate_tickers:
                    clean_t = str(ticker).strip().upper()
                    norm_t = clean_t if clean_t.startswith("$") else f"${clean_t}"
                    if norm_t in watched_tickers and norm_t not in matched_items:
                        matched_items.append(norm_t)

                # Candidate sectors
                candidate_sectors = list(analysis.affected_sectors or [])
                if analysis.causal_chain:
                    candidate_sectors.extend(analysis.causal_chain.affected_sectors or [])

                for sector in candidate_sectors:
                    clean_s = str(sector).strip()
                    if clean_s.lower() in watched_sectors and clean_s not in matched_items:
                        matched_items.append(clean_s)

                if matched_items:
                    analysis.is_watchlist_match = True
                    analysis.matched_watchlist_items = matched_items
                    boost = getattr(self.watchlist, "score_boost", 1)
                    new_score = min(10, analysis.confidence_score + boost)
                    analysis.confidence_score = new_score
                    if new_score >= 7 and analysis.impact_grade in (ImpactGrade.LOW, ImpactGrade.AVERAGE):
                        analysis.impact_grade = ImpactGrade.HIGH
                        analysis.is_market_moving = True

            # TradingView Technical Confluence check for candidate tickers
            confluence_mod = 0.0
            if getattr(self, "technical_engine", None) is not None:
                candidate_tickers: List[str] = []
                if analysis.causal_chain:
                    candidate_tickers.extend(analysis.causal_chain.beneficiaries[:2])
                    candidate_tickers.extend(analysis.causal_chain.negatively_exposed[:2])
                if not candidate_tickers and analysis.affected_tickers:
                    candidate_tickers.extend(analysis.affected_tickers[:2])

                seen_tech_syms = set()
                stance_val = (
                    analysis.directional_stance.value
                    if hasattr(analysis.directional_stance, "value")
                    else str(analysis.directional_stance)
                )
                for sym in candidate_tickers:
                    clean_sym = str(sym).strip().upper()
                    if clean_sym and clean_sym not in seen_tech_syms and clean_sym.startswith("$"):
                        seen_tech_syms.add(clean_sym)
                        try:
                            tech_info = self.technical_engine.get_technical_summary(
                                clean_sym, directional_stance=stance_val
                            )
                            if tech_info:
                                analysis.technicals.append(tech_info)
                        except Exception as tech_err:
                            logger.debug("TradingView technicals failed for %s: %s", clean_sym, tech_err)

                if analysis.technicals:
                    primary_tech = analysis.technicals[0]
                    try:
                        confluence_mod, _, _ = self.technical_engine.calculate_technical_confluence(
                            stance=stance_val,
                            rec=primary_tech.recommendation,
                            rsi=primary_tech.rsi,
                            close=primary_tech.close_price,
                            sma50=primary_tech.sma50,
                            sma200=primary_tech.sma200,
                        )
                    except Exception as c_err:
                        logger.debug("Error computing technical confluence: %s", c_err)
                        confluence_mod = 0.0

            # Two-Layer Decision Gate evaluation
            if analysis.four_d_sentiment is not None:
                gate_min_opp = float(self.min_confidence_score) if self.min_confidence_score is not None else 8.0
                gate_min_event = min(8.0, float(self.min_confidence_score)) if self.min_confidence_score is not None else 8.0
            else:
                gate_min_opp = (
                    (float(self.min_confidence_score) / 10.0) * 3.5
                    if self.min_confidence_score is not None
                    else 8.0
                )
                gate_min_event = 6.0

            gate_decision = self.gate_engine.evaluate_gates(
                analysis,
                min_event_quality=gate_min_event,
                min_opportunity_score=gate_min_opp,
                technical_confluence_boost=confluence_mod,
            )
            analysis.two_layer_gate = gate_decision

            # SQLite Event Lineage persistence
            if hasattr(self.storage, "record_event_lineage") and callable(self.storage.record_event_lineage):
                try:
                    primary_sym = (
                        analysis.affected_tickers[0]
                        if analysis.affected_tickers
                        else (
                            analysis.causal_chain.beneficiaries[0]
                            if analysis.causal_chain and analysis.causal_chain.beneficiaries
                            else (
                                analysis.causal_chain.negatively_exposed[0]
                                if analysis.causal_chain and analysis.causal_chain.negatively_exposed
                                else (primary_ticker_cand or None)
                            )
                        )
                    )
                    initial_p = None
                    if analysis.technicals and analysis.technicals[0].close_price is not None:
                        initial_p = analysis.technicals[0].close_price

                    lineage_rec = EventLineageRecord(
                        event_id=analysis.event_id,
                        title=canonical.title,
                        primary_ticker=primary_sym or None,
                        first_seen=existing_lineage.first_seen if isinstance(existing_lineage, EventLineageRecord) else (canonical.published_at or datetime.now(timezone.utc)),
                        last_updated=datetime.now(timezone.utc),
                        source_count=analysis.corroboration_count,
                        all_sources=analysis.corroborating_sources or [canonical.source_name],
                        event_quality_score=gate_decision.event_quality_score,
                        opportunity_score=gate_decision.opportunity_score,
                        status=gate_decision.gate_verdict,
                        directional_stance=stance_val,
                        initial_price=initial_p,
                    )
                    self.storage.record_event_lineage(lineage_rec)
                except Exception as lin_err:
                    logger.debug("Error recording event lineage: %s", lin_err)

            # Filtering & alert criteria (enforcing Hard Gate Decision Engine)
            alert_warranted = self.should_alert(analysis)
            alert_delivered = False

            if alert_warranted:
                try:
                    alert_delivered = bool(self.notifier.send_alert(canonical, analysis))
                except Exception as exc:
                    logger.error(
                        "Error dispatching notification for article '%s': %s",
                        canonical.title,
                        exc,
                        exc_info=True,
                    )
                    alert_delivered = False

                if alert_delivered:
                    alerted_count += 1
            else:
                logger.info(
                    "Article '%s' (%s) routed to storage without alert (status=%s, score=%d, alerted=False).",
                    canonical.title,
                    canonical.url,
                    analysis.status,
                    analysis.confidence_score,
                )

            # Record candidates for the executive investment bottom line
            if (
                alert_delivered
                or analysis.status == EventStatus.OPPORTUNITY_CANDIDATE
                or analysis.confidence_score >= 6
                or (
                    analysis.causal_chain is not None
                    and (analysis.causal_chain.beneficiaries or analysis.causal_chain.negatively_exposed)
                )
            ):
                top_catalysts.append((canonical, analysis))

            # Storage persistence:
            # 1. Record canonical article with actual alerted status
            try:
                self.storage.record_article(
                    canonical,
                    analysis,
                    alerted=alert_delivered,
                )
            except Exception as exc:
                logger.error(
                    "Error recording article '%s' to storage: %s",
                    canonical.title,
                    exc,
                    exc_info=True,
                )
                errors_count += 1

            # 2. Record all corroborating secondary articles with alerted=False to prevent re-analysis
            for secondary in cluster.articles:
                if secondary.content_hash != canonical.content_hash:
                    try:
                        self.storage.record_article(
                            secondary,
                            analysis,
                            alerted=False,
                        )
                    except Exception as exc:
                        logger.error(
                            "Error recording corroborating article '%s' to storage: %s",
                            secondary.title,
                            exc,
                            exc_info=True,
                        )

        return CycleSummary(
            fetched=fetched_count,
            new_unseen=new_unseen_count,
            analyzed=analyzed_count,
            alerted=alerted_count,
            errors=errors_count,
            top_catalysts=top_catalysts,
        )

    def run_cycle(self, force: Optional[bool] = None) -> CycleSummary:
        """Execute a complete scan cycle across all configured sources."""
        if force is not None:
            self.force = force
        logger.info("Starting pipeline scan cycle across sources...")
        fetch_error = False
        batch_size = self.batch_size_per_source
        max_age = self.max_article_age_minutes
        try:
            if self.fetcher is not None:
                try:
                    articles = self.fetcher(
                        self.sources,
                        max_items_per_source=batch_size,
                        max_article_age_minutes=max_age,
                    )
                except TypeError:
                    try:
                        articles = self.fetcher(self.sources, max_items_per_source=batch_size)
                    except TypeError:
                        try:
                            articles = self.fetcher(self.sources)
                        except TypeError:
                            articles = self.fetcher()
            else:
                enable_etag = (
                    self.settings.agent.enable_etag_cache
                    if self.settings is not None and getattr(self.settings, "agent", None) is not None
                    else True
                )
                max_retries = (
                    getattr(self.settings.agent, "fetch_max_retries", 3)
                    if self.settings is not None and getattr(self.settings, "agent", None) is not None
                    else 3
                )
                retry_backoff = (
                    getattr(self.settings.agent, "fetch_retry_backoff", 1.0)
                    if self.settings is not None and getattr(self.settings, "agent", None) is not None
                    else 1.0
                )
                articles = fetch_all_sources(
                    self.sources,
                    timeout=self.fetch_timeout,
                    max_items_per_source=batch_size,
                    max_article_age_minutes=max_age,
                    storage=self.storage,
                    enable_etag_cache=enable_etag,
                    max_retries=max_retries,
                    retry_backoff=retry_backoff,
                )
        except Exception as exc:
            logger.error("Critical error during source ingestion: %s", exc, exc_info=True)
            articles = []
            fetch_error = True

        logger.info("Ingestion completed: %d raw articles fetched.", len(articles))
        summary = self.process_articles(articles)
        if fetch_error:
            summary.errors += 1

        logger.info(
            "Cycle finished: %d fetched, %d new/unseen, %d analyzed, %d alerted, %d errors.",
            summary.fetched,
            summary.new_unseen,
            summary.analyzed,
            summary.alerted,
            summary.errors,
        )
        if summary.alerted == 0:
            logger.info("No articles qualified for Telegram alerts in this scan cycle (0 alerts sent).")
        return summary

    def close(self) -> None:
        """Clean up storage connection and associated pipeline resources."""
        if hasattr(self.storage, "close") and callable(self.storage.close):
            self.storage.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
