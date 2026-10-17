import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Union

from src.models import RawArticle, MarketImpactAnalysis, ImpactGrade, EventStatus
from src.config import Settings, SourceConfig, load_settings, load_sources
from src.ingestion.rss import fetch_all_sources
from src.storage.base import BaseStorage
from src.storage.sqlite import SQLiteStorage
from src.analysis.analyzer import MarketImpactAnalyzer
from src.notification.base import BaseNotifier
from src.notification.telegram import TelegramNotifier

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
        sources: Optional[List[SourceConfig]] = None,
        fetcher: Optional[Callable[[], List[RawArticle]]] = None,
        min_score: Optional[int] = None,
        min_confidence_score: Optional[int] = None,
        min_impact_grade: Optional[Union[ImpactGrade, str]] = None,
        alert_on_all: Optional[bool] = None,
        settings: Optional[Settings] = None,
        fetch_timeout: float = 15.0,
    ) -> None:
        self.settings = settings
        self.fetch_timeout = fetch_timeout
        self.fetcher = fetcher

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
            self.analyzer = MarketImpactAnalyzer(settings.llm)
        else:
            self.analyzer = MarketImpactAnalyzer()

        # Notification subsystem
        if notifier is not None:
            self.notifier = notifier
        elif settings is not None:
            self.notifier = TelegramNotifier(
                bot_token=settings.telegram.bot_token,
                chat_id=settings.telegram.chat_id,
            )
        else:
            self.notifier = TelegramNotifier()

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
        # 1. Hard Gate: Workflow Status
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

        # 3. CLI Testing Override
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

        # 1. Deduplication via SQLite
        unseen_articles = self.storage.filter_unseen(articles)
        new_unseen_count = len(unseen_articles)

        analyzed_count = 0
        alerted_count = 0
        errors_count = 0

        # 2. Iterate through candidate unseen items
        for article in unseen_articles:
            try:
                analysis = self.analyzer.analyze(article)
                analyzed_count += 1
            except Exception as exc:
                logger.error(
                    "Error analyzing article '%s' (%s): %s",
                    article.title,
                    article.url,
                    exc,
                    exc_info=True,
                )
                errors_count += 1
                continue

            # Record LLM observability and token usage telemetry
            if analysis.token_usage is not None and self.storage is not None:
                try:
                    self.storage.record_llm_metric(
                        content_hash=article.content_hash,
                        model_name=analysis.token_usage.model_name or "unknown",
                        prompt_tokens=analysis.token_usage.prompt_tokens,
                        completion_tokens=analysis.token_usage.completion_tokens,
                        total_tokens=analysis.token_usage.total_tokens,
                        cost_usd=analysis.token_usage.estimated_cost_usd,
                    )
                except Exception as exc:
                    logger.error(
                        "Error recording LLM telemetry for article '%s': %s",
                        article.title,
                        exc,
                        exc_info=True,
                    )

            # 3. Filtering & alert criteria (enforcing Hard Gate Decision Engine)
            alert_warranted = self.should_alert(analysis)
            alert_delivered = False

            if alert_warranted:
                try:
                    alert_delivered = bool(self.notifier.send_alert(article, analysis))
                except Exception as exc:
                    logger.error(
                        "Error dispatching notification for article '%s': %s",
                        article.title,
                        exc,
                        exc_info=True,
                    )
                    alert_delivered = False

                if alert_delivered:
                    alerted_count += 1
            else:
                logger.info(
                    "Article '%s' (%s) routed to storage without alert (status=%s, score=%d, alerted=False).",
                    article.title,
                    article.url,
                    analysis.status,
                    analysis.confidence_score,
                )

            # 4. Storage persistence (non-alerted articles saved with alerted=False to prevent re-analysis)
            try:
                self.storage.record_article(
                    article,
                    analysis,
                    alerted=alert_delivered,
                )
            except Exception as exc:
                logger.error(
                    "Error recording article '%s' to storage: %s",
                    article.title,
                    exc,
                    exc_info=True,
                )
                errors_count += 1

        return CycleSummary(
            fetched=fetched_count,
            new_unseen=new_unseen_count,
            analyzed=analyzed_count,
            alerted=alerted_count,
            errors=errors_count,
        )

    def run_cycle(self) -> CycleSummary:
        """Execute a complete scan cycle across all configured sources."""
        logger.info("Starting pipeline scan cycle across sources...")
        fetch_error = False
        batch_size = (
            self.settings.agent.batch_size_per_source
            if self.settings is not None and getattr(self.settings, "agent", None) is not None
            else None
        )
        try:
            if self.fetcher is not None:
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
                articles = fetch_all_sources(
                    self.sources,
                    timeout=self.fetch_timeout,
                    max_items_per_source=batch_size,
                    storage=self.storage,
                    enable_etag_cache=enable_etag,
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
        return summary

    def close(self) -> None:
        """Clean up storage connection and associated pipeline resources."""
        if hasattr(self.storage, "close") and callable(self.storage.close):
            self.storage.close()

    def __enter__(self) -> "Pipeline":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()
