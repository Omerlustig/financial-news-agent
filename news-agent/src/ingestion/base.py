from abc import ABC, abstractmethod
from typing import List

from src.config import SourceConfig
from src.models import RawArticle


class BaseFetcher(ABC):
    """Abstract base class for all news source ingestion fetchers."""

    def __init__(self, source_config: SourceConfig) -> None:
        self.source_config = source_config

    @abstractmethod
    def fetch(self) -> List[RawArticle]:
        """Fetch and normalize articles from the configured source.

        Returns:
            List[RawArticle]: A list of parsed and normalized raw articles.
        """
        pass
