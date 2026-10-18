"""
Firebase Cloud Functions (2nd Gen) entrypoint for Market News Agent.
Schedules a news intelligence scan every 15 minutes, synchronizing persistent
SQLite state with Firebase Cloud Storage to prevent duplicate alerts.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

# Configure standard logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("firebase_news_agent")

try:
    from firebase_functions import options, scheduler_fn
    from firebase_admin import initialize_app, storage
    _FIREBASE_AVAILABLE = True
except ImportError:
    _FIREBASE_AVAILABLE = False
    scheduler_fn = None
    options = None
    storage = None


def get_storage_bucket_name() -> Optional[str]:
    """Retrieve the target Firebase Cloud Storage bucket name from environment."""
    return (
        os.getenv("FIREBASE_STORAGE_BUCKET")
        or os.getenv("GCS_BUCKET_NAME")
        or (f"{os.getenv('GCLOUD_PROJECT')}.appspot.com" if os.getenv("GCLOUD_PROJECT") else None)
    )


def sync_db_from_cloud_storage(local_path: str, bucket_name: Optional[str]) -> bool:
    """Download the SQLite database from Firebase Storage if it exists."""
    if not _FIREBASE_AVAILABLE or not bucket_name or storage is None:
        logger.info("Cloud storage sync skipped; using local file: %s", local_path)
        return False

    try:
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob("data/agent.db")
        if blob.exists():
            os.makedirs(os.path.dirname(os.path.abspath(local_path)), exist_ok=True)
            blob.download_to_filename(local_path)
            logger.info("Downloaded persistent database from gs://%s/data/agent.db to %s", bucket_name, local_path)
            return True
        else:
            logger.info("gs://%s/data/agent.db does not exist yet; initializing fresh database.", bucket_name)
            return False
    except Exception as exc:
        logger.warning("Could not download database from Cloud Storage: %s. Continuing with fresh DB.", exc)
        return False


def sync_db_to_cloud_storage(local_path: str, bucket_name: Optional[str]) -> bool:
    """Upload the updated SQLite database to Firebase Storage."""
    if not _FIREBASE_AVAILABLE or not bucket_name or storage is None or not os.path.exists(local_path):
        return False

    try:
        bucket = storage.bucket(bucket_name)
        blob = bucket.blob("data/agent.db")
        blob.upload_from_filename(local_path)
        logger.info("Uploaded updated database to gs://%s/data/agent.db", bucket_name)
        return True
    except Exception as exc:
        logger.error("Failed to upload database to Cloud Storage: %s", exc)
        return False


def run_agent_scan() -> Dict[str, Any]:
    """Execute a single news intelligence scan cycle."""
    from src.config import load_settings, load_sources
    from src.pipeline import Pipeline

    local_db_path = os.getenv("DATABASE_PATH", "/tmp/agent.db")
    bucket_name = get_storage_bucket_name()

    # 1. Sync persistent SQLite DB from Firebase Storage
    sync_db_from_cloud_storage(local_db_path, bucket_name)

    # 2. Configure pipeline with database path
    settings = load_settings()
    settings.agent.database_path = local_db_path
    sources = load_sources()

    # 3. Execute cycle
    with Pipeline(settings, sources) as pipeline:
        summary = pipeline.run_cycle()

    # 4. Sync persistent SQLite DB back to Firebase Storage
    sync_db_to_cloud_storage(local_db_path, bucket_name)

    return {
        "fetched": summary.fetched,
        "analyzed": summary.analyzed,
        "alerted": summary.alerted,
        "duration_seconds": summary.duration_seconds,
    }


# Initialize Firebase app and define scheduler function if Firebase is available
if _FIREBASE_AVAILABLE:
    try:
        initialize_app()
    except Exception as init_err:
        logger.debug("Firebase app initialization note: %s", init_err)

    @scheduler_fn.on_schedule(
        schedule="every 15 minutes",
        timeout_sec=300,
        memory=options.MemoryOption.MB_512,
    )
    def scan_financial_news(event: scheduler_fn.ScheduledEvent) -> None:
        """Triggered automatically every 15 minutes by Cloud Scheduler."""
        logger.info("Executing scheduled news intelligence cycle (Event ID: %s)", getattr(event, "id", "local"))
        result = run_agent_scan()
        logger.info("Completed news intelligence cycle: %s", result)
