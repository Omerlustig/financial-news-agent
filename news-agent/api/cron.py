"""Vercel Serverless Function Cron Handler for Market News Agent.

Invoked automatically by Vercel Cron according to the schedule defined in
vercel.json. Executes a single news scan, deduplication, LLM analysis,
alerting, and persistence cycle via Pipeline.run_cycle().
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

# Ensure news-agent root is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from src.config import load_settings, load_sources
from src.pipeline import Pipeline

logger = logging.getLogger("news-agent.cron")


def handler(request: Any = None) -> Dict[str, Any]:
    """Execute a single news ingestion and market impact analysis cycle.

    Args:
        request: Incoming HTTP request object (or None for internal invocation).

    Returns:
        Dict[str, Any]: Structured execution outcome containing HTTP statusCode,
            status ('success' or 'error'), and cycle telemetry summary.
    """
    try:
        settings = load_settings()
        sources = load_sources()
        with Pipeline(settings=settings, sources=sources) as pipeline:
            summary = pipeline.run_cycle()

        result = {
            "status": "success",
            "summary": summary.to_dict(),
        }
        return {
            "statusCode": 200,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
            "status": "success",
            "summary": summary.to_dict(),
        }
    except Exception as exc:
        logger.error("Vercel serverless cron execution failed: %s", exc, exc_info=True)
        result = {
            "status": "error",
            "error": str(exc),
        }
        return {
            "statusCode": 500,
            "headers": {"Content-Type": "application/json"},
            "body": json.dumps(result),
            "status": "error",
            "error": str(exc),
        }
