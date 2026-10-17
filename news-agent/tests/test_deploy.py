import os
from pathlib import Path
import yaml
import pytest


def test_dockerfile_structure():
    """Verify Dockerfile meets security, non-root user, and entrypoint standards."""
    root_dir = Path(__file__).resolve().parent.parent
    dockerfile_path = root_dir / "Dockerfile"
    assert dockerfile_path.exists(), "Dockerfile must exist in news-agent/"

    content = dockerfile_path.read_text()
    assert "python:3.11-slim" in content
    assert "appuser" in content
    assert "/app/data" in content
    assert "ENTRYPOINT" in content
    assert "run" in content
    assert "--daemon" in content
    assert "--interval" in content
    assert "15" in content


def test_docker_compose_valid_yaml():
    """Verify docker-compose.yml is valid YAML and defines expected services."""
    root_dir = Path(__file__).resolve().parent.parent
    compose_path = root_dir / "docker-compose.yml"
    assert compose_path.exists(), "docker-compose.yml must exist in news-agent/"

    content = compose_path.read_text()
    data = yaml.safe_load(content)

    assert "services" in data
    assert "news-agent" in data["services"]
    svc = data["services"]["news-agent"]

    assert svc.get("restart") == "unless-stopped"
    assert ".env" in svc.get("env_file", [])
    volumes = svc.get("volumes", [])
    has_data_vol = any("./data:/app/data" in v for v in volumes)
    assert has_data_vol, "docker-compose must mount ./data:/app/data"


def test_systemd_service_file():
    """Verify systemd service configuration directives."""
    root_dir = Path(__file__).resolve().parent.parent
    service_path = root_dir / "financial-news-agent.service"
    assert service_path.exists(), "financial-news-agent.service must exist"

    redundant_service = root_dir / "news-agent.service"
    assert not redundant_service.exists(), "news-agent.service should be removed in favor of financial-news-agent.service"

    content = service_path.read_text()
    assert "Restart=always" in content
    assert "RestartSec=30" in content
    assert "/etc/financial-news-agent.env" in content
    assert "python -m src.main run --daemon --interval 15" in content


def test_readme_comprehensive_content():
    """Verify README.md covers all required architectural and deployment topics."""
    root_dir = Path(__file__).resolve().parent.parent
    readme_path = root_dir / "README.md"
    root_readme_path = root_dir.parent / "README.md"

    assert readme_path.exists(), "news-agent/README.md must exist"
    assert root_readme_path.exists(), "Root README.md must exist"

    content = readme_path.read_text()
    assert "Architecture Overview" in content
    assert "Quickstart" in content
    assert "run-once" in content
    assert "--daemon" in content
    assert "Telegram Integration Guide" in content
    assert "@BotFather" in content
    assert "test-telegram" in content
    assert "Deployment Guide" in content
    assert "Docker & Docker Compose" in content
    assert "Systemd" in content
    assert "Vercel" in content
    assert "Cloud Run" in content or "Firebase" in content
    assert "test-connections" in content
    assert "test-source" in content
    assert "stats" in content


def test_vercel_manifest():
    """Verify vercel.json defines cron schedule for /api/cron every 15 minutes."""
    import json
    root_dir = Path(__file__).resolve().parent.parent
    vercel_path = root_dir / "vercel.json"
    assert vercel_path.exists(), "vercel.json must exist in news-agent/"

    content = json.loads(vercel_path.read_text())
    assert "crons" in content
    assert len(content["crons"]) >= 1

    cron_entry = content["crons"][0]
    assert cron_entry.get("path") == "/api/cron"
    assert cron_entry.get("schedule") == "*/15 * * * *"


def test_serverless_cron_handler():
    """Verify api/cron.py handler invokes Pipeline.run_cycle and returns JSON status and summary."""
    from unittest.mock import patch, MagicMock
    from api.cron import handler
    from src.pipeline import CycleSummary

    mock_summary = CycleSummary(fetched=5, new_unseen=2, analyzed=2, alerted=1, errors=0)

    with patch("api.cron.Pipeline") as mock_pipeline_cls:
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.run_cycle.return_value = mock_summary
        mock_pipeline_cls.return_value = mock_instance

        resp = handler()
        assert resp["statusCode"] == 200
        assert resp["status"] == "success"
        assert resp["summary"]["fetched"] == 5
        assert resp["summary"]["alerted"] == 1


def test_serverless_cron_handler_error_isolation():
    """Verify api/cron.py catches unhandled pipeline exceptions and returns 500 error JSON."""
    from unittest.mock import patch
    from api.cron import handler

    with patch("api.cron.Pipeline", side_effect=RuntimeError("Database lock failure")):
        resp = handler()
        assert resp["statusCode"] == 500
        assert resp["status"] == "error"
        assert "Database lock failure" in resp["error"]


def test_deploy_cloudrun_script():
    """Verify deploy_cloudrun.sh exists, is executable, and contains required deployment steps."""
    root_dir = Path(__file__).resolve().parent.parent
    script_path = root_dir / "deploy_cloudrun.sh"
    assert script_path.exists(), "deploy_cloudrun.sh must exist in news-agent/"
    assert os.access(script_path, os.X_OK), "deploy_cloudrun.sh must be executable"

    content = script_path.read_text()
    assert "set -euo pipefail" in content
    assert "gcloud builds submit" in content
    assert "gcloud run jobs" in content or "gcloud run deploy" in content
    assert "gcloud scheduler jobs" in content
    assert "*/15 * * * *" in content

