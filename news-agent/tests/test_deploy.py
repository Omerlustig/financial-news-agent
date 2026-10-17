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
