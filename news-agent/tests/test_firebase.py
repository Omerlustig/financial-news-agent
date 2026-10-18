"""Tests for Firebase Cloud Functions entrypoint and cloud storage sync."""
import os
from unittest.mock import MagicMock, patch

import pytest

import main


def test_get_storage_bucket_name():
    with patch.dict(os.environ, {"FIREBASE_STORAGE_BUCKET": "my-custom-bucket.appspot.com"}):
        assert main.get_storage_bucket_name() == "my-custom-bucket.appspot.com"

    with patch.dict(os.environ, {"FIREBASE_STORAGE_BUCKET": "", "GCS_BUCKET_NAME": "gcs-bucket"}, clear=True):
        assert main.get_storage_bucket_name() == "gcs-bucket"

    with patch.dict(os.environ, {"FIREBASE_STORAGE_BUCKET": "", "GCS_BUCKET_NAME": "", "GCLOUD_PROJECT": "demo-proj"}, clear=True):
        assert main.get_storage_bucket_name() == "demo-proj.appspot.com"


def test_sync_db_from_cloud_storage_when_disabled():
    with patch("main._FIREBASE_AVAILABLE", False):
        result = main.sync_db_from_cloud_storage("/tmp/test.db", None)
        assert result is False


def test_sync_db_from_cloud_storage_success(tmp_path):
    mock_storage = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_blob.exists.return_value = True
    mock_bucket.blob.return_value = mock_blob
    mock_storage.bucket.return_value = mock_bucket

    target_file = str(tmp_path / "sub" / "agent.db")

    with patch("main._FIREBASE_AVAILABLE", True), patch("main.storage", mock_storage):
        result = main.sync_db_from_cloud_storage(target_file, "my-bucket")
        assert result is True
        mock_bucket.blob.assert_called_once_with("data/agent.db")
        mock_blob.download_to_filename.assert_called_once_with(target_file)


def test_sync_db_to_cloud_storage_success(tmp_path):
    mock_storage = MagicMock()
    mock_bucket = MagicMock()
    mock_blob = MagicMock()
    mock_bucket.blob.return_value = mock_blob
    mock_storage.bucket.return_value = mock_bucket

    db_file = tmp_path / "agent.db"
    db_file.write_text("sqlite-mock-data")

    with patch("main._FIREBASE_AVAILABLE", True), patch("main.storage", mock_storage):
        result = main.sync_db_to_cloud_storage(str(db_file), "my-bucket")
        assert result is True
        mock_blob.upload_from_filename.assert_called_once_with(str(db_file))


def test_run_agent_scan(tmp_path):
    db_file = str(tmp_path / "test_run.db")

    with patch.dict(os.environ, {"DATABASE_PATH": db_file}):
        with patch("main.sync_db_from_cloud_storage") as mock_sync_from, \
             patch("main.sync_db_to_cloud_storage") as mock_sync_to, \
             patch("src.pipeline.Pipeline") as mock_pipeline_cls:

            mock_pipeline = MagicMock()
            mock_summary = MagicMock(fetched=5, new_unseen=4, analyzed=3, alerted=1, errors=0)
            mock_pipeline.run_cycle.return_value = mock_summary
            mock_pipeline.__enter__.return_value = mock_pipeline
            mock_pipeline_cls.return_value = mock_pipeline

            result = main.run_agent_scan()

            assert result["fetched"] == 5
            assert result["new_unseen"] == 4
            assert result["analyzed"] == 3
            assert result["alerted"] == 1
            assert result["errors"] == 0
            assert "duration_seconds" in result
            mock_sync_from.assert_called_once()
            mock_sync_to.assert_called_once()
