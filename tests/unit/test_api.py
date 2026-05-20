"""Unit tests for the serving API."""

import base64
import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image


@pytest.fixture
def mock_model():
    """Create a mock loaded model."""
    mock = MagicMock()
    mock.model_uri = "models:/test-model/1"
    mock.model_version = "1"
    mock.loaded_at = 1234567890.0

    # Mock prediction returning a tensor
    mock_model = MagicMock()
    mock_model.return_value = MagicMock()
    mock_model.return_value.squeeze.return_value.cpu.return_value.numpy.return_value = 0.75
    mock.model = mock_model

    return mock

    


@pytest.fixture
def client(mock_model):
    """Create test client with mocked model."""
    with patch("src.serving.app._load_model", return_value=mock_model):
        with patch("src.serving.app.mlflow"):
            from src.serving.app import app
            yield TestClient(app)


@pytest.fixture
def sample_image_b64():
    """Create a valid base64 encoded test image."""
    img = Image.new("RGB", (100, 100), color="red")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_health_returns_ok(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    def test_liveness_returns_alive(self, client):
        """Test liveness probe."""
        response = client.get("/health/live")
        assert response.status_code == 200
        assert response.json()["status"] == "alive"

    def test_readiness_returns_ready(self, client):
        """Test readiness probe when model is loaded."""
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


class TestPredictEndpoint:
    """Tests for prediction endpoint."""

    def test_predict_success(self, client, sample_image_b64):
        """Test successful prediction."""
        response = client.post(
            "/predict",
            json={"image_b64": sample_image_b64, "threshold": 0.5},
        )
        assert response.status_code == 200
        data = response.json()
        assert "score" in data
        assert "is_anomaly" in data
        assert "model_uri" in data
        assert "model_version" in data
        assert "inference_time_ms" in data
        assert "request_id" in data

    def test_predict_invalid_base64(self, client):
        """Test prediction with invalid base64."""
        response = client.post(
            "/predict",
            json={"image_b64": "invalid-base64!!!", "threshold": 0.5},
        )
        assert response.status_code == 400

    def test_predict_empty_image(self, client):
        """Test prediction with empty image."""
        response = client.post(
            "/predict",
            json={"image_b64": "", "threshold": 0.5},
        )
        assert response.status_code == 422  # Validation error

    def test_predict_invalid_threshold(self, client, sample_image_b64):
        """Test prediction with out-of-range threshold."""
        response = client.post(
            "/predict",
            json={"image_b64": sample_image_b64, "threshold": 1.5},
        )
        assert response.status_code == 422


class TestModelReload:
    """Tests for model reload endpoint."""

    def test_reload_success(self, client):
        """Test successful model reload."""
        response = client.post("/model/reload")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert "new_model_uri" in data


class TestMetrics:
    """Tests for metrics endpoint."""

    def test_metrics_returns_prometheus(self, client):
        """Test metrics endpoint returns Prometheus format."""
        response = client.get("/metrics")
        assert response.status_code == 200
        assert "text/plain" in response.headers["content-type"]
        # Should contain some metric names
        content = response.text
        assert "inference_latency_seconds" in content or "predictions_total" in content


class TestRoot:
    """Tests for root endpoint."""

    def test_root_returns_info(self, client):
        """Test root endpoint returns app info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "app" in data
        assert "version" in data
        assert "environment" in data