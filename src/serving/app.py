from __future__ import annotations

import base64
import io
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

import mlflow
import numpy as np
import torch
from fastapi import (
    FastAPI,
    HTTPException,
    Request,
    Response,
    Security,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest
from pydantic import BaseModel, Field, field_validator
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from tenacity import retry, stop_after_attempt, wait_exponential

# Structured logging
import structlog
from structlog.stdlib import ProcessorFormatter
from PIL import Image

from src.serving.config import Settings, get_settings

# Initialize structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)
settings = get_settings()

# Prometheus metrics
REQUEST_LATENCY = Histogram(
    "inference_latency_seconds",
    "Model inference time in seconds",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
PREDICTION_COUNTER = Counter(
    "predictions_total",
    "Total predictions made",
    ["status"],
)
DRIFT_ALERTS = Counter(
    "drift_alerts_total",
    "Total drift detection alerts",
)
MODEL_LOAD_SUCCESS = Counter(
    "model_load_success_total",
    "Total successful model loads",
)
MODEL_LOAD_FAILURE = Counter(
    "model_load_failure_total",
    "Total failed model loads",
)
REQUEST_COUNTER = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)


class PredictRequest(BaseModel):
    """Prediction request schema with validation."""

    image_b64: str = Field(..., description="Base64-encoded image bytes (PNG/JPG).")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Decision threshold.")

    @field_validator("image_b64")
    @classmethod
    def validate_image_not_empty(cls, v: str) -> str:
        if not v or len(v) < 10:
            raise ValueError("image_b64 must be a valid base64 string")
        return v


class PredictResponse(BaseModel):
    """Prediction response schema."""

    score: float = Field(..., description="Anomaly score (0-1)")
    is_anomaly: bool = Field(..., description="Whether the image is anomalous")
    threshold: float = Field(..., description="Threshold used for prediction")
    model_uri: str = Field(..., description="URI of the loaded model")
    model_version: str = Field(..., description="Model version from MLflow")
    inference_time_ms: float = Field(..., description="Inference time in milliseconds")
    request_id: str = Field(..., description="Unique request identifier")


@dataclass
class LoadedModel:
    """Container for loaded model and metadata."""

    model: Any
    model_uri: str
    model_version: str
    loaded_at: float


# Rate limiter
limiter = Limiter(key_func=get_remote_address)


def _decode_image(image_b64: str, request_id: str) -> Image.Image:
    """Decode and validate base64 image."""
    try:
        raw = base64.b64decode(image_b64)

        # Check size limit
        size_mb = len(raw) / (1024 * 1024)
        if size_mb > settings.api.max_image_size_mb:
            logger.warning(
                "image_too_large",
                request_id=request_id,
                size_mb=round(size_mb, 2),
                limit_mb=settings.api.max_image_size_mb,
            )
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Image exceeds maximum size of {settings.api.max_image_size_mb}MB",
            )

        img = Image.open(io.BytesIO(raw))

        # Validate format
        if img.format not in settings.api.allowed_image_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported image format. Allowed: {settings.api.allowed_image_formats}",
            )

        return img.convert("RGB")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("image_decode_failed", request_id=request_id, error=str(exc))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid image_b64: {exc}",
        )


def _preprocess(img: Image.Image) -> torch.Tensor:
    """Preprocess image for model input."""
    img = img.resize((224, 224))
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr).unsqueeze(0)
    return x


def _try_resolve_model_uri_from_registry(model_name: str) -> Optional[tuple[str, str]]:
    """Try to resolve model from MLflow registry with version."""
    try:
        from mlflow.tracking import MlflowClient

        client = MlflowClient()
        for stage in ("Production", "Staging", "None"):
            try:
                latest = client.get_latest_versions(model_name, stages=[stage])
            except Exception:
                latest = []
            if latest:
                return (f"models:/{model_name}/{stage}", latest[0].version)

        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            newest = max(versions, key=lambda v: int(v.version))
            return (f"models:/{model_name}/{newest.version}", newest.version)

        return None
    except Exception:
        return None


def _try_resolve_model_uri_from_latest_run(
    experiment_name: str,
) -> Optional[tuple[str, str]]:
    """Try to resolve model from latest run."""
    try:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if not exp:
            return None

        df = mlflow.search_runs(
            experiment_ids=[exp.experiment_id],
            order_by=["start_time DESC"],
            max_results=20,
        )
        if df.empty:
            return None

        for _, row in df.iterrows():
            run_id = row["run_id"]
            return (f"runs:/{run_id}/model", run_id[:8])

        return None
    except Exception:
        return None


def _resolve_model_uri() -> tuple[str, str]:
    """Resolve model URI and version."""
    model_name = settings.mlflow.model_name
    experiment_name = settings.mlflow.experiment_name

    result = _try_resolve_model_uri_from_registry(model_name)
    if result:
        return result

    result = _try_resolve_model_uri_from_latest_run(experiment_name)
    if result:
        return result

    raise RuntimeError(
        "Could not resolve a model URI. Run training once (run_pipeline.py) "
        "so a model is logged/registered, then try again."
    )


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    reraise=True,
)
def _load_model_with_retry() -> LoadedModel:
    """Load model with retry logic for resilience."""
    model_uri, version = _resolve_model_uri()
    logger.info("loading_model", model_uri=model_uri, version=version)

    try:
        model = mlflow.pytorch.load_model(model_uri)
        MODEL_LOAD_SUCCESS.inc()
    except Exception as e:
        logger.warning("pytorch_load_failed", error=str(e), trying="pyfunc")
        try:
            model = mlflow.pyfunc.load_model(model_uri)
            MODEL_LOAD_SUCCESS.inc()
        except Exception as e2:
            MODEL_LOAD_FAILURE.inc()
            logger.error("model_load_failed", error=str(e2))
            raise

    return LoadedModel(
        model=model,
        model_uri=model_uri,
        model_version=version,
        loaded_at=time.time(),
    )


def _load_model() -> LoadedModel:
    """Wrapper that handles retries and metrics."""
    try:
        return _load_model_with_retry()
    except Exception as e:
        MODEL_LOAD_FAILURE.inc()
        raise RuntimeError(f"Failed to load model after retries: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    # Startup
    logger.info(
        "application_starting",
        app_name=settings.app_name,
        version=settings.app_version,
        environment=settings.environment,
    )

    # Configure MLflow
    mlflow.set_tracking_uri(settings.mlflow.tracking_uri)
    if settings.mlflow.tracking_uri.startswith("http"):
        os.environ.setdefault(
            "MLFLOW_S3_ENDPOINT_URL", settings.mlflow.s3_endpoint_url
        )
        os.environ.setdefault("AWS_ACCESS_KEY_ID", settings.mlflow.aws_access_key_id)
        os.environ.setdefault(
            "AWS_SECRET_ACCESS_KEY", settings.mlflow.aws_secret_access_key
        )
        os.environ.setdefault(
            "AWS_DEFAULT_REGION", settings.mlflow.aws_default_region
        )

    # Load model if configured
    if settings.api.model_warmup:
        try:
            app.state.model = _load_model()
            logger.info(
                "model_loaded",
                model_uri=app.state.model.model_uri,
                version=app.state.model.model_version,
            )
        except Exception as e:
            logger.error("model_warmup_failed", error=str(e))
            app.state.model = None
    else:
        app.state.model = None

    yield

    # Shutdown
    logger.info("application_shutting_down")


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    lifespan=lifespan,
    docs_url="/docs" if settings.environment != "production" else None,
    redoc_url="/redoc" if settings.environment != "production" else None,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.environment == "development" else [],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception handlers
@app.exception_handler(RateLimitExceeded)
async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
    """Handle rate limit exceeded errors."""
    logger.warning("rate_limit_exceeded", ip=get_remote_address(request))
    return JSONResponse(
        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
        content={"detail": "Rate limit exceeded. Please try again later."},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions."""
    logger.warning(
        "http_error",
        request_id=getattr(request.state, "request_id", "unknown"),
        status_code=exc.status_code,
        detail=exc.detail,
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unhandled exceptions."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.exception("unhandled_error", request_id=request_id, error=str(exc))
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics and add request ID."""
    request.state.request_id = str(uuid.uuid4())

    start_time = time.time()
    response = await call_next(request)

    duration = time.time() - start_time
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNTER.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code,
    ).inc()

    logger.info(
        "request_completed",
        request_id=request.state.request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )

    return response


# Health check endpoints
@app.get("/health")
def health() -> dict:
    """Basic liveness check."""
    return {"status": "ok", "app": settings.app_name}


@app.get("/health/live")
def liveness() -> dict:
    """Liveness probe - is the app running?"""
    return {"status": "alive"}


@app.get("/health/ready")
def readiness() -> dict:
    """Readiness probe - is the app ready to serve traffic?"""
    if app.state.model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    # Check MLflow connectivity
    try:
        mlflow.get_experiment_by_name(settings.mlflow.experiment_name)
    except Exception as e:
        logger.warning("mlflow_health_check_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLflow not available",
        )

    return {
        "status": "ready",
        "model_uri": app.state.model.model_uri,
        "model_version": app.state.model.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
@limiter.limit(f"{settings.api.rate_limit_per_minute}/minute")
async def predict(request: Request, req: PredictRequest) -> PredictResponse:
    """Make anomaly prediction on an image."""
    request_id = request.state.request_id

    if app.state.model is None:
        PREDICTION_COUNTER.labels(status="error_no_model").inc()
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model not loaded",
        )

    start_time = time.time()
    try:
        img = _decode_image(req.image_b64, request_id)
        x = _preprocess(img)

        # Support both torch module and pyfunc
        with torch.no_grad():
            try:
                y = app.state.model.model(x)
                score = float(y.squeeze().cpu().numpy().item())
            except Exception:
                y2 = app.state.model.model.predict(x.numpy())
                score = float(np.asarray(y2).reshape(-1)[0])

        is_anomaly = score >= req.threshold
        inference_time = (time.time() - start_time) * 1000

        PREDICTION_COUNTER.labels(status="success").inc()

        return PredictResponse(
            score=score,
            is_anomaly=is_anomaly,
            threshold=req.threshold,
            model_uri=app.state.model.model_uri,
            model_version=app.state.model.model_version,
            inference_time_ms=round(inference_time, 2),
            request_id=request_id,
        )

    except HTTPException:
        raise
    except Exception:
        PREDICTION_COUNTER.labels(status="error").inc()
        raise


@app.post("/model/reload")
async def reload_model(request: Request) -> dict:
    """Reload the model from MLflow."""
    request_id = request.state.request_id
    logger.info("model_reload_requested", request_id=request_id)

    try:
        new_model = _load_model()
        old_uri = (
            app.state.model.model_uri if app.state.model else "none"
        )
        app.state.model = new_model

        logger.info(
            "model_reloaded",
            request_id=request_id,
            old_uri=old_uri,
            new_uri=new_model.model_uri,
        )

        return {
            "status": "ok",
            "old_model_uri": old_uri,
            "new_model_uri": new_model.model_uri,
            "new_model_version": new_model.model_version,
        }

    except Exception as e:
        logger.error("model_reload_failed", request_id=request_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to reload model: {e}",
        )


@app.get("/metrics")
def metrics() -> Response:
    """Prometheus metrics endpoint."""
    return Response(content=generate_latest(), media_type="text/plain")


@app.get("/")
def root() -> dict:
    """Root endpoint with API info."""
    return {
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "docs": "/docs" if settings.environment != "production" else None,
    }