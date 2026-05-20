"""Production configuration management using pydantic-settings."""

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class MLflowSettings(BaseSettings):
    """MLflow-related configuration."""

    tracking_uri: str = Field(
        default="file:./mlruns",
        description="MLflow tracking server URI"
    )
    s3_endpoint_url: str = Field(
        default="http://127.0.0.1:9000",
        description="S3/MinIO endpoint for MLflow artifacts"
    )
    aws_access_key_id: str = Field(default="minioadmin")
    aws_secret_access_key: str = Field(default="minioadmin")
    aws_default_region: str = Field(default="us-east-1")
    experiment_name: str = Field(default="yosai-edge-mlops")
    model_name: str = Field(default="yosai-anomaly-detector")


class APISettings(BaseSettings):
    """API server configuration."""

    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)
    workers: int = Field(default=1)
    reload: bool = Field(default=False)
    log_level: str = Field(default="info")

    # Security
    api_keys: list[str] = Field(
        default_factory=list,
        description="API keys for authentication"
    )
    rate_limit_per_minute: int = Field(
        default=60,
        description="Rate limit per IP per minute"
    )

    # Model settings
    model_warmup: bool = Field(
        default=True,
        description="Load model on startup"
    )
    max_image_size_mb: int = Field(
        default=10,
        description="Maximum image file size in MB"
    )
    allowed_image_formats: list[str] = Field(
        default_factory=lambda: ["PNG", "JPEG", "JPG"]
    )


class Settings(BaseSettings):
    """Main application settings combining all configurations."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
        extra="ignore"
    )

    app_name: str = Field(default="Yōsai Inference API")
    app_version: str = Field(default="0.1.0")
    environment: str = Field(default="development")

    mlflow: MLflowSettings = Field(
        default_factory=MLflowSettings
    )
    api: APISettings = Field(
        default_factory=APISettings
    )


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance.

    Use this instead of instantiating Settings directly.
    """
    return Settings()