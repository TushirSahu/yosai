import os

import mlflow

from src.pipelines.continuous_training import edge_continuous_training_pipeline

if __name__ == "__main__":
    # MLflow configuration
    # - Default: local file store (works with no Docker)
    # - Optional: point to local MLflow server (e.g. http://127.0.0.1:5001)
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
    mlflow.set_tracking_uri(tracking_uri)

    # If you're using the local docker-compose (MinIO as S3), MLflow needs these.
    # Keeping defaults here makes the demo "just work" without cloud credentials.
    if tracking_uri.startswith("http"):
        os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://127.0.0.1:9000"))
        os.environ.setdefault("AWS_ACCESS_KEY_ID", os.getenv("AWS_ACCESS_KEY_ID", "minioadmin"))
        os.environ.setdefault("AWS_SECRET_ACCESS_KEY", os.getenv("AWS_SECRET_ACCESS_KEY", "minioadmin"))
        os.environ.setdefault("AWS_DEFAULT_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    
    # Create experiment if it doesn't exist
    experiment_name = "yosai-edge-mlops"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if not experiment:
        mlflow.create_experiment(experiment_name)
    
    mlflow.set_experiment(experiment_name)
    
    # Execute the MLOps pipeline with automatic MLflow run tracking
    with mlflow.start_run():
        mlflow.log_param("project_name", "Yosai (要塞)")
        mlflow.log_param("target_market", "Japan Manufacturing")
        mlflow.log_param("mlflow_tracking_uri", tracking_uri)
        edge_continuous_training_pipeline()
        
    if tracking_uri.startswith("file:"):
        print("\n✅ MLflow tracking data saved to ./mlruns/")
        print("📊 To view results, run: mlflow ui")
        print("   Then open http://127.0.0.1:5001")
    else:
        print(f"\n✅ MLflow tracking server: {tracking_uri}")
        print("   Open it in your browser to view runs.")