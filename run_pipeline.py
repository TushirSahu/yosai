import mlflow
from src.pipelines.continuous_training import edge_continuous_training_pipeline

if __name__ == "__main__":
    # Set MLflow tracking URI to local directory
    mlflow.set_tracking_uri("file:./mlruns")
    
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
        edge_continuous_training_pipeline()
        
    print("\n✅ MLflow tracking data saved to ./mlruns/")
    print("📊 To view results, run: mlflow ui")