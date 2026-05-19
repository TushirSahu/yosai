import mlflow
import torch
import torch.nn as nn


def register_if_improved(
    model: nn.Module, accuracy: float, min_threshold: float = 0.55
) -> str:
    """Checks threshold and promotes the model if conditions are met.

    For production: min_threshold should be 0.95+ (99.5% manufacturing precision)
    For demo/synthetic data: min_threshold is 0.55 (above random 0.50)
    """
    if accuracy > min_threshold:
        print(
            f"🏆 Model passed threshold ({accuracy:.2%}). Promoting to Model Registry!"
        )

        # Log model using MLflow's PyTorch flavor
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name="yosai-anomaly-detector",
            pip_requirements=["torch>=2.0.0", "torchvision>=0.15.0"],
            tags={
                "project": "yosai",
                "target_market": "japan_manufacturing",
                "model_type": "cnn_binary_classifier",
            },
        )

        print(f"✅ Model successfully registered in MLflow Model Registry!")
        mlflow.log_metric("model_status", 1.0)  # 1 = registered
        mlflow.log_param("registration_threshold", min_threshold)
        return "REGISTERED"
    else:
        print(
            f"❌ Model failed threshold ({accuracy:.2%}, required: {min_threshold:.2%}). Dropping."
        )
        mlflow.log_metric("model_status", 0.0)  # 0 = rejected
        return "REJECTED"
