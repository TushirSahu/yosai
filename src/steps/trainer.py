import mlflow
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from src.models.anomaly_net import SimpleAnomalyNet


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> torch.nn.Module:
    """Trains the CNN with MLflow experiment tracking."""
    model = SimpleAnomalyNet()
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", factor=0.5, patience=5
    )

    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32)

    mlflow.log_param("epochs", 100)
    mlflow.log_param("learning_rate", 0.0005)
    mlflow.log_param("optimizer", "Adam with ReduceLROnPlateau")
    mlflow.log_param("model_architecture", "SimpleAnomalyNet_CNN_v2_with_BatchNorm")
    mlflow.log_param("batch_norm", True)
    mlflow.log_param("dropout", 0.3)

    print("🚀 Training model...")
    for epoch in range(100):
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        mlflow.log_metric("training_loss", loss.item(), step=epoch)
        if (epoch + 1) % 20 == 0:
            print(f"   Epoch {epoch+1}/100 - Loss: {loss.item():.4f}")

    return model
