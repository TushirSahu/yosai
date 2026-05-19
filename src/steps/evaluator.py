import mlflow
import torch
import torch.nn as nn
import numpy as np

def evaluate_model(model: nn.Module, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Evaluates the model and logs metrics."""
    model.eval()
    criterion = nn.BCELoss()
    
    X_tensor = torch.tensor(X_test)
    y_tensor = torch.tensor(y_test)
    
    with torch.no_grad():
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor).item()
        
        preds = (outputs > 0.5).float()
        acc = (preds == y_tensor).float().mean().item()
        
    mlflow.log_metric("test_loss", loss)
    mlflow.log_metric("test_accuracy", acc)
    
    print(f"📊 Evaluation complete - Test Accuracy: {acc*100:.2f}%")
    return float(acc)
