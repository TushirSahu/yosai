import numpy as np

def detect_drift(X_train: np.ndarray) -> bool:
    """Mock drift detection step. In production, use Evidently profiles on embeddings."""
    # Simulated drift check: Average brightness should be within expected bounds
    mean_brightness = np.mean(X_train)
    drift_detected = not (0.2 < mean_brightness < 0.8)
    
    if drift_detected:
        print(f"⚠️ Data Drift Detected! Mean brightness: {mean_brightness}")
    else:
        print("✅ No data drift detected.")
    
    return drift_detected
