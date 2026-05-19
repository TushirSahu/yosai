import numpy as np
from src.steps.drift_detection import detect_drift

def test_no_drift_detected():
    """Test standard normal data where drift should NOT be detected."""
    # Create mock image data with mean brightness around 0.5 (within 0.2 to 0.8)
    mock_data = np.full((10, 3, 224, 224), 0.5)
    
    drift_status = detect_drift(mock_data)
    assert drift_status is False, "Drift was detected incorrectly on normal data."

def test_drift_detected_high_brightness():
    """Test data with high brightness where drift SHOULD be detected."""
    # Create mock image data with very high brightness (mean 0.9)
    mock_data = np.full((10, 3, 224, 224), 0.9)
    
    drift_status = detect_drift(mock_data)
    assert drift_status is True, "Drift was not detected on high-brightness data."

def test_drift_detected_low_brightness():
    """Test data with low brightness where drift SHOULD be detected."""
    # Create mock image data with very low brightness (mean 0.1)
    mock_data = np.full((10, 3, 224, 224), 0.1)
    
    drift_status = detect_drift(mock_data)
    assert drift_status is True, "Drift was not detected on low-brightness data."
