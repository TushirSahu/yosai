import torch
from src.models.anomaly_net import SimpleAnomalyNet

def test_anomaly_net_initialization():
    """Test if the model initializes correctly."""
    model = SimpleAnomalyNet()
    assert isinstance(model, SimpleAnomalyNet)
    
    # Check if expected layers exist
    assert hasattr(model, 'conv1')
    assert hasattr(model, 'fc2')
    
def test_anomaly_net_forward_pass():
    """Test the forward pass with a dummy tensor of correct shape."""
    model = SimpleAnomalyNet()
    model.eval()  # Set to evaluation mode
    
    # Batch size 2, 3 channels (RGB), 224x224 image
    dummy_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    # Check output shape (Batch size 2, 1 output node for binary classification)
    assert output.shape == (2, 1)
    
    # Check if sigmoid activation works (outputs between 0 and 1)
    assert torch.all(output >= 0.0) and torch.all(output <= 1.0)
