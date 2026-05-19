import torch.nn as nn
import torch
import torch.nn.functional as F

class SimpleAnomalyNet(nn.Module):
    """A lightweight CNN model designed for edge deployment on Jetson/Triton."""
    def __init__(self):
        super(SimpleAnomalyNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1) # Binary classification: Normal/Anomaly

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 32 * 56 * 56)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
