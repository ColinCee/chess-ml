import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ChessboardFENNet(nn.Module):
    def __init__(self):
        super(ChessboardFENNet, self).__init__()
        # Define your network architecture here
        # Example: Convolutional layers followed by fully connected layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        # ... additional layers ...
        self.fc1 = nn.Linear(
            64 * 56 * 56, 1000
        )  # Adjust the input features accordingly
        self.fc2 = nn.Linear(1000, 832)  # 64 squares * 13 classes

    def forward(self, x):
        # Define the forward pass
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # ... additional layers ...
        x = x.view(x.size(0), -1)  # Flatten the output
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
