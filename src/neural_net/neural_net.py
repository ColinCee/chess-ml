import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader


class ChessboardFENNet(nn.Module):
    def __init__(self):
        super(ChessboardFENNet, self).__init__()
        self.flatten = nn.Flatten()
        self.cnn_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=5, padding=1),
            nn.ReLU(),
        )
        self.linear_output = nn.Linear(238144, 8 * 8 * 13)

    def forward(self, x):
        # Define the forward pass
        x = self.cnn_stack(x)
        x = self.flatten(x)
        x = self.linear_output(x)
        return x
