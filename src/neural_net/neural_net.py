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
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.linear_output = nn.Sequential(
            nn.Dropout2d(p=0.5),
            nn.Linear(64 * 32 * 32, 13),
        )

    def forward(self, x):
        # Define the forward pass
        x = self.cnn_stack(x)
        x = self.flatten(x)
        x = self.linear_output(x)
        return x
