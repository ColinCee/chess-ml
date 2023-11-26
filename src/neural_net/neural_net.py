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
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.linear_output = nn.Sequential(
            nn.Linear(131072, 512),
            nn.Dropout(p=0.5),
            nn.Linear(512, 128),
            nn.Dropout(p=0.5),
            nn.Linear(128, 13)
        )

    def forward(self, x):
        # Define the forward pass
        x = self.cnn_stack(x)
        x = self.flatten(x)
        x = self.linear_output(x)
        return x
