import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, HIDDEN_CHANNELS

class ValueNetwork(nn.Module):
    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS, output_dim: int = 2):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(hidden_channels * 4)
        self.fc = nn.Linear((hidden_channels * 4) * 5 * 5, output_dim)
        self.activation = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.activation(self.bn1(self.conv1(x)))
        h = self.activation(self.bn2(self.conv2(h)))
        h = self.activation(self.bn3(self.conv3(h)))
        h = h.view(h.size(0), -1)
        v = self.fc(h)
        return v
