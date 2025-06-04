import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, HIDDEN_CHANNELS

class ValueNetwork(nn.Module):
    """
    Convolutional Value Network:
    - Input: tensor [B, INPUT_CHANNELS, 5, 5]
    - Three Conv→BN→LeakyReLU blocks
    - Flatten and Linear to scalar output per position
    """
    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS, output_dim: int = 1):
        super().__init__()
        # Block 1: conv 3×3, BN, activation
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, hidden_channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_channels)
        # Block 2: double channels
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels * 2, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_channels * 2)
        # Block 3: quadruple channels
        self.conv3 = nn.Conv2d(hidden_channels * 2, hidden_channels * 4, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(hidden_channels * 4)
        # Final fully-connected head
        self.fc    = nn.Linear((hidden_channels * 4) * 5 * 5, output_dim)
        self.act   = nn.LeakyReLU(negative_slope=0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Conv block 1
        h = self.act(self.bn1(self.conv1(x)))
        # Conv block 2
        h = self.act(self.bn2(self.conv2(h)))
        # Conv block 3
        h = self.act(self.bn3(self.conv3(h)))
        # Flatten spatial dims
        h = h.view(h.size(0), -1)
        # Linear → scalar(s)
        v = self.fc(h)

        return v.squeeze(-1)  # result shape [B]        return v.squeeze(-1)  # result shape [B]
