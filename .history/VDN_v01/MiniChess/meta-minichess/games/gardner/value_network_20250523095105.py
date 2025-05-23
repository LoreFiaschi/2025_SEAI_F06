import torch
import torch.nn as nn
import torch.nn.functional as F
from config import INPUT_CHANNELS, HIDDEN_CHANNELS

class ValueNetwork(nn.Module):
    """
    hidden_channels : int
        Numero di filtri del primo conv-layer.
    output_dim : int
        Dimensione del vettore di valore in uscita
        (1 = scalar; 2 = [v_white, v_black]).
    """
    def __init__(self, hidden_channels: int = HIDDEN_CHANNELS, output_dim: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, hidden_channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(hidden_channels)
        self.conv2 = nn.Conv2d(hidden_channels,
                               hidden_channels * 2,
                               3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(hidden_channels * 2)

        self.fc = nn.Linear((hidden_channels * 2) * 5 * 5, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = h.view(h.size(0), -1)
        v = torch.tanh(self.fc(h))          # (B, output_dim)
        return v
