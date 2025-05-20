import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNetwork(nn.Module):
    """
    Architettura semplificata per MiniChess 5×5:
    - Input: (batch_size, 3, 5, 5)
      canali:
        0 = presenza pezzo bianco (1 se c'è bianco, 0 altrimenti)
        1 = presenza pezzo nero (1 se c'è nero, 0 altrimenti)
        2 = turno (costante +1 o -1)
    - 2 strati conv (32 e 64 filtri, kernel 3×3, padding=1) + BatchNorm
    - 1 fully‐connected lineare con tanh in uscita (scalare in [-1,1])
    """
    def __init__(self, hidden_channels: int = 32):
        super().__init__()
        # Primo strato conv: da 3 canali a hidden_channels
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=hidden_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(hidden_channels)

        # Secondo strato conv: da hidden_channels a hidden_channels*2
        self.conv2 = nn.Conv2d(in_channels=hidden_channels,
                               out_channels=hidden_channels * 2,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(hidden_channels * 2)

        # Fully‐connected finale
        self.fc = nn.Linear((hidden_channels * 2) * 5 * 5, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 3, 5, 5)
        ritorna: (B, 1) con tanh
        """
        h = F.relu(self.bn1(self.conv1(x)))    # (B, hidden_channels, 5, 5)
        h = F.relu(self.bn2(self.conv2(h)))    # (B, hidden_channels*2, 5, 5)
        h = h.view(h.size(0), -1)              # (B, (hidden_channels*2)*5*5)
        v = torch.tanh(self.fc(h))             # (B, 1)
        return v
