import torch
import torch.nn as nn
import torch.nn.functional as F
#from games.gardner.mcts_pt import INPUT_CHANNELS

INPUT_CHANNELS = 13  # 6 bianchi + 6 neri + turno = 13

class ValueNetwork(nn.Module):
    """
    CNN molto leggera per Gardner MiniChess 5×5.

    Args
    ----
    hidden_channels : int
        Numero di filtri del primo conv-layer.
    output_dim : int
        Dimensione del vettore di valore in uscita
        (1 = scalar; 2 = [v_white, v_black]).
    """
    def __init__(self, hidden_channels: int = 32, output_dim: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(INPUT_CHANNELS, hidden_channels, 3, padding=1)
        self.bn1   = nn.BatchNorm2d(hidden_channels)

        self.conv2 = nn.Conv2d(hidden_channels,
                               hidden_channels * 2,
                               3, padding=1)
        self.bn2   = nn.BatchNorm2d(hidden_channels * 2)

        self.fc = nn.Linear((hidden_channels * 2) * 5 * 5, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.bn1(self.conv1(x)))
        h = F.relu(self.bn2(self.conv2(h)))
        h = h.view(h.size(0), -1)
        v = torch.tanh(self.fc(h))          # (B, output_dim)
        return v
