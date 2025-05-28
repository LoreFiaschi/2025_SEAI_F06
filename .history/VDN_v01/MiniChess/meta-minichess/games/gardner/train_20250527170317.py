import os
import sys
import math
import random
import logging
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────────────────
# Path‑hack: aggiunge due livelli sopra (la cartella che contiene "games/")
this_dir: str = os.path.dirname(__file__)                              # .../games/gardner
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))  # .../meta‑minichess
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

# ─── CONFIG.PY ────────────────────────────────────────────────────────────
from config import HIDDEN_CHANNELS, PIECE_VALUES, STEP_COST, ALPHA
from config import num_cycles, arena_games, games_per_cycle
from config import max_buffer_size, iterations_MCTS
from config import learning_rate, weight_decay, batch_size, num_epochs
from config import DEVICE
# ────────────────────────────────────────────────────────────────────────────

# ─── setup logging: write only to file, no console output ────────────────
log_path = os.path.join(this_dir, 'training.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
# Remove existing handlers
for h in logger.handlers[:]:
    logger.removeHandler(h)
# File handler
file_handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Contatore diagnostico per la distribuzione dei risultati finali
z_counter: Counter = Counter()

# ─── Gestione dei checkpoint ───────────────────────────────────────────────
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_ckpt(net: torch.nn.Module, name: str) -> None:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    torch.save(net.state_dict(), path)
    logger.info(f"Checkpoint saved: {path}")


def load_ckpt(name: str, device: str = DEVICE) -> torch.nn.Module:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    logger.info(f"Checkpoint loaded: {path}")
    return net

# ─── Dataset con target (z_white, z_black) ─────────────────────────────────
class ChessValueDataset(Dataset):
    """Ogni elemento: (tensor_stato, torch.tensor([z_w, z_b]))"""

    def __init__(self, buffer: List[Tuple[torch.Tensor, float, float]]):
        self.items = buffer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        st_t, z_w, z_b = self.items[idx]
        return st_t.float(), torch.tensor([z_w, z_b], dtype=torch.float32, device=DEVICE)


def value_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


def compute_white_black_rewards(
    phi_prev: float,
    phi_next: float,
    mover: int  # +1 Bianco, -1 Nero
) -> Tuple[float, float]:
    delta = phi_next - phi_prev
    if mover == 1:
        r_w =  ALPHA * delta - STEP_COST
        r_b = -ALPHA * delta
    else:
        r_w =  ALPHA * delta
        r_b = -ALPHA * delta - STEP_COST
    return r_w, r_b


def compute_material_potential(board: Tuple[Tuple[int, ...], ...]) -> float:
    total = 0.0
    for row in board:
        for piece in row:
            if piece == 0 or abs(piece) == 60000:
                continue
            val = PIECE_VALUES.get(abs(piece), 1.0)
            total += math.copysign(val, piece)
    return total


def self_play_game(
    mcts_white: MCTS,
    mcts_black: MCTS,
) -> List[Tuple[torch.Tensor, float, float]]:
    transitions: List[Tuple[torch.Tensor, float, float]] = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())

    while not state.is_terminal():
        mover = state.current_player()
        mcts = mcts_white if mover == 1 else mcts_black
        move = mcts.search(state, temperature=)
        next_state = state.next_state(move)

        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        transitions.append((encode_state_as_tensor(state), r_w, r_b))

        state, phi_prev = next_state, phi_next

    z_w = state.result()
    z_b = -z_w
    if z_w == 1e-4:
        z_b = z_w

    data: List[Tuple[torch.Tensor, float, float]] = []
    cum_w = cum_b = 0.0
    for st_t, r_w, r_b in reversed(transitions):
        cum_w += r_w
        cum_b += r_b
        data.append((st_t, z_w + cum_w, z_b + cum_b))
    return data


def play_det(
    net_white: torch.nn.Module,
    net_black: torch.nn.Module,
    mcts_w: MCTS,
    mcts_b: MCTS,
) -> int:
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mcts = mcts_w if state.current_player() == 1 else mcts_b
        move = mcts.search(state, temperature=0.0)
        state = state.next_state(move)
    return state.result()


def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games) -> float:
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS)
    mcts_best = MCTS(best_net, iterations_MCTS)
    half = games // 2

    for _ in range(half):
        res = play_det(current_net, best_net, mcts_curr, mcts_best)
        if res == 1:
            wins += 1
        elif res == 1e-4:
            draws += 1
    wins_white = wins

    for _ in range(games - half):
        res = play_det(best_net, current_net, mcts_best, mcts_curr)
        if res == -1:
            wins += 1
        elif res == 1e-4:
            draws += 1

    logger.info(f"Arena: wins_white={wins_white}, wins_black={wins - wins_white}, draws={draws}")
    return wins / (games - draws)


def train_value_net(
    value_net: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = num_epochs,
    device: torch.device = DEVICE,
) -> float:
    value_net.to(device)
    value_net.train()
    scaler = torch.GradScaler()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for states_batch, targets_batch in train_loader:
            states_batch = states_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            with torch.autocast(device_type=DEVICE, dtype=torch.float16):
                preds = value_net(states_batch)
                loss = value_loss_fn(preds, targets_batch)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * states_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.5f}")
    return avg_loss

if __name__ == "__main__":
    best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    save_ckpt(best_net, "best_0")

    for cycle in range(1, num_cycles + 1):
        logger.info(f"===== GENERAZIONE {cycle} =====")
        current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
        current_net.load_state_dict(best_net.state_dict())
        optimizer = torch.optim.AdamW(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        buffer: list[tuple[torch.Tensor, float, float]] = []
        mcts_curr = MCTS(current_net, iterations_MCTS)
        mcts_best = MCTS(best_net, iterations_MCTS)

        for _ in range(games_per_cycle):
            buffer.extend(self_play_game(mcts_curr, mcts_best))
            buffer.extend(self_play_game(mcts_best, mcts_curr))

        if len(buffer) > max_buffer_size:
            buffer = random.sample(buffer, max_buffer_size)

        train_loader = DataLoader(ChessValueDataset(buffer), batch_size=batch_size, shuffle=True)
        train_value_net(current_net, train_loader, optimizer, num_epochs=num_epochs, device=DEVICE)

        wr = arena(current_net, best_net, games=arena_games)
        logger.info(f"Win-rate vs best: {wr*100:.1f}%")
        if wr > 0.55:
            best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
        else:
            logger.info("Rete non promossa, si continua…")
