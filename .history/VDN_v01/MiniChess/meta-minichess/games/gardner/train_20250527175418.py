import os
import sys
import math
import random
import logging
from collections import Counter
from typing import List, Tuple
import torch
from torch import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.amp import GradScaler, autocast

# Limit internal PyTorch threading to avoid oversubscription
torch.set_num_threads(2)
torch.set_num_interop_threads(2)

# ────────────────────────────────────────────────────────────
this_dir: str = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

from config import (
    HIDDEN_CHANNELS, PIECE_VALUES, STEP_COST, ALPHA,
    num_cycles, arena_games, games_per_cycle,
    max_buffer_size, iterations_MCTS,
    learning_rate, weight_decay, batch_size, num_epochs,
    DEVICE
)

# setup logging
log_path = os.path.join(this_dir, 'training.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
file_handler = logging.FileHandler(log_path, mode='a')
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

z_counter: Counter = Counter()

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

class ChessValueDataset(Dataset):
    def __init__(self, buffer: List[Tuple[torch.Tensor, float, float]]):
        self.items = buffer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        st_t, z_w, z_b = self.items[idx]
        return st_t.float(), torch.tensor([z_w, z_b], dtype=torch.float32, device=DEVICE)

def value_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)

def compute_white_black_rewards(phi_prev: float, phi_next: float, mover: int) -> Tuple[float, float]:
    delta = phi_next - phi_prev
    if mover == 1:
        return ALPHA * delta - STEP_COST, -ALPHA * delta
    else:
        return ALPHA * delta, -ALPHA * delta - STEP_COST

def compute_material_potential(board: Tuple[Tuple[int, ...], ...]) -> float:
    total = 0.0
    for row in board:
        for piece in row:
            if piece == 0 or abs(piece) == 60000:
                continue
            val = PIECE_VALUES.get(abs(piece), 1.0)
            total += math.copysign(val, piece)
    return total

def self_play_wrapper(args):
    net_w, net_b, seed = args
    random.seed(seed)
    torch.manual_seed(seed)
    mcts_w = MCTS(net_w, iterations_MCTS, rng=random.Random(seed))
    mcts_b = MCTS(net_b, iterations_MCTS, rng=random.Random(seed+1))
    return self_play_game(mcts_w, mcts_b)

def self_play_game(mcts_white: MCTS, mcts_black: MCTS) -> List[Tuple[torch.Tensor, float, float]]:
    transitions = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())
    for i in range(1000):
        if state.is_terminal():
            break
        mover = state.current_player()
        mcts = mcts_white if mover == 1 else mcts_black
        tau = 1.0 if i < 10 else 0.0
        move = mcts.search(state, temperature=tau)
        next_state = state.next_state(move)
        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        transitions.append((encode_state_as_tensor(state), r_w, r_b))
        state, phi_prev = next_state, phi_next
    logger.info(f"game ended after {state.turns} turns, result: {state.result()}")
    z_w = state.result()
    z_b = -z_w if z_w != 1e-4 else z_w
    data = []
    cum_w = cum_b = 0.0
    for st_t, r_w, r_b in reversed(transitions):
        cum_w += r_w
        cum_b += r_b
        data.append((st_t, z_w + cum_w, z_b + cum_b))
    return data

def play_det(mcts_w: MCTS, mcts_b: MCTS) -> int:
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mcts = mcts_w if state.current_player() == 1 else mcts_b
        move = mcts.search(state, temperature=0.0)
        state = state.next_state(move)
    return state.result()

def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games) -> float:
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS, rng=random.Random(1))
    mcts_best = MCTS(best_net, iterations_MCTS, rng=random.Random(2))
    for _ in range(games // 2):
        res = play_det(mcts_curr, mcts_best)
        if res == 1: wins += 1
        elif res == 1e-4: draws += 1
    for _ in range(games - games // 2):
        res = play_det(mcts_best, mcts_curr)
        if res == -1: wins += 1
        elif res == 1e-4: draws += 1
    logger.info(f"Arena: wins_white={wins}, draws={draws}")
    return wins / (games - draws)

def train_value_net(value_net: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, num_epochs: int = num_epochs, device: torch.device = DEVICE) -> float:
    value_net.to(device)
    value_net.train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for states_batch, targets_batch in train_loader:
            states_batch = states_batch.to(device, non_blocking=True)
            targets_batch = targets_batch.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast():
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
    torch.manual_seed(0)
    random.seed(0)
    best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    save_ckpt(best_net, "best_0")
    for cycle in range(1, num_cycles + 1):
        torch.manual_seed(cycle)
        random.seed(cycle)
        logger.info(f"===== GENERAZIONE {cycle} =====")
        current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
        current_net.load_state_dict(best_net.state_dict())
        optimizer = torch.optim.AdamW(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        seeds = [cycle*100 + i for i in range(games_per_cycle)]
        args = []
        for i, s in enumerate(seeds):
            if i % 2 == 0:
                args.append((current_net, best_net, s))
            else:
                args.append((best_net, current_net, s))
        with mp.Pool(processes=2) as pool:
            results = pool.map(self_play_wrapper, args)
        buffer = [t for sub in results for t in sub]
        if len(buffer) > max_buffer_size:
            buffer = random.sample(buffer, max_buffer_size)
        train_loader = DataLoader(ChessValueDataset(buffer), batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        train_value_net(current_net, train_loader, optimizer, num_epochs=num_epochs, device=DEVICE)
        wr = arena(current_net, best_net, games=arena_games)
        logger.info(f"Win-rate vs best: {wr*100:.1f}%")
        if wr > 0.55:
            best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
        else:
            logger.info("Rete non promossa, si continua…")
