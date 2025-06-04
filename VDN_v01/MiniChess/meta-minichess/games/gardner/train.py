import os
import sys
import math
import random
import logging
import csv
from collections import Counter
from typing import List, Tuple

import torch
from torch import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast

# Manage PyTorch threading & GPU memory
torch.set_num_threads(3)
torch.set_num_interop_threads(3)
if torch.cuda.is_available():
    dev = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(0.4, dev)

# Project path setup for imports
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Domain imports
from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from config import (
    MAX_TURNS, DRAW_REPETITION, GAMMA, PIECE_VALUES, ALPHA, STEP_COST,
    num_cycles, arena_games, games_per_cycle, max_buffer_size,
    iterations_MCTS, learning_rate, weight_decay,
    batch_size, num_epochs, DEVICE, REWARD_DRAW
)

# Setup logger to file
log_path = os.path.join(this_dir, "training.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in list(logger.handlers):
    logger.removeHandler(h)
file_handler = logging.FileHandler(log_path, mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# Checkpoint helpers
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
def save_ckpt(net, name):
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    torch.save(net.state_dict(), path)
    logger.info(f"Checkpoint saved: {path}")

def load_ckpt(name, device=DEVICE):
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    net = ValueNetwork().to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    logger.info(f"Checkpoint loaded: {path}")
    return net

# Dataset for replay buffer
class ChessValueDataset(Dataset):
    """(state_tensor, return) pairs for training value net."""
    def __init__(self, buffer: List[Tuple[torch.Tensor, float]]):
        self.items = buffer
    def __len__(self):
        return len(self.items)
    def __getitem__(self, idx):
        st_t, z = self.items[idx]
        # clamp and normalize return
        z = max(-5.0, min(5.0, z)) / 5.0
        return st_t.float(), torch.tensor(z)

def value_loss_fn(pred, target):
    return torch.nn.functional.mse_loss(pred, target)

# Reward utilities
def compute_white_black_rewards(phi_prev, phi_next, mover):
    """
    Compute shaped reward:
    - mover gets α(γφ'−φ)−step, opponent gets negative shaping.
    """
    shaping = ALPHA * (GAMMA * phi_next - phi_prev)
    if mover == 1:
        return shaping - STEP_COST, -shaping
    else:
        return shaping, -shaping - STEP_COST

def compute_material_potential(board):
    """Sum piece values over board, ignore kings."""
    total = 0.0
    for row in board:
        for piece in row:
            if piece == 0 or abs(piece) == 60000:
                continue
            val = PIECE_VALUES.get(abs(piece), 1.0)
            total += math.copysign(val, piece)
    return total

# Self-play functions
def self_play_wrapped(args):
    """Multiprocessing wrapper for one self-play game."""
    net_train, net_other, play_as_white, seed = args
    random.seed(seed); torch.manual_seed(seed)
    mcts_train = MCTS(net_train, iterations_MCTS, rng=random.Random(seed))
    mcts_other = MCTS(net_other, iterations_MCTS, rng=random.Random(seed+1))
    return self_play_game(
        mcts_train, mcts_other, collect_for_white=play_as_white
    )

def self_play_game(mcts_w, mcts_b, collect_for_white):
    """
    Play one game, collect transitions (s, r) for the training side only,
    then compute Monte-Carlo returns.
    """
    transitions = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())
    for ply in range(MAX_TURNS):
        if state.is_terminal():
            break
        mover = state.current_player()
        mcts = mcts_w if mover == 1 else mcts_b
        tau = 1.0 if ply < 50 else 0.0
        move = mcts.search(state, temperature=tau)
        next_state = state.next_state(move)
        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        r_self = r_w if mover == 1 else r_b
        if (collect_for_white and mover == 1) or (not collect_for_white and mover == -1):
            st_t = encode_state_as_tensor(state).cpu()
            transitions.append((st_t, r_self))
        state, phi_prev = next_state, phi_next
    result = state.result()
    result_self = result if collect_for_white else -result
    transitions.append((None, result_self))
    # Back-propagate returns
    data = []
    cum_r = 0.0
    for st_t, r in reversed(transitions):
        cum_r = r + GAMMA * cum_r
        if st_t is not None:
            data.append((st_t, cum_r))
    return data

# Deterministic evaluation for arena
def play_det(mcts_w, mcts_b):
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mcts = mcts_w if state.current_player()==1 else mcts_b
        state = state.next_state(mcts.search(state, temperature=0.0))
    return state.result()

def arena(current_net, best_net, games=arena_games):
    """Run half games as white/black, count W/D/L for current_net."""
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS, rng=random.Random(1))
    mcts_best = MCTS(best_net,   iterations_MCTS, rng=random.Random(2))
    for _ in range(games//2):
        res = play_det(mcts_curr, mcts_best)
        if res==1: wins+=1
        elif res==0: draws+=1
    for _ in range(games//2):
        res = play_det(mcts_best, mcts_curr)
        if res==-1: wins+=1
        elif res==0: draws+=1
    losses = games - wins - draws
    win_rate = wins/(games-draws) if games!=draws else 0.0
    logger.info(f"Arena W/D/L={wins}/{draws}/{losses}")
    return win_rate, wins, draws, losses

# Training loop
def train_value_net(value_net, train_loader, optimizer, num_epochs, device):
    """
    Standard PyTorch training:
    - mixed-precision with GradScaler
    - gradient clipping to norm 3.0
    - log average loss per epoch
    """
    value_net.to(device).train()
    scaler = GradScaler()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for states, targets in train_loader:
            states, targets = states.to(device), targets.to(device)
            optimizer.zero_grad()
            with autocast():
                preds = value_net(states)
                loss = value_loss_fn(preds, targets)
            scaler.scale(loss).backward()
            if torch.cuda.is_available():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 3.0)
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item() * states.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.5f}")
    return avg_loss

# Metrics CSV & TrueSkill setup
metrics_path = os.path.join(this_dir, "metrics.csv")
if not os.path.exists(metrics_path):
    with open(metrics_path,"w",newline="") as f:
        csv.writer(f).writerow([
            "cycle","wins","draws","losses",
            "elo_curr","elo_best","avg_loss","avg_moves","avg_nodes"
        ])

import trueskill
_ts = trueskill.TrueSkill(draw_probability=0.05)
elo_current = _ts.Rating(); elo_best = _ts.Rating()

# Main entry
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0); random.seed(0)

    # Initialise networks & buffer
    replay_buffer = []
    best_net = ValueNetwork().to(DEVICE)
    save_ckpt(best_net, "best_0")
    current_net = ValueNetwork().to(DEVICE)
    optimizer = torch.optim.AdamW(current_net.parameters(),
                                  lr=learning_rate, weight_decay=weight_decay)

    logger.info("CONFIG | STEP_COST=%s | cycles=%s | iterations_MCTS=%s",
                STEP_COST, num_cycles, iterations_MCTS)

    for cycle in range(1, num_cycles+1):
        torch.manual_seed(cycle); random.seed(cycle)
        logger.info(f"===== CYCLE {cycle} =====")
        # Prepare self-play args & launch parallel games
        seeds = [cycle*100+i for i in range(games_per_cycle)]
        cpu_curr, cpu_best = current_net.to("cpu"), best_net.to("cpu")
        args = [(cpu_curr, cpu_best, i%2==0, s) for i,s in enumerate(seeds)]
        with mp.Pool(min(4, mp.cpu_count())) as pool:
            results = pool.map(self_play_wrapped, args)
        # Back to GPU
        current_net.to(DEVICE); best_net.to(DEVICE)

        # Flatten and subsample transitions to avoid excess draws
        games_transitions: List[List[Tuple[torch.Tensor, float]]] = results

        draw_games     = [g for g in games_transitions if g and g[-1][1] == REWARD_DRAW]
        non_draw_games = [g for g in games_transitions if not g or g[-1][1] != REWARD_DRAW]

        # keep draws up to a ratio
        desired_ratio = 0.6
        max_draws     = int(len(non_draw_games) * desired_ratio)
        if len(draw_games) > max_draws:
            draw_games = random.sample(draw_games, max_draws)

        selected_games = non_draw_games + draw_games
        new_buffer = [t for game in selected_games for t in game]

        replay_buffer.extend(new_buffer)
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]
        

        # ── TRAIN VALUE‑NET ─────────────────────────────────────────
        train_loader = DataLoader(
            ChessValueDataset(replay_buffer),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )

        avg_loss = train_value_net(current_net, train_loader, optimizer, num_epochs=num_epochs, device=DEVICE)

        # ── ARENA EVALUATION ───────────────────────────────────────
        wr, w, d, l = arena(current_net, best_net, games=arena_games)

        # update Elo ratings
        if w > l:
            elo_current, elo_best = _ts.rate_1vs1(elo_current, elo_best)
        elif w < l:
            elo_best, elo_current = _ts.rate_1vs1(elo_best, elo_current)
        else:
            elo_current, elo_best = _ts.rate_1vs1(elo_current, elo_best, drawn=True)

        # naïve stats (placeholder)
        avg_moves = sum(len(g) for g in games_transitions) / len(games_transitions)
        avg_nodes = 0.0  # Node counting not yet wired through

        # write CSV
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([
                cycle, w, d, l,
                round(elo_current.mu, 1), round(elo_best.mu, 1),
                round(avg_loss, 5), round(avg_moves, 1), round(avg_nodes, 1)
            ])

        # ── PROMOTION LOGIC ────────────────────────────────────────
        if wr > 0.55:
            best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
            logger.info("Current net promoted to BEST!")
        else:
            logger.info("Current net NOT promoted – continuing…")
