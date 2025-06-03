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

Z_MAX = 5.0

# ── PYTORCH THREADING / GPU MEM ─────────────────────────────────────
torch.set_num_threads(3)
torch.set_num_interop_threads(3)
if torch.cuda.is_available():
    device_id = torch.cuda.current_device()
    torch.cuda.set_per_process_memory_fraction(0.4, device_id)

# ── PATH SETUP ───────────────────────────────────────────────────────
this_dir: str = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# ── DOMAIN IMPORTS ───────────────────────────────────────────────────
from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from config import (
    DRAW_REPETITION, GAMMA, HIDDEN_CHANNELS, MAX_TURNS, N_TYPES, PIECE_TO_IDX, PIECE_VALUES,
    STEP_COST, ALPHA,
    num_cycles, arena_games, games_per_cycle,
    max_buffer_size, iterations_MCTS,
    learning_rate, weight_decay, batch_size, num_epochs,
    DEVICE, REWARD_DRAW
)

# ── LOGGING ──────────────────────────────────────────────────────────
log_path = os.path.join(this_dir, "training.log")
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
file_handler = logging.FileHandler(log_path, mode="a")
file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
logger.addHandler(file_handler)

# ── CHECKPOINT HELPERS ───────────────────────────────────────────────
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_ckpt(net: torch.nn.Module, name: str) -> None:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    torch.save(net.state_dict(), path)
    logger.info(f"Checkpoint saved: {path}")

def load_ckpt(name: str, device: str = DEVICE) -> torch.nn.Module:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
    logger.info(f"Checkpoint loaded: {path}")
    return net

# ── DATASET ──────────────────────────────────────────────────────────
class ChessValueDataset(Dataset):
    """Replay‑buffer dataset. Each item is (state_tensor, z_scalar)."""

    def __init__(self, buffer: List[Tuple[torch.Tensor, float]]):
        self.items = buffer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx: int):
        st_t, z = self.items[idx]
        z = max(-Z_MAX, min(Z_MAX, z))
        z = z / Z_MAX
        return st_t.float(), torch.tensor(z, dtype=torch.float32)


def value_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)

# ── DOMAIN‑SPECIFIC REWARD UTILS ─────────────────────────────────────

def compute_white_black_rewards(phi_prev: float, phi_next: float, mover: int) -> Tuple[float, float]:
    """
    Shaped reward: potential‐based shaping + step‐cost, 
    dal punto di vista di White (+1) e Black (-1).
    """
    # Calcola il shaping: α[γ·φ(s') − φ(s)]
    shaping = ALPHA * (GAMMA * phi_next - phi_prev)

    if mover == 1:
        # Se tocca a White, White riceve shaping meno step‐cost, Black riceve −shaping “puro”
        return shaping - STEP_COST, -shaping
    else:
        # Se tocca a Black, Black riceve shaping meno step‐cost, White riceve −shaping “puro”
        return shaping, -shaping - STEP_COST



def compute_material_potential(board: Tuple[Tuple[int, ...], ...]) -> float:
    total = 0.0
    for row in board:
        for piece in row:
            if piece == 0 or abs(piece) == 60000:  # ignore empty & king (constant value)
                continue
            val = PIECE_VALUES.get(abs(piece), 1.0)
            total += math.copysign(val, piece)
    return total

# ── SELF‑PLAY ────────────────────────────────────────────────────────

def self_play_wrapped(args):
    """Wrapper to use in mp.Pool."""
    net_train, net_other, play_as_white, seed = args
    random.seed(seed)
    torch.manual_seed(seed)

    mcts_train = MCTS(net_train, iterations_MCTS, rng=random.Random(seed))
    mcts_other = MCTS(net_other, iterations_MCTS, rng=random.Random(seed + 1))

    if play_as_white:
        return self_play_game(mcts_train, mcts_other, collect_for_white=True)
    else:
        return self_play_game(mcts_other, mcts_train, collect_for_white=False)


def self_play_game(mcts_white: MCTS, mcts_black: MCTS, collect_for_white: bool) -> List[Tuple[torch.Tensor, float]]:
    """Plays one game.
    Returns a list of (state_tensor, z_scalar) **only** for positions
    where *our* network (the one we are training/evaluating) is to move.
    `collect_for_white` indicates whether the training net is playing white.
    """

    transitions: List[Tuple[torch.Tensor, float]] = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())

    # ── PLAY THE GAME ───────────────────────────────────────────────
    for ply in range(MAX_TURNS):
        if state.is_terminal():
            break

        mover = state.current_player()           # +1 white, -1 black *about to move*
        mcts  = mcts_white if mover == 1 else mcts_black
        tau   = 1.0 if ply < 50 else 0.0
        move  = mcts.search(state, temperature=tau)
        next_state = state.next_state(move)

        # shaped reward for the mover
        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        r_self   = r_w if mover == 1 else r_b

        # store transition *only* if it's our training side
        if (collect_for_white and mover == 1) or (not collect_for_white and mover == -1):
            st_t = encode_state_as_tensor(state).cpu()
            transitions.append((st_t, r_self))

        state, phi_prev = next_state, phi_next

    # ── GAME RESULT FROM EACH SIDE'S PERSPECTIVE ───────────────────
    result_white = state.result()          # +1 white win | 0 draw | -1 white lose
    result_self  = result_white if collect_for_white else -result_white
    transitions.append((None, result_self))
    # ── RETURN (s, z) WITH MONTE‑CARLO RETURNS ─────────────────────
    data: List[Tuple[torch.Tensor, float]] = []
    cum_r = 0.0
    for st_t, r in reversed(transitions):
        cum_r = r + GAMMA * cum_r
        # Attenzione: se st_t is None significa che è la transizione terminale
        if st_t is not None:
            data.append((st_t, cum_r))
    return data

# ── DETERMINISTIC PLAY FOR ARENA ────────────────────────────────────

def play_det(mcts_w: MCTS, mcts_b: MCTS) -> int:
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mcts = mcts_w if state.current_player() == 1 else mcts_b
        move = mcts.search(state, temperature=0.0)
        state = state.next_state(move)
    return state.result()   # +1 white win | 0 draw | -1 white lose


def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games):
    """Returns win‑rate, wins, draws, losses from the POV of `current_net`."""
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS, rng=random.Random(1))
    mcts_best = MCTS(best_net,   iterations_MCTS, rng=random.Random(2))

    # current_net as white
    for _ in range(games // 2):
        res = play_det(mcts_curr, mcts_best)
        if res == 1:
            wins += 1
        elif res == REWARD_DRAW:
            draws += 1

    # current_net as black
    for _ in range(games // 2):
        res = play_det(mcts_best, mcts_curr)
        if res == -1:
            wins += 1
        elif res == REWARD_DRAW:
            draws += 1

    losses = games - wins - draws
    logger.info(f"Arena: W/D/L = {wins}/{draws}/{losses}")
    win_rate = wins / (games - draws) if games != draws else 0.0
    return win_rate, wins, draws, losses

# ── TRAIN LOOP FOR VALUE‑NET ────────────────────────────────────────

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
                preds = value_net(states_batch).squeeze(-1)   # shape [B]
                loss = value_loss_fn(preds, targets_batch)
            scaler.scale(loss).backward()
            if torch.cuda.is_available():      # evita warning su CPU
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * states_batch.size(0)
        avg_loss = total_loss / len(train_loader.dataset)
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.5f}")
    return avg_loss

# ── METRICS CSV / TRUE‑SKILL ────────────────────────────────────────
metrics_path = os.path.join(this_dir, "metrics.csv")
if not os.path.exists(metrics_path):
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow([
            "cycle", "wins", "draws", "losses", "elo_current", "elo_best", "avg_loss", "avg_moves", "avg_nodes"
        ])

import trueskill
_ts_env = trueskill.TrueSkill(draw_probability=0.05)
elo_current = _ts_env.Rating()
elo_best    = _ts_env.Rating()

# ── MAIN TRAINING LOOP ──────────────────────────────────────────────
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0)
    random.seed(0)

    # ── INITIALISE NETS & BUFFER ────────────────────────────────────
    replay_buffer: List[Tuple[torch.Tensor, float]] = []

    best_net    = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
    save_ckpt(best_net, "best_0")
    current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)

    optimizer = torch.optim.AdamW(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

    logger.info(
        "CONFIG | STEP_COST=%s | num_cycles=%s | arena_games=%s | "
        "learning_rate=%s | batch_size=%s | HIDDEN_CHANNELS=%s | REWARD_DRAW=%s | iterations_MCTS=%s",
        STEP_COST, num_cycles, arena_games,
        learning_rate, batch_size, HIDDEN_CHANNELS, REWARD_DRAW, iterations_MCTS
    )

    # ── CYCLES ──────────────────────────────────────────────────────
    for cycle in range(1, num_cycles + 1):
        torch.manual_seed(cycle)
        random.seed(cycle)
        logger.info(f"===== CYCLE {cycle} =====")

        # ── SELF‑PLAY PARALLEL GAMES ────────────────────────────────
        seeds = [cycle * 100 + i for i in range(games_per_cycle)]
        cpu_curr = current_net.to("cpu")
        cpu_best = best_net.to("cpu")

        args = []
        for i, s in enumerate(seeds):
            if i % 2 == 0:
                args.append((cpu_curr, cpu_best, True, s))   # training net plays WHITE
            else:
                args.append((cpu_best, cpu_curr, False, s))  # training net plays BLACK

        with mp.Pool(processes=min(4, mp.cpu_count())) as pool:
            results = pool.map(self_play_wrapped, args)

        # back to GPU
        current_net.to(DEVICE)
        best_net.to(DEVICE)

        # ── FLATTEN BUFFER & OPTIONALLY SUBSAMPLE DRAWS ─────────────
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
            elo_current, elo_best = _ts_env.rate_1vs1(elo_current, elo_best)
        elif w < l:
            elo_best, elo_current = _ts_env.rate_1vs1(elo_best, elo_current)
        else:
            elo_current, elo_best = _ts_env.rate_1vs1(elo_current, elo_best, drawn=True)

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
