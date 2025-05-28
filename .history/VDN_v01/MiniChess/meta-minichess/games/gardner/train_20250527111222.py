import os
import sys
import math
import random
import logging
from collections import Counter
from typing import List, Tuple

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
BASE_SEED = 0  # base seed per riproducibilità

# ────────────────────────────────────────────────────────────
# Path‑hack: aggiunge due livelli sopra (la cartella che contiene "games/")
this_dir: str = os.path.dirname(__file__)                              # .../games/gardner
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))    # .../meta-minichess
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────

from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.train import ChessValueDataset, train_value_net, play_det  # adattato come necessario
from config import DEVICE, ALPHA, STEP_COST, HIDDEN_CHANNELS,max_buffer_size, num_cycles, games_per_cycle, iterations_MCTS, num_epochs, batch_size, learning_rate, weight_decay, arena_games

def save_ckpt(net: torch.nn.Module, name: str) -> None:
    """
    Salva lo stato della rete come file .pt con nome dato.
    """
    path = f"{name}.pt"
    torch.save(net.state_dict(), path)
    logging.info(f"Checkpoint salvato: {path}")
def compute_material_potential(board: np.ndarray) -> float:
    """
    Calcola il potenziale materiale della board:
    somma dei valori assoluti dei pezzi (da config.PIECE_VALUES), con segno per il colore.
    board: array 5x5 con int (+pezzo bianco, -pezzo nero).
    """
    from config import PIECE_VALUES
    total = 0.0
    for r in range(board.shape[0]):
        for c in range(board.shape[1]):
            p = board[r][c]
            if p == 0:
                continue
            val = PIECE_VALUES[abs(p)]
            total += val if p > 0 else -val
    return total

# Setup logging
log_path = os.path.join(this_dir, 'training.log')
logger = logging.getLogger()
logger.setLevel(logging.INFO)
for h in logger.handlers[:]:
    logger.removeHandler(h)
file_handler = logging.FileHandler(log_path, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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
        # temperatura dinamica: tau=1.0 per le prime 10 mosse, altrimenti 0.0
        if state._turns < 10:
            temperature = 1.0
        else:
            temperature = 0.0
        move = mcts.search(state, temperature=temperature)
        next_state = state.next_state(move)

        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        transitions.append((encode_state_as_tensor(state), r_w, r_b))

        state = next_state
        phi_prev = phi_next

    return transitions


def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games, seed: int = None) -> float:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS, rng=random.Random(seed if seed is not None else 42))
    mcts_best = MCTS(best_net, iterations_MCTS, rng=random.Random(seed+1 if seed is not None else 43))
    half = games // 2

    for _ in range(half):
        res = play_det(current_net, best_net, mcts_curr, mcts_best)
        if res == 1:
            wins += 1
        elif res == 1e-4:
            draws += 1
    for _ in range(games - half):
        res = play_det(best_net, current_net, mcts_best, mcts_curr)
        if res == -1:
            wins += 1
        elif res == 1e-4:
            draws += 1

    return wins / games



if __name__ == "__main__":
    # ----------------------------------
    # Parametri (presi da config)
    # ----------------------------------
    num_cycles = num_cycles
    games_per_cycle = games_per_cycle

    best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
    best_net.load_state_dict(torch.load("best_0.pt"))

    for cycle in range(1, num_cycles + 1):
        seed = BASE_SEED + cycle
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        logger.info(f"===== GENERAZIONE {cycle} =====")
        current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
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

        wr = arena(current_net, best_net, games=arena_games, seed=BASE_SEED+cycle)
        logger.info(f"Win-rate vs best: {wr*100:.1f}%")
        if wr > 0.55:
            best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
        else:
            logger.info("Rete non promossa, si continua…")
