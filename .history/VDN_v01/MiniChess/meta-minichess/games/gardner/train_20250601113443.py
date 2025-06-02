import os
import sys
import math
import random
import logging
import csv
import trueskill
from collections import Counter
from typing import List, Tuple
import torch
from torch import multiprocessing as mp
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast


# Limit internal PyTorch threading to avoid oversubscription
torch.set_num_threads(3)
torch.set_num_interop_threads(3)
device_id = torch.cuda.current_device()
torch.cuda.set_per_process_memory_fraction(0.4, device_id)

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
    DRAW_REPETITION, HIDDEN_CHANNELS, MAX_TURNS, N_TYPES, PIECE_TO_IDX, PIECE_VALUES, STEP_COST, ALPHA,
    num_cycles, arena_games, games_per_cycle,
    max_buffer_size, iterations_MCTS,
    learning_rate, weight_decay, batch_size, num_epochs,
    DEVICE, REWARD_DRAW, learning_rate
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
        # restituisco CPU‐tensors, il training loop li porterà su GPU
        return st_t.float(), torch.tensor([z_w, z_b], dtype=torch.float32)

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

def self_play_wrapped(args):
    net_w, net_b, seed, curr_white = args
    random.seed(seed)
    torch.manual_seed(seed)
    mcts_w = MCTS(net_w, iterations_MCTS, rng=random.Random(seed))
    mcts_b = MCTS(net_b, iterations_MCTS, rng=random.Random(seed+1))
    return self_play_game(mcts_w, mcts_b, curr_white)

def self_play_game(
    mcts_white: MCTS,
    mcts_black: MCTS,
    curr_white: bool
) -> List[Tuple[torch.Tensor, float, float]]:
    transitions = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())
    for i in range(1000):
        if state.is_terminal():
            break
        mover = state.current_player()
        mcts = mcts_white if mover == 1 else mcts_black
        tau = 1.0 if i < 50 else 0.0
        move = mcts.search(state, temperature=tau)
        next_state = state.next_state(move)
        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        st_t = encode_state_as_tensor(state).cpu()       # <-- force CPU
        # memorizzo anche chi ha mosso (1=white, 2=black)
        transitions.append((st_t, r_w, r_b, mover))
        state, phi_prev = next_state, phi_next
    #logger.info(f"game ended after {state._turns} turns, result: {state.result()}")
    z_w = state.result() * 5
    z_b = -z_w if z_w != REWARD_DRAW*20 else z_w
    data: List[Tuple[torch.Tensor,float,float]] = []
    cum_w = cum_b = 0.0
    for st_t, r_w, r_b, mover in reversed(transitions):
        cum_w += r_w
        cum_b += r_b
        # se il mover è white e curr_white==True, oppure mover nero e curr_white==False
        if (mover == 1 and curr_white) or (mover != 1 and not curr_white):
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

def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games) -> Tuple[float,int,int,int]:
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS, rng=random.Random(1))
    mcts_best = MCTS(best_net, iterations_MCTS, rng=random.Random(2))
    for _ in range(games//2):
        res = play_det(mcts_curr, mcts_best)
        if res==1: wins+=1
        elif res==REWARD_DRAW: draws+=1
    for _ in range(games//2):
        res = play_det(mcts_best, mcts_curr)
        if res==-1: wins+=1
        elif res==REWARD_DRAW: draws+=1
    losses = games - wins - draws
    logger.info(f"Arena: W/D/L = {wins}/{draws}/{losses}")
    return wins/(games-draws), wins, draws, losses

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

# ── SETUP METRICS CSV & ELO ───────────────────────────────────────────
metrics_path = os.path.join(this_dir, "metrics.csv")
# header file
if not os.path.exists(metrics_path):
    with open(metrics_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "cycle",
            "wins","draws","losses",
            "elo_current","elo_best",
            "avg_loss",
            "avg_moves","avg_nodes"
        ])
# trueskill env & initial ratings
ts_env     = trueskill.TrueSkill(draw_probability=0.05)
elo_current = ts_env.Rating()
elo_best    = ts_env.Rating()

# ── MAIN LOOP ────────────────────────────────────────────────────────
if __name__=="__main__":
    mp.set_start_method("spawn", force=True)
    torch.manual_seed(0); random.seed(0)

    # replay-buffer e reti…
    replay_buffer: List[Tuple[torch.Tensor,float,float]] = []
    best_net    = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE); save_ckpt(best_net,"best_0")
    current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    optimizer   = torch.optim.AdamW(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    logger.info(
        "CONFIG | STEP_COST=%s | num_cycles=%s | arena_games=%s | "
        "learning_rate=%s | batch_size=%s | HIDDEN_CHANNELS=%s | REWARD_DRAW=%s | iterations_MCTS=%s | REWARD * 20",
        STEP_COST, num_cycles, arena_games,
        learning_rate, batch_size, HIDDEN_CHANNELS, REWARD_DRAW, iterations_MCTS
    )
    for cycle in range(1, num_cycles+1):
        torch.manual_seed(cycle)
        random.seed(cycle)
        logger.info(f"===== GENERAZIONE {cycle} ===== ")
        seeds = [cycle*100 + i for i in range(games_per_cycle)]
        # prima del for i,s in enumerate(seeds):
        cpu_curr = current_net.to("cpu")
        cpu_best = best_net.to("cpu")
        args = []
        for i, s in enumerate(seeds):
            if i % 2 == 0:
                # current_net gioca Bianco
                args.append((cpu_curr, cpu_best, s, True))
            else:
                # current_net gioca Nero
                args.append((cpu_best, cpu_curr, s, False))

        with mp.Pool(processes=3) as pool:
            results = pool.map(self_play_wrapped, args)

        # subito dopo riportale su GPU
        current_net.to(DEVICE)
        best_net.to(DEVICE)

        # results: List[List[Tuple[state, z_w, z_b]]], un elemento per ogni partita
        games = results

        # separa i giochi in draw vs non-draw guardando l’ultimo z_w di ciascuna partita
        draw_games = [g for g in games if g[-1][1] == REWARD_DRAW]
        non_draw_games = [g for g in games if g[-1][1] != REWARD_DRAW]

        # undersample dei giochi in patta 
        desired_ratio = 0.6  # rapporto desiderato tra partite in patta e non in patta
        num_keep_draws = int(len(non_draw_games) * desired_ratio)
        if len(draw_games) > num_keep_draws:
            draw_games = random.sample(draw_games, num_keep_draws)

        # ricostruisci il buffer come flatten delle partite selezionate
        selected_games = non_draw_games + draw_games

        # flatten delle transizioni di questo ciclo
        new_buffer = [t for game in selected_games for t in game]

        # estendo il replay buffer e lo taglio alla massima lunghezza
        replay_buffer.extend(new_buffer)
        if len(replay_buffer) > max_buffer_size:
            replay_buffer = replay_buffer[-max_buffer_size:]

        # uso il replay_buffer per l’allenamento
        train_loader = DataLoader(
            ChessValueDataset(replay_buffer),
            batch_size=batch_size,
            shuffle=True,
            num_workers=3,
            pin_memory=True
        )

        avg_loss = train_value_net(current_net, train_loader, optimizer, num_epochs=num_epochs, device=DEVICE)

        # 1) arena & raccogli W/D/L
        wr, w, d, l = arena(current_net, best_net, games=arena_games)

        # 2) aggiorna Elo basato su esito complessivo: win/loss/draw
        if w > l:
            # current_net vince
            r_curr, r_best = ts_env.rate_1vs1(elo_current, elo_best)
        elif w < l:
            # best_net vince: invertiamo l’ordine
            r_best, r_curr = ts_env.rate_1vs1(elo_best, elo_current)
        else:
            # pareggio complessivo
            r_curr, r_best = ts_env.rate_1vs1(elo_current, elo_best, drawn=True)
        elo_current, elo_best = r_curr, r_best

        # 3) statistiche di lunghezza e nodi MCTS
        #   assumi di avere accumulato in self_play_wrapped i turns e mcts.total_nodes
        #   qui facciamo un semplice calcolo dummy:
        avg_moves = sum(len(game) for game in games) / len(games)
        avg_nodes = sum(getattr(game,"_nodes",0) for game in games) / len(games)

        # 4) scrivi su CSV
        with open(metrics_path, "a", newline="") as f:
            wcsv = csv.writer(f)
            wcsv.writerow([
                cycle,
                w, d, l,
                round(elo_current.mu,1), round(elo_best.mu,1),
                round(avg_loss,5),
                round(avg_moves,1), round(avg_nodes,1)
            ])

        # 5) checkpoint come prima…
        if wr>0.55:
            best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
            logger.info("Rete promossa!")
        else:
            logger.info("Rete non promossa, si continua…")
