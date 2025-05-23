import os
import sys
import math
import random
from collections import Counter
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

# ────────────────────────────────────────────────────────────────────────────
# Path‑hack: aggiunge due livelli sopra (la cartella che contiene "games/")
this_dir: str = os.path.dirname(__file__)                              # .../games/gardner
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))  # .../meta‑minichess
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────────────────

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

# ─── CONFIG.PY ──────────────────────────────────────────────────────────────────────── 
from config import HIDDEN_CHANNELS, PIECE_VALUES, STEP_COST, ALPHA
from config import num_cycles, arena_games, games_per_cycle
from config import max_buffer_size, iterations_MCTS
from config import learning_rate, weight_decay, batch_size, num_epochs
from config import DEVICE
# ──────────────────────────────────────────────────────────────────────────── 


# Contatore diagnostico per la distribuzione dei risultati finali
z_counter: Counter = Counter()

# ─── Gestione dei checkpoint ───────────────────────────────────────────────
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_ckpt(net: torch.nn.Module, name: str) -> None:
    torch.save(net.state_dict(), os.path.join(CKPT_DIR, f"{name}.pth"))


def load_ckpt(name: str, device: str = DEVICE) -> torch.nn.Module:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    net = ValueNetwork(hidden_channels=32, output_dim=2).to(device)
    net.load_state_dict(torch.load(path, map_location=device))
    net.eval()
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


# ---------- Loss MSE su due output ----------

def value_loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(pred, target)


# ---------- Reward shaping separato per colore ----------

def compute_white_black_rewards(
    phi_prev: float,
    phi_next: float,
    mover: int  # +1 Bianco, -1 Nero
) -> Tuple[float, float]:
    """Restituisce (reward_white, reward_black) per la mossa appena giocata."""
    delta = phi_next - phi_prev  # >0 se il Bianco guadagna materiale
    if mover == 1:
        r_w =  ALPHA * delta - STEP_COST
        r_b = -ALPHA * delta
    else:
        r_w =  ALPHA * delta
        r_b = -ALPHA * delta - STEP_COST
    return r_w, r_b


# ---------- Potenziale materiale ----------

def compute_material_potential(board: Tuple[Tuple[int, ...], ...]) -> float:
    total = 0.0
    for row in board:
        for piece in row:
            if piece == 0 or abs(piece) == 60000:
                continue
            val = PIECE_VALUES.get(abs(piece), 1.0)
            total += math.copysign(val, piece)  # positivo se pezzo bianco
    return total


# ---------- Self‑play asimmetrico (due reti) ----------

def self_play_game(
    mcts_white: MCTS,
    mcts_black: MCTS,
) -> List[Tuple[torch.Tensor, float, float]]:
    transitions: List[Tuple[torch.Tensor, float, float]] = []  # (tensor, r_w, r_b)
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    phi_prev = compute_material_potential(state.board())

    while not state.is_terminal():
        
        mover = state.current_player()
        mcts = mcts_white if mover == 1 else mcts_black
        move = mcts.search(state, temperature=1.0)
        next_state = state.next_state(move)

        phi_next = compute_material_potential(next_state.board())
        r_w, r_b = compute_white_black_rewards(phi_prev, phi_next, mover)
        transitions.append((encode_state_as_tensor(state), r_w, r_b))

        state, phi_prev = next_state, phi_next
        
    # Esito dal punto di vista del Bianco (serve board come LISTA di liste)

   #board_lists = [list(r) for r in state.board()]
    z_w = state.result()   # +1 win W, -1 win B, 0 draw
        
    z_b = -z_w
        
    if(z_w == 1e-4):
        z_b = z_w
    # Propagazione retrograda dei reward di shaping
    data: List[Tuple[torch.Tensor, float, float]] = []
    cum_w = cum_b = 0.0
    for st_t, r_w, r_b in reversed(transitions):
        cum_w += r_w
        cum_b += r_b
        data.append((st_t, z_w + cum_w, z_b + cum_b))
    return data


# ---------- Arena deterministica ----------

def play_det(
    net_white: torch.nn.Module,
    net_black: torch.nn.Module,
    mcts_w: MCTS,
    mcts_b: MCTS,
) -> int:
    """Singola partita a temperatura 0. Ritorna +1, 0, -1 dal POV del Bianco."""
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mcts = mcts_w if state.current_player() == 1 else mcts_b
        move = mcts.search(state, temperature=0.0)
        state = state.next_state(move)
        
    return state.result()


def arena(current_net: torch.nn.Module, best_net: torch.nn.Module, games: int = arena_games) -> float:
    """Percentuale di vittorie di current_net contro best_net."""
    wins = draws = 0
    mcts_curr = MCTS(current_net, iterations_MCTS)
    mcts_best = MCTS(best_net, iterations_MCTS)
    half = games // 2

    # current_net con il Bianco
    for i in range(half):
        res = play_det(current_net, best_net, mcts_curr, mcts_best)
        if res == 1:
            wins += 1
            
        elif res == 0:
            draws += 1
        print(res)
    print(f"current_net Vittorie con il Bianco: {wins}")
    wins_white = wins
    # current_net con il Nero
    for _ in range(games - half):
        res = play_det(best_net, current_net, mcts_best, mcts_curr)
        if res == -1:  # il Nero vince ⇒ current_net vince
            wins += 1
        elif res == 0:
            draws += 1
    print(f"current_net Vittorie con il Nero: { wins - wins_white}")
    print(f"current_net Pareggi: {draws}")
    return wins / (games-draws)  # percentuale di vittorie del current_net


# ---------- Training della rete di valore ----------

def train_value_net(
    value_net: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    num_epochs: int = num_epochs,
    device: torch.device = DEVICE,
) -> float:
    value_net.to(device)
    value_net.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for states_batch, targets_batch in train_loader:
            # sposto batch su GPU
            states_batch = states_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            preds = value_net(states_batch)
            loss = value_loss_fn(preds, targets_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * states_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"    Epoch {epoch + 1}/{num_epochs} | Loss = {avg_loss:.5f}")
    return avg_loss

if __name__ == "__main__":
    # Primo checkpoint casuale
    best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    save_ckpt(best_net, "best_0")

    for cycle in range(1, num_cycles + 1):
        print(f"\n===== GENERAZIONE {cycle} =====")

        # Clone del best per iniziare l'apprendimento
        current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
        current_net.load_state_dict(best_net.state_dict())
        optimizer = torch.optim.Adam(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # ---- SELF‑PLAY ASIMMETRICO ----
        buffer: list[tuple[torch.Tensor, float, float]] = []
        mcts_curr = MCTS(current_net, iterations_MCTS)
        mcts_best = MCTS(best_net, iterations_MCTS)

        for _ in range(games_per_cycle):
            # current_net Bianco
            buffer.extend(self_play_game(mcts_curr, mcts_best))
            # current_net Nero
            buffer.extend(self_play_game(mcts_best, mcts_curr))

        # Trim buffer se troppo grande
        if len(buffer) > max_buffer_size:
            buffer = random.sample(buffer, max_buffer_size)

        # ---- TRAINING ----
        train_loader = DataLoader(ChessValueDataset(buffer), batch_size=batch_size, shuffle=True)
        train_value_net(current_net, train_loader, optimizer, num_epochs=num_epochs, device=DEVICE)

        # ---- ARENA TEST ----
        wr = arena(current_net, best_net, games=arena_games)
        print(f"  Win‑rate vs best = {wr * 100:.1f}%")
        if wr > 0.55:
            print("  ✅ Nuova rete promossa!")
            best_net = ValueNetwork(hidden_channels=32, output_dim=2).to(DEVICE)
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
        else:
            print("  ❌ Rete non promossa, si continua…")
