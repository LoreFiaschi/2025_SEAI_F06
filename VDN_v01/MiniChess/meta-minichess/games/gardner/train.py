import os
import sys

# ────────────────────────────────────────────────────────────────────────────
# Path-hack: aggiunge due livelli sopra (la cartella che contiene "games/")
this_dir = os.path.dirname(__file__)                                  # .../games/gardner
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))    # .../meta-minichess
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────────────────

import math
import random
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS, encode_state_as_tensor
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

# ───────────── Funzione di potenziale basata sul materiale ────────────────
PIECE_VALUES = {
    100: 0.001,    # pedone
    280: 0.003,    # cavallo
    320: 0.003,    # alfiere
    479: 0.005,    # torre
    929: 0.011,    # regina
    60000: 0.0       # re → non influisce sul potenziale
}
STEP_COST = 0.0001  # costo per ogni mossa (per evitare che il reward shaping sia sempre positivo)
ALPHA = 1        # fattore di scaling per il reward shaping
# ────────────────────────────────────────────────────────────────────────────


# ---------- Hyperparametri globali ----------
num_cycles      = 50         # cicli self-play + training
games_per_cycle = 20         # partite self-play per ciclo
max_buffer_size = 10000      # dimensione massima del buffer
iterations_MCTS = 100        # simulazioni MCTS per mossa

learning_rate = 5e-3         # inizialmente 5e-3
weight_decay  = 0.0          # nessuna regolarizzazione

batch_size = 64              # dimensione del minibatch
num_epochs = 3               # epoche di training per ciclo
device = "cuda" if torch.cuda.is_available() else "cpu"

# Contatore per monitorare la distribuzione di z_final
z_counter = Counter()


# ---------- Dataset per i valori di allenamento ----------
class ChessValueDataset(Dataset):
    def __init__(self, buffer: list[tuple[torch.Tensor, float]]):
        self.items = buffer

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        st_t, z_t = self.items[idx]
        return st_t.float(), torch.tensor([z_t], dtype=torch.float32)


def value_loss_fn(value_pred: torch.Tensor, value_target: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.mse_loss(value_pred.view(-1), value_target.view(-1))


# ---------- Funzione di potenziale basata sul materiale ────────────────────
def compute_material_potential(board: tuple[tuple[int]]) -> float:
    """
    Calcola la differenza di materiale di un board 5×5.
    Esclude il re (abs(piece) == 6) dal conteggio.
    """
    total = 0.0
    for r in range(len(board)):
        for c in range(len(board[r])):
            piece = board[r][c]
            if piece == 0:
                continue
            if abs(piece) == 6:
                # Ignora il re
                continue
            val = PIECE_VALUES.get(abs(piece), 1)
            total += math.copysign(val, piece)
    return total
# ────────────────────────────────────────────────────────────────────────────


# ---------- Self-play con reward shaping potenziale-based ------------
def self_play_game(mcts: MCTS) -> list[tuple[torch.Tensor, float]]:
    """
    Gioca una partita in self-play tra due agenti MCTS.
    Applica reward shaping potenziale-based per ogni transizione.
    Ritorna una lista di coppie (state_tensor, z_state_modificato).
    """
    data = []
    game = GardnerMiniChessGame()
    initial_board = game.getInitBoard()       # List[List[int]] 5×5
    state = MiniChessState(board=initial_board, player=1, turns=0)

    # Calcolo potenziale iniziale
    phi_prev = compute_material_potential(state.board())

    while not state.is_terminal():
        board_before = state.board()
        phi_prev = compute_material_potential(board_before)

        # MCTS per scegliere la mossa (temperature=1.0 per self-play)
        move = mcts.search(state, temperature=1.0)
        next_state = state.next_state(move)
        # Calcolo potenziale dopo la mossa
        board_after = next_state.board()
        phi_next = compute_material_potential(board_after)

        # Reward shaping potenziale-based: differenza di potenziale
        shaping_reward = ALPHA * (phi_next - phi_prev) - STEP_COST
        print(f"  mossa: {next_state}, reward shaping: {shaping_reward:.4f}")

        # Creo il tensore per lo stato precedente come (4,5,5) tramite encode_state_as_tensor
        state_tensor = encode_state_as_tensor(state)  # restituisce (4,5,5)

        data.append((state_tensor, shaping_reward))
        state = next_state

    # Al termine della partita
    print(f" board finale: {state}")
    z_final = state.result()  # +1 vittoria bianco, -1 vittoria nero, 0 patta
    print(f" risultato finale: {z_final}")
    z_counter[z_final] += 1

    # Costruisco i target finali z_state
    result_data = []
    cumulative_shaping = 0.0

    for st_t, reward_i in data:
        player_i = st_t[2, 0, 0].item()  # canale turno (+1 o -1)
        cumulative_shaping += reward_i * player_i
        z_terminal = z_final * player_i
        z_state = float(z_terminal + cumulative_shaping)
        result_data.append((st_t, z_state))

    return result_data


# ---------- Funzione di training della rete di valore ----------
def train_value_net(value_net, train_loader, optimizer, num_epochs=3, device="cpu"):
    """
    Esegue il training di value_net per num_epochs epoche sul train_loader.
    Stampa diagnostica su distribuzione target e norma dei gradienti.
    """
    value_net.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for (states_batch, targets_batch) in train_loader:
            # states_batch: (B, 4, 5, 5)
            # targets_batch: (B, 1)

            # Debug: distribuzione dei target nel batch (decommentare se serve)
            # values, counts = torch.unique(targets_batch.view(-1), return_counts=True)
            # dist = dict(zip(values.tolist(), counts.tolist()))
            # print(f"    [DEBUG] Batch targets distribution: {dist}")

            states_batch = states_batch.to(device)
            targets_batch = targets_batch.to(device)

            optimizer.zero_grad()
            preds = value_net(states_batch)         # (B,1)
            loss = value_loss_fn(preds, targets_batch)
            loss.backward()

            # Debug: norma del gradiente sul primo parametro (decommentare se serve)
            # grad_norm = next(value_net.parameters()).grad.norm().item()
            # print(f"    [DEBUG] Grad norm conv1.weight: {grad_norm:.6f}")

            optimizer.step()
            total_loss += loss.item() * states_batch.size(0)

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"  Epoch {epoch+1}/{num_epochs}: Loss media = {avg_loss:.4f}")
    return avg_loss


# ────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # 1) Creo rete e MCTS
    value_net = ValueNetwork(hidden_channels=32).to(device)
    optimizer = torch.optim.Adam(
        value_net.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    mcts = MCTS(value_net=value_net, iterations=iterations_MCTS)

    buffer = []
    for cycle in range(num_cycles):
        print(f"\n===== Ciclo {cycle+1}/{num_cycles} =====")

        # --- A) Self-Play e popolamento buffer ---
        for _ in range(games_per_cycle):
            game_data = self_play_game(mcts)
            for (st_t, z_t) in game_data:
                if len(buffer) < max_buffer_size:
                    buffer.append((st_t, z_t))
                else:
                    idx = random.randrange(max_buffer_size)
                    buffer[idx] = (st_t, z_t)

        print(f" Buffer size dopo self-play: {len(buffer)}")
        print(f"  Distribuzione z_final (ultimo ciclo): {dict(z_counter)}")
        z_counter.clear()

        # --- B) Addestramento su buffer corrente ---
        train_dataset = ChessValueDataset(buffer)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        avg_loss = train_value_net(
            value_net, train_loader, optimizer, num_epochs=num_epochs, device=device
        )
        print(f" Ciclo {cycle+1} terminato, Loss media complessiva: {avg_loss:.4f}")

    # 3) Salvataggio del modello
    save_path = "value_net_final.pth"
    torch.save(value_net.state_dict(), save_path)
    print(f"\nModello addestrato salvato in: {save_path}")
