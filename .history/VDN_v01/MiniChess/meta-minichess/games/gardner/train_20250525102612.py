import os
import sys
import math
import random
import logging
from collections import Counter, deque
from typing import List, Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# Path-hack
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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

logger = logging.getLogger("MiniChessTrainer")
logging.basicConfig(level=logging.INFO)

CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_ckpt(net: torch.nn.Module, name: str) -> None:
    torch.save(net.state_dict(), os.path.join(CKPT_DIR, f"{name}.pth"))


def load_ckpt(name: str) -> torch.nn.Module:
    path = os.path.join(CKPT_DIR, f"{name}.pth")
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net


class ChessValueDataset(Dataset):
    def __init__(self, buffer: List[Tuple[torch.Tensor, float, float]]):
        # prioritized replay: use deque for easy popleft
        self.buffer = buffer
        # priorities: abs error placeholder (initial equal)
        self.priorities = [1.0] * len(buffer)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx: int):
        st, z_w, z_b = self.buffer[idx]
        target = torch.tensor([z_w, z_b], dtype=torch.float32, device=DEVICE)
        return st.to(DEVICE), target, idx

    def update_priorities(self, indices: List[int], errors: List[float]):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + 1e-6

    def sample_indices(self, batch_size: int) -> List[int]:
        total = sum(self.priorities)
        probs = [p/total for p in self.priorities]
        return random.choices(range(len(self.buffer)), weights=probs, k=batch_size)


def train_value_net(value_net, dataset, optimizer, scheduler, num_epochs):
    value_net.to(DEVICE)
    for epoch in range(num_epochs):
        total_loss = 0.0
        errors = []
        indices = []
        # use prioritized sampling
        for _ in range(len(dataset)//batch_size):
            batch_idx = dataset.sample_indices(batch_size)
            batch = [dataset[i] for i in batch_idx]
            states, targets, idxs = zip(*batch)
            states = torch.stack(states)
            targets = torch.stack(targets)
            preds = value_net(states)
            loss = torch.nn.functional.mse_loss(preds, targets, reduction='none')
            loss_mean = loss.mean()
            optimizer.zero_grad()
            loss_mean.backward()
            optimizer.step()
            total_loss += loss_mean.item()
            # update priorities
            batch_errors = (preds - targets).abs().mean(dim=1).tolist()
            dataset.update_priorities(idxs, batch_errors)
            indices.extend(idxs)
            errors.extend(batch_errors)
        avg_loss = total_loss / (len(dataset)//batch_size)
        scheduler.step(avg_loss)
        logger.info(f"Epoch {epoch+1}/{num_epochs} Loss={avg_loss:.5f}")
    return avg_loss




def self_play_game(mcts_white, mcts_black):
    transitions = []
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    history = []
    phi_prev = sum(p if p>0 else -p for row in state.board() for p in row)
    while not state.is_terminal():
        mover = state.current_player()
        mcts = mcts_white if mover==1 else mcts_black
        st_tensor = encode_state_as_tensor(state)
        history.append((st_tensor, None, None))
        move = mcts.search(state, temperature=1.0)
        next_state = state.next_state(move)
        phi_next = sum(p if p>0 else -p for row in next_state.board() for p in row)
        r_w = ALPHA*(phi_next-phi_prev) - (STEP_COST if mover==1 else 0)
        r_b = -ALPHA*(phi_next-phi_prev) - (STEP_COST if mover==-1 else 0)
        history[-1] = (st_tensor, r_w, r_b)
        state, phi_prev = next_state, phi_next
    z_w = state.result()
    z_b = -z_w if z_w!=1e-4 else z_w
    data = []
    cum_w = cum_b = 0.0
    for st, r_w, r_b in reversed(history):
        cum_w += r_w
        cum_b += r_b
        data.append((st, z_w + cum_w, z_b + cum_b))
    return (list(reversed(data)))


def play_det(net_w, net_b, mcts_w, mcts_b):
    game = GardnerMiniChessGame()
    state = MiniChessState(game.getInitBoard(), player=1, turns=0)
    while not state.is_terminal():
        mover = state.current_player()
        mcts = mcts_w if mover==1 else mcts_b
        move = mcts.search(state, temperature=0.0)
        state = state.next_state(move)
    return state.result()


def arena(current_net, best_net, games):
    wins=draws=0
    mcts_c = MCTS(current_net, iterations_MCTS)
    mcts_b = MCTS(best_net, iterations_MCTS)
    for _ in range(games//2):
        res = play_det(current_net, best_net, mcts_c, mcts_b)
        if res==1: wins+=1
        elif res==1e-4: draws+=1
    for _ in range(games - games//2):
        res = play_det(best_net, current_net, mcts_b, mcts_c)
        if res==-1: wins+=1
        elif res==1e-4: draws+=1
    rate = wins / games
    logger.info(f"Arena win-rate: {rate*100:.1f}% (draws {draws})")
    return rate


def main():
    best_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
    save_ckpt(best_net, "best_0")
    for cycle in range(1, num_cycles+1):
        logger.info(f"===== CICLO {cycle} =====")
        current_net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=2).to(DEVICE)
        current_net.load_state_dict(best_net.state_dict())
        optimizer = torch.optim.Adam(current_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
        scheduler = ReduceLROnPlateau(optimizer, factor=0.5, patience=1)
        buffer = deque(maxlen=max_buffer_size)
        mcts_c = MCTS(current_net, iterations_MCTS)
        mcts_b = MCTS(best_net, iterations_MCTS)
        for _ in range(games_per_cycle):
            for st, zw, zb in self_play_game(mcts_c, mcts_b): buffer.append((st,zw,zb))
            for st, zw, zb in self_play_game(mcts_b, mcts_c): buffer.append((st,zw,zb))
        dataset = ChessValueDataset(list(buffer))
        avg_loss = train_value_net(current_net, dataset, optimizer, scheduler, num_epochs)
        rate = arena(current_net, best_net, games=arena_games)
        if rate>0.6:
            best_net.load_state_dict(current_net.state_dict())
            save_ckpt(best_net, f"best_{cycle}")
            logger.info("Nuova rete promossa!")
        else:
            logger.info("Rete non promossa.")

if __name__=="__main__":
    main()
