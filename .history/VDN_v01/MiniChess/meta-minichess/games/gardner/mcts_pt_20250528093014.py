from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional
import torch
import numpy as np
from games.gardner.minichess_state import MiniChessState, Move
from games.gardner.value_network import ValueNetwork
from config import DEVICE, PIECE_TO_IDX, N_TYPES, INPUT_CHANNELS


def encode_state_as_tensor(state: MiniChessState) -> torch.Tensor:
    board = state.board()
    player = state.current_player()
    t = torch.zeros(INPUT_CHANNELS, 5, 5, dtype=torch.float32)
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p == 0:
                continue
            idx = PIECE_TO_IDX[abs(p)]
            if p > 0:
                t[idx, r, c] = 1.0
            else:
                t[idx + N_TYPES, r, c] = 1.0
    t[-1, :, :] = float(player)
    return t.to(DEVICE)

@dataclass
class _Node:
    state: MiniChessState
    parent: Optional[_Node]
    move: Optional[Move]
    wins: float = 0.0
    visits: int = 0
    children: List[_Node] = None
    untried: List[Move] = None
    prior: float = 1.0

    def __post_init__(self):
        self.children = []
        self.untried = self.state.legal_moves()

    def ucb1(self, exploration: float) -> float:
        q = 0.0 if self.visits == 0 else self.wins / self.visits
        u = exploration * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits) if self.parent else float('inf')
        return q + u

    def best_child(self, exploration: float) -> _Node:
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def expand(self, rng: random.Random) -> _Node:
        move = rng.choice(self.untried)
        self.untried.remove(move)
        next_state = self.state.next_state(move)
        child = _Node(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, value: float) -> None:
        node = self
        while node:
            node.visits += 1
            node.wins += value
            value = -value
            node = node.parent

class MCTS:
    def __init__(
        self,
        value_net: ValueNetwork,
        iterations: int = 50,
        exploration: float = math.sqrt(2.0),
        rng: Optional[random.Random] = None,
        batch_size: int = 32,
    ):
        self.iterations = iterations
        self.C = exploration
        self.rng = rng or random.Random()
        self.root: Optional[_Node] = None
        self.value_net = value_net.to(DEVICE)
        self.value_net.eval()
        self.batch_size = batch_size
        self._queue: List[_Node] = []

    def search(self, root_state: MiniChessState, temperature: float = 0.8) -> Move:
        # new root
        self.root = _Node(root_state, parent=None, move=None)
        # dirichlet noise
        if temperature > 0.0:
            alpha, eps = 0.3, 0.25
            moves = self.root.state.legal_moves()
            for mv in moves:
                child = _Node(self.root.state.next_state(mv), parent=self.root, move=mv)
                self.root.children.append(child)
            self.root.untried = []
            noise = np.random.dirichlet([alpha]*len(self.root.children))
            uniform = 1.0/len(self.root.children)
            for c, n in zip(self.root.children, noise):
                c.prior = (1-eps)*uniform + eps*n
        # simulations
        for _ in range(self.iterations):
            node = self._select(self.root)
            self._enqueue(node)
        self._flush()
        return self._select_move(temperature)

    def _select(self, node: _Node) -> _Node:
        while not node.untried and node.children:
            node = node.best_child(self.C)
        if node.untried:
            node = node.expand(self.rng)
        return node

    def _enqueue(self, leaf: _Node) -> None:
        self._queue.append(leaf)
        if len(self._queue) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        if not self._queue:
            return
        states = torch.stack([encode_state_as_tensor(n.state) for n in self._queue], dim=0)
        states = states.to(DEVICE)
        with torch.no_grad():
            raw = self.value_net(states)
        for node, out in zip(self._queue, raw):
            if out.numel()==1:
                v = out.item()
            else:
                v = out[0].item() if node.state.current_player()==1 else out[1].item()
            node.backpropagate(v)
        self._queue.clear()

    def _select_move(self, temperature: float) -> Move:
        children = self.root.children
        visits = [c.visits for c in children]
        if temperature==0.0:
            return max(children, key=lambda c: c.visits).move
        weights = [v**(1/temperature) for v in visits]
        total = sum(weights)
        probs = [w/total for w in weights]
        logger.info(f"mossa selezionata")
        return self.rng.choices(children, weights=probs, k=1)[0].move
