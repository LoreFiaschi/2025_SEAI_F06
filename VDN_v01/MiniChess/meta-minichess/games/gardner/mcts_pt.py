from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import List, Optional
from venv import logger
import torch
import numpy as np
from games.gardner.minichess_state import MiniChessState, Move
from games.gardner.value_network import ValueNetwork
from config import DEVICE, PIECE_TO_IDX


def encode_state_as_tensor(state):
    """
    Convert a MiniChessState into a 12×5×5 tensor:
    - 6 channels for the mover’s pieces, 6 for the opponent’s.
    - Board is assumed already rotated so that the side to move is 'white'.
    """
    board = state.board()              # get 5×5 int matrix
    mover = state.current_player()     # +1 or -1
    t = torch.zeros(12, 5, 5, dtype=torch.float32)
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p == 0:
                continue
            idx = PIECE_TO_IDX[abs(p)]  # map piece id → 0..5
            off = 0 if p * mover > 0 else 6  # same-color vs opponent
            t[idx + off, r, c] = 1.0
    return t.to(DEVICE)


@dataclass
class _Node:
    """
    One node in the MCTS tree:
    - state: the game state at this node
    - parent: reference to parent node (None for root)
    - move: the Move that led here
    - wins, visits: aggregated statistics for backpropagation
    - children: list of expanded child nodes
    - untried: moves yet to be expanded
    - prior: prior probability (set via Dirichlet or uniform)
    """
    state: MiniChessState
    parent: Optional[_Node]
    move: Optional[Move]
    wins: float = 0.0
    visits: int = 0
    children: List[_Node] = None
    untried: List[Move] = None
    prior: float = 1.0

    def __post_init__(self):
        # Initialize expansion list
        self.children = []
        self.untried = self.state.legal_moves()

    def ucb1(self, exploration: float) -> float:
        """
        Compute UCB1 score: Q/N + c * prior * sqrt(parent.N)/(1+N)
        """
        q = 0.0 if self.visits == 0 else self.wins / self.visits
        if self.parent:
            u = exploration * self.prior * math.sqrt(self.parent.visits) / (1 + self.visits)
        else:
            u = float('inf')  # root has infinite priority
        return q + u

    def best_child(self, exploration: float) -> _Node:
        # Select child with highest UCB1
        return max(self.children, key=lambda c: c.ucb1(exploration))

    def expand(self, rng: random.Random) -> _Node:
        """
        Randomly pick an untried move, apply it, create and append the child.
        """
        move = rng.choice(self.untried)
        self.untried.remove(move)
        next_state = self.state.next_state(move)
        child = _Node(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, value: float) -> None:
        """
        Propagate value up to the root, flipping sign at each step
        since players alternate.
        """
        node = self
        while node:
            node.visits += 1
            node.wins += value
            value = -value
            node = node.parent


class MCTS:
    """
    Monte Carlo Tree Search with batch evaluation using a ValueNetwork.
    """
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
        self.value_net.eval()         # set network to eval mode
        self.batch_size = batch_size
        self._queue: List[_Node] = []  # nodes waiting for evaluation

    def search(self, root_state: MiniChessState, temperature: float = 0.8) -> Move:
        """
        Main entry: build tree from root_state, add Dirichlet noise,
        run 'iterations' simulations, then pick a move given temperature.
        """
        # 1. Initialize root node
        self.root = _Node(root_state, parent=None, move=None)
        # 2. Add Dirichlet noise for exploration (AlphaZero-style)
        if temperature > 0.0:
            alpha, eps = 0.3, 0.25
            for mv in self.root.state.legal_moves():
                child = _Node(self.root.state.next_state(mv), parent=self.root, move=mv)
                self.root.children.append(child)
            self.root.untried = []
            noise = np.random.dirichlet([alpha] * len(self.root.children))
            uniform = 1.0 / len(self.root.children)
            for c, n in zip(self.root.children, noise):
                c.prior = (1 - eps) * uniform + eps * n
        # 3. Simulations
        for _ in range(self.iterations):
            node = self._select(self.root)
            self._enqueue(node)
        self._flush()
        # 4. Select a final move
        return self._select_move(temperature)

    def _select(self, node: _Node) -> _Node:
        # Traverse until a leaf or a node with untried moves
        while not node.untried and node.children:
            node = node.best_child(self.C)
        if node.untried:
            node = node.expand(self.rng)
        return node

    def _enqueue(self, leaf: _Node) -> None:
        """
        Add leaf to evaluation queue; flush when enough are collected.
        """
        self._queue.append(leaf)
        if len(self._queue) >= self.batch_size:
            self._flush()

    def _flush(self) -> None:
        """
        Batch-evaluate all queued nodes with the value network,
        then backpropagate each estimate.
        """
        if not self._queue:
            return
        states = torch.stack(
            [encode_state_as_tensor(n.state) for n in self._queue], dim=0
        ).to(DEVICE)
        with torch.no_grad():
            raw = self.value_net(states)
        for node, out in zip(self._queue, raw):
            v = max(-1.0, min(1.0, out.item()))  # clamp to [-1,1]
            node.backpropagate(v)
        self._queue.clear()

    def _select_move(self, temperature: float) -> Move:
        """
        Choose final action based on visit counts:
        - temperature=0 → greedy by win ratio
        - else → sample proportionally to visits^(1/temperature)
        """
        children = self.root.children
        if temperature == 0.0:
            def score(c):
                return (c.wins / c.visits) if c.visits > 0 else float("-inf")
            return max(children, key=score).move

        visits = [c.visits for c in children]
        weights = [v ** (1 / temperature) for v in visits]
        total = sum(weights)
        probs = [w / total for w in weights]
        return self.rng.choices(children, weights=probs, k=1)[0].move
