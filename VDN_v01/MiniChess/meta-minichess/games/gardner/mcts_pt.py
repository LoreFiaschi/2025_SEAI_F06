from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict

from games.gardner.minichess_state import MiniChessState, Move
import torch
from games.gardner.value_network import ValueNetwork
from config import DEVICE, PIECE_TO_IDX, N_TYPES, INPUT_CHANNELS


def encode_state_as_tensor(state: MiniChessState) -> torch.Tensor:
    """
    Ritorna un tensore (13, 5, 5):
      canali   0-5 : un tipo di pezzo bianco per canale
      canali   6-11: un tipo di pezzo nero per canale
      canale      12: intero +1 / -1 per il side-to-move
    """
    board  = state.board()
    player = state.current_player()

    t = torch.zeros(INPUT_CHANNELS, 5, 5, dtype=torch.float32)
    for r in range(5):
        for c in range(5):
            p = board[r][c]
            if p == 0:
                continue
            idx = PIECE_TO_IDX[abs(p)]
            if p > 0:                         # bianco
                t[idx,        r, c] = 1.0
            else:                             # nero
                t[idx+N_TYPES, r, c] = 1.0
    t[-1, :, :] = float(player)
    return t.to(DEVICE)



@dataclass(slots=True)
class _Node:
    state: MiniChessState
    parent: Optional[_Node]
    move: Optional[Move]
    wins: float = 0.0
    visits: int = 0
    children: List[_Node] = None
    untried: List[Move] = None

    def __post_init__(self):
        self.children = []
        self.untried = self.state.legal_moves()

    def ucb1(self, child: _Node, exploration: float) -> float:
        """
        Q = child.wins / child.visits
        U = exploration * sqrt(log(self.visits) / child.visits)
        Ritorna Q + U.
        """
        return (child.wins / child.visits) + exploration * math.sqrt(
            math.log(self.visits) / child.visits
        )

    def best_child(self, exploration: float) -> _Node:
        """
        Sceglie il figlio c che massimizza UCB1(c, exploration).
        """
        return max(self.children, key=lambda c: self.ucb1(c, exploration))

    def expand(self, rng: random.Random) -> _Node:
        """
        Espande un nodo non completamente esplorato:
        - Pesca casualmente una mossa da self.untried
        - Crea il nuovo stato e il nuovo nodo figlio
        - Azzera le liste di mosse non esplorate del figlio
        - Aggiunge il figlio a self.children e rimuove la mossa da self.untried
        """
        move = rng.choice(self.untried)
        self.untried.remove(move)
        next_state = self.state.next_state(move)
        child = _Node(next_state, parent=self, move=move)
        self.children.append(child)
        return child

    def backpropagate(self, value: float) -> None:
        """
        Retropropaga il valore stimato v fino alla radice:
        - Ad ogni livello incrementa visits
        - Aggiunge value a wins
        - Inverte il segno di value perché il turno alterna
        """
        node: Optional[_Node] = self
        while node is not None:
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
    ) -> None:
        """
        value_net: istanza di ValueNetwork già caricata su CPU/GPU
        iterations: numero di simulazioni per mossa
        exploration: fattore di esplorazione (tipicamente sqrt(2))
        """
        self.iterations = iterations
        self.C = exploration
        self.rng = rng or random.Random()
        self.root: Optional[_Node] = None

        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.value_net = value_net.to(device)
        self.value_net.eval()

    def search(self, root_state: MiniChessState, temperature: float = 0.8) -> Move:
        """
        Esegue MCTS a partire dallo stato root_state.
        Ogni chiamata ricrea la root da zero (nessun tree reuse).
        La scelta finale è stocastica, basata su visits^(1/temperature).
        Se temperature=0.0, torna la mossa con più visite (deterministico).
        """
        # 1) Creo sempre una nuova root
        self.root = _Node(root_state, parent=None, move=None)

        # 2) Eseguo le simulazioni
        for _ in range(self.iterations):
            node = self._select(self.root)
            value = self._simulate(node.state)
            node.backpropagate(value)

        # 3) Scelgo la mossa usando temperatura
        return self._select_move_with_temperature(temperature)

    def _select(self, node: _Node) -> _Node:
        """
        Selezione/Espansione:
        - Finché node non ha mosse non esplorate e ha figli, scendo sul best_child
        - Se node ha mosse in node.untried, lo espando e restituisco il nuovo figlio
        """
        while not node.untried and node.children:
            node = node.best_child(self.C)
        if node.untried:
            node = node.expand(self.rng)
        return node

    def _simulate(self, state: MiniChessState) -> float:
        if state.is_terminal():
            return float(state.result())

        state_tensor = encode_state_as_tensor(state).unsqueeze(0)
        device = next(self.value_net.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            raw_v = self.value_net(state_tensor)   # (1, 2) o (1, 1)

        # Se la rete è scalar-output, uso l’unico valore
        if raw_v.shape[-1] == 1:
            v = raw_v.item()
        else:
            v_white, v_black = raw_v[0]            # unpack
            v = v_white if state.current_player() == 1 else v_black
            v = v.item()

        # clamp in [-1, +1] (difesa da overshoot numerico)
        return max(-1.0, min(1.0, v))


    def _select_move_with_temperature(self, temperature: float) -> Move:
        """
        Costruisce una distribuzione di probabilità proporzionale a visits^(1/temperature)
        e campiona una mossa. Se temperature=0, restituisce la mossa con più visite.
        """
        children = self.root.children
        visits = [c.visits for c in children]
        if temperature == 0.0:
            best = max(children, key=lambda c: c.visits)
            return best.move

        # Calcolo pesi = visits^(1/temperature)
        weights = [v ** (1.0 / temperature) for v in visits]
        total = sum(weights)
        probs = [w / total for w in weights]
        chosen = self.rng.choices(children, weights=probs, k=1)[0]
        return chosen.move
