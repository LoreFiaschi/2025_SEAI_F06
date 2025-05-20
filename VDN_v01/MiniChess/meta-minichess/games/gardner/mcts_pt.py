from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Dict

from games.gardner.minichess_state import MiniChessState, Move
import torch
from games.gardner.value_network import ValueNetwork


def encode_state_as_tensor(state: MiniChessState) -> torch.Tensor:
    """
    Converte uno stato MiniChessState in un tensore (3, 5, 5):
      - canale 0: 1 se pezzo bianco, 0 altrimenti
      - canale 1: 1 se pezzo nero, 0 altrimenti
      - canale 2: valore +1 o -1 per il turno corrente
    Ritorna tensor su dispositivo CPU. Il batch dimension verrà aggiunto da MCTS.
    """
    board = state.board()           # tupla-di-tupla 5×5 di int (codici pezzi)
    player = state.current_player() # +1 o -1

    t = torch.zeros(3, 5, 5, dtype=torch.float32)
    for i in range(5):
        for j in range(5):
            p = board[i][j]
            if p > 0:
                t[0, i, j] = 1.0   # pezzo bianco
            elif p < 0:
                t[1, i, j] = 1.0   # pezzo nero
            # se p==0, rimane 0 (cella vuota)
    t[2, :, :] = float(player)      # canale “turno”
    return t


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

    def search(self, root_state: MiniChessState, temperature: float = 0.78) -> Move:
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
        """
        Valuta uno stato utilizzando la rete di valore:
        - Se lo stato terminale, restituisce state.result() (±1 o 0)
        - Altrimenti converte lo stato in (3,5,5) e chiama la rete CNN
        - Clampa l'output in [-1, +1]
        """
        if state.is_terminal():
            return float(state.result())

        # Costruisco (1,3,5,5) per la CNN
        state_tensor = encode_state_as_tensor(state).unsqueeze(0)  # (1,3,5,5)
        device = next(self.value_net.parameters()).device
        state_tensor = state_tensor.to(device)

        with torch.no_grad():
            raw_v = self.value_net(state_tensor)  # (1,1)
        v = raw_v.item()

        # Clampo in [-1, +1]
        if v > 1.0:
            return 1.0
        if v < -1.0:
            return -1.0
        return v

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
