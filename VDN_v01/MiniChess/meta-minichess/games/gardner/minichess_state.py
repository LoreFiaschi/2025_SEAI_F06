from __future__ import annotations
from typing import List, Tuple, Optional
import os
import sys
# ────────────────────────────────────────────────────────────────────────────
# Path-hack: aggiunge due livelli sopra (la cartella che contiene "games/")
this_dir = os.path.dirname(__file__)                                  # .../games/gardner
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))    # .../meta-minichess
if project_root not in sys.path:
    sys.path.insert(0, project_root)
# ────────────────────────────────────────────────────────────────────────────

from games.gardner.GardnerMiniChessLogic import Board
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from config import MAX_TURNS, DRAW_REPETITION, REWARD_DRAW

Move = Tuple[int, int, int]      # (piece, src, dst)



class MiniChessState:
    """
    Wrapper leggerissimo su Board; fornisce la minima API necessaria a MCTS.
    È *immutabile* (board e player non cambiano dopo __init__), quindi hashable.
    """

    __slots__ = ("_board", "_player", "_hash", "_turns", "_history")

    N = 5                               # dimensione fissa in Gardner

    _game_singleton = GardnerMiniChessGame()     # riusa logica di fine‐partita

    def __init__(self, board, player=1, turns=0, history: Tuple[int, ...] = (),):
        # salva come tupla‑di‑tuple: immutabile & hashable
        self._board = tuple(tuple(r) for r in board)
        self._player = player
        self._hash = hash((self._board, self._player))
        self._turns = turns
        self._history = history + (self._hash,)

    def _as_lists(self):
        """Ritorna la board come list[list[int]] (deep‑copy leggera)."""
        return [list(r) for r in self._board]
    # ----------- API richiesta dall’MCTS ---------------------------------

    def current_player(self) -> int:
        return self._player

    def legal_moves(self):
        b = Board(self.N, self._as_lists())
        return b.get_legal_moves(self._player)

    def next_state(self, move):
        b = Board(self.N, self._as_lists())
        b.execute_move(move, self._player)
        return MiniChessState(b.pieces_without_padding(), -self._player,self._turns + 1, self._history)

    def is_terminal(self):
        return (self._is_threefold() or
                self._turns >= MAX_TURNS or
                self._game_singleton.getGameEnded(self._as_lists(), 
                                                  self._player) != 0)
    
    def result(self):         
        if self._turns >= MAX_TURNS:             
            return REWARD_DRAW  # patta   
        if self._is_threefold():
            #print(f"RIPETIZIONE MOSSE!")
            return REWARD_DRAW    
        r = self._game_singleton.getGameEnded(self._as_lists(), self._player) 
        if r == REWARD_DRAW:   
            return r             
        return -self._player # 1e-4 = patta, 0 = in corso, 1 = vittoria, -1 = sconfitta
    # ---------------- hashing & equality ----------------------------------

    def __hash__(self) -> int:
        if self._hash is None:
            # stringRepresentation() di Game è già MD5 ⇒ costante O(1)
            self._hash = hash((self._board, self._player))
        return self._hash

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MiniChessState):
            return False
        return self._player == other._player and self._board == other._board

    # ----------------- utilità opzionali ----------------------------------

    def board(self) -> Tuple[Tuple[int, ...], ...]:
        """Accesso read‑only allo scacchiere senza padding."""
        return self._board

    # per debug: str(state) mostra la scacchiera ruotata correttamente
    def __str__(self) -> str:
        b = Board(self.N, [list(r) for r in self._board])
        b.display(self._player)
        return ""
    
    def _is_threefold(self) -> bool:
        """True se la posizione corrente (board + side-to-move) è apparsa ≥ 3 volte."""
        return self._history.count(self._hash) >= DRAW_REPETITION
