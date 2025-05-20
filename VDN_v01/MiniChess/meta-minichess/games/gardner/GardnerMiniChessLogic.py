from __future__ import print_function
import random
import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple
import numpy as np
from enum import Enum
class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
 
class Board:

    FILES = "abcde"               # n = 5 costante in Gardner
    RANKS = "12345"
 
    PAWN, KNIGHT, BISHOP, ROOK, QUEEN, KING, BLANK, INF = 100, 280, 320, 479, 929, 60000, 0, 1000000
    PLAYER1, PLAYER2 = 1, -1
 
    def __init__(self, n, pieces):
        self.n = n
        self.last_cell, self.bottom_left, self.bottom_right = 62, 43, 47
        self.top_left, self.top_right = 15, 19
        self.pieces = self.add_padding(pieces)
        self.is_rotated, self.player_won = False, 0
        self.wc, self.bc, self.ep, self.kp = (True, True), (True, True), 0, 0
        self.north, self.east, self.south, self.west = -(self.n+2), 1, (self.n+2), -1
        self.directions = {
            # Pawn: un passo avanti + catture diagonali (no doppio passo)
            Board.PAWN: (self.north,self.north + self.west,self.north + self.east),            
            Board.KNIGHT: (-9, -15, -13, -5, 15, 13, 9, 5),
            Board.BISHOP: (8, 6, -8, -6),
            Board.ROOK: (-7, 1, 7, -1),
            Board.QUEEN: (8, 6, -8, -6, -7, 1, 7, -1),
            Board.KING: (7, -7, 6, 8, -6, -8, -1, 1)
        }
 
    def add_padding(self, board):
        pad = [Board.INF]*(self.n+2)
        return [pad, pad] + [[Board.INF]+row+[Board.INF] for row in board] + [pad, pad]
   
    def set(self, row, col, piece):
        self.pieces[row+2][col+1] = piece

    def greedy_move(self, player):
        flat = [i for r in self.pieces for i in r]
        moves = list(self._get_legal_moves(player))
        if not moves: return None
        return min(moves, key=lambda m: flat[m[2]] if flat[m[2]] < 0 else 0)
   
    def random_move(self, player):
        moves = list(self._get_legal_moves(player))
        if not moves:
            return None   # oppure un “pass move” se lo gestite
        return random.choice(moves)
    
    def _get_pseudo_legal_moves(self, player):
        flat = [p for row in self.pieces for p in row]
        N = len(flat)

        for i, p_val in enumerate(flat):
            if p_val == Board.BLANK or abs(p_val) == Board.INF or p_val * player < 0:
                continue

            piece = abs(p_val)

            if piece == Board.PAWN:
                # PEDONE: sempre "in su" nella matrice, perché rotate mette il player in basso
                forward = self.north

                # 1) avanzamento
                j = i + forward
                if 0 <= j < N and flat[j] == Board.BLANK:
                    yield (piece, i, j)

                # 2) catture diagonali
                for delta in (forward + self.west, forward + self.east):
                    j2 = i + delta
                    if 0 <= j2 < N and flat[j2] * player < 0 and abs(flat[j2]) != Board.INF:
                        yield (piece, i, j2)

            else:
                # sliding pieces, cavalli e king invariati
                for d in self.directions[piece]:
                    for j in count(i + d, d):
                        if not (0 <= j < N):
                            break
                        q = flat[j]
                        if abs(q) == Board.INF or (q * player > 0):
                            break
                        yield (piece, i, j)
                        if q != Board.BLANK or piece in (Board.KNIGHT, Board.KING):
                            break


    
    def get_legal_moves(self, player):
        """
        Filtra le pseudo-legali rimuovendo quelle che lasciano il re sotto scacco.
        """
        moves = []
        flat_orig = [p for row in self.pieces for p in row]
        for move in self._get_pseudo_legal_moves(player):
            flat_new = self._simulate_flat_move(flat_orig, move)

            # se il re sparisce → catturato
            king = Board.KING * player
            if king not in flat_new:
                continue

            # indice del re dopo la mossa
            kpos = flat_new.index(king)
            # scacco? se sì, scarta
            if self._attacked_in_flat(flat_new, -player, kpos):
                continue

            moves.append(move)
        return moves

    def _attacked_in_flat(self, flat, attacker, square):
        """
        Restituisce True se il giocatore `attacker` (±1) attacca la
        casa indicata da `square` nell’array piatto `flat`.
        """

        # --------------- SLIDING PIECES ---------------
        for piece, dirs in ((Board.ROOK,   self.directions[Board.ROOK]),
                            (Board.BISHOP, self.directions[Board.BISHOP]),
                            (Board.QUEEN,  self.directions[Board.QUEEN])):
            for d in dirs:
                for sq in count(square + d, d):
                    if abs(flat[sq]) == Board.INF:            # padding
                        break
                    if flat[sq] == Board.BLANK:               # casa vuota
                        continue
                    if flat[sq] * attacker > 0 and \
                       abs(flat[sq]) in (piece, Board.QUEEN): # colpito
                        return True
                    break                                     # qualsiasi pezzo blocca

        # ------------------- CAVALLO -------------------
        for d in self.directions[Board.KNIGHT]:
            sq = square + d
            if 0 <= sq < len(flat) and \
               flat[sq] * attacker > 0 and \
               abs(flat[sq]) == Board.KNIGHT:
                return True
            
        # ---------- RE (attacco a una casa di distanza) ----------
        for d in self.directions[Board.KING]:
            sq = square + d
            if 0 <= sq < len(flat) \
               and flat[sq] * attacker > 0 \
               and abs(flat[sq]) == Board.KING:
                return True
        # ------------------- PEDONE --------------------
        # Dopo ogni `rotate()` il side‑to‑move è in BASSO e avanza a NORD.
        # L’attaccante è quindi in ALTO: i suoi pedoni catturano verso SUD,
        # cioè il Re è minacciato se, guardando una casa, sopra di lui
        # (N‑O / N‑E) c’è un pedone dell’avversario.
        pawn_dirs = (self.north + self.west,
                     self.north + self.east)

        for d in pawn_dirs:
            sq = square + d
            if 0 <= sq < len(flat) and \
               flat[sq] * attacker > 0 and \
               abs(flat[sq]) == Board.PAWN:
                return True

        return False


    def rotate(self, board):
        self.is_rotated = not self.is_rotated
        self.pieces = np.array(board).reshape((self.n+4, self.n+2))[::-1]*(-1)
        self.ep = self.last_cell-self.ep if self.ep else 0
        self.kp = self.last_cell-self.kp if self.kp else 0
        return (self.pieces, self.bc, self.wc, self.ep, self.kp)
   
    def execute_move(self, move, player):
        """
        Esegue la mossa `move` di `player`,
        gestendo arrocco, promozione e en passant, e infine ruota la board.
        """
        piece, i, j = move
        flat = [p for row in self.pieces for p in row]
        p, q = flat[i], flat[j]

        def put(arr, idx, val):
            return arr[:idx] + [val] + arr[idx+1:]

        # 1) cattura re?
        if abs(q) == Board.KING:
            self.player_won = player

        # 2) muovi pezzo
        flat = put(flat, j, flat[i])
        flat = put(flat, i, Board.BLANK)

        # 3) arrocco (stessi indici e logica di prima)
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        if piece == Board.KING:
            wc = (False, False) if p > 0 else wc
            bc = (False, False) if p < 0 else bc
            if abs(j - i) == 2:
                # sposto anche la torre
                rook_from = self.bottom_left if j < i else self.bottom_right
                kp = (i + j) // 2
                flat = put(flat, rook_from, Board.BLANK)
                flat = put(flat, kp, Board.ROOK if p > 0 else -Board.ROOK)

        # 4) promozione pedone (sempre in top row prima del rotate)
        if abs(p) == Board.PAWN:
            if self.top_left <= j <= self.top_right:
                # p>0 => bianco, p<0 => nero
                flat = put(flat, j, Board.QUEEN if p > 0 else -Board.QUEEN)
            # en passant (salto diagonale su vuoto)
            delta = j - i
            if delta in (self.north + self.west, self.north + self.east) and q == Board.BLANK:
                flat = put(flat, j + (self.south if p>0 else self.north), Board.BLANK)

        # 5) aggiorna stati e ruota
        self.wc, self.bc, self.ep, self.kp = wc, bc, ep, kp
        
        '''
        print(self.move_to_algebraic(move,
                                   capture = (q != self.BLANK),
                                   check   = self.is_in_check(-player),
                                   mate    = self.is_checkmate(-player)))
        '''

        return self.rotate(flat)

   
    def has_legal_moves(self, player):
        return any(self.get_legal_moves(player))
   
    def is_win(self, player):
        flat = [i for r in self.pieces for i in r]
        if player == Board.PLAYER1 and (Board.PLAYER2*Board.KING not in flat): return True
        if player == Board.PLAYER2 and (Board.PLAYER1*Board.KING not in flat): return True
        return False
   
    def __getitem__(self, index):
        return self.pieces[index]
   
    def pieces_without_padding(self):
        result = []
        for row in self.pieces:
            clean_row = [a for a in row if abs(a) != Board.INF]
            if clean_row:
                if self.is_rotated: clean_row = [-x for x in clean_row]
                result.append(clean_row)
        return result
    
    def is_in_check(self, player):
        flat = [c for row in self.pieces for c in row]
        king = Board.KING * player
        if king not in flat: return False
        kpos = flat.index(king)
        return self._attacked_in_flat(flat, -player, kpos)

    def is_checkmate(self, player):
        if not self.is_in_check(player):
            return False
        return len(self.get_legal_moves(player)) == 0
    
    def is_insufficient_material(self):
        """
        Ritorna True se non esiste alcun modo realistico di dare matto.
        Viene considerato materiale insufficiente:
        - Re vs Re
        - Re + Cavallo vs Re
        - Re + Alfiere vs Re
        """
        pieces = [abs(p) for row in self.pieces for p in row if abs(p) != self.INF and p != self.BLANK]
        pieces = [p for p in pieces if p != self.KING]  # ignora i Re
        if not pieces:
            return True  # Solo Re contro Re
        if len(pieces) == 1 and pieces[0] in (self.KNIGHT, self.BISHOP):
            return True
        return False


    def _simulate_flat_move(self, flat, move):
        piece, src, dst = move
        new_flat = flat.copy()
        new_flat[dst] = new_flat[src]
        new_flat[src] = Board.BLANK
        return new_flat

   
    def display(self, player):
        # Pulisce il terminale (Windows vs Unix)
        import os
        #os.system('cls' if os.name == 'nt' else 'clear')

        uni = {
            Board.ROOK: '♜', Board.KNIGHT: '♞', Board.BISHOP: '♝',
            Board.QUEEN: '♛', Board.KING: '♚', Board.PAWN: '♟',
             -Board.ROOK: '♖',  -Board.KNIGHT: '♘',  -Board.BISHOP: '♗',
             -Board.QUEEN: '♕',  -Board.KING: '♔',  -Board.PAWN: '♙',
             Board.BLANK: '⊙'
        }

        board = self.pieces_without_padding()
        # Se tocca al nero, inverto l’orientamento della matrice
        if player < 0:
            board = board[::-1]

        # Stampo esattamente 5 righe
        for row in board:
            print(' '.join(uni[t] for t in row))

                # --- messaggi di “SCACCO!” ---------------------------------
        if self.is_in_check(player):          # il side‑to‑move è sotto scacco
            print(f"{bcolors.FAIL}♚ SCACCO al tuo Re!{bcolors.ENDC}")
        elif self.is_in_check(-player):       # il side‑to‑move dà scacco
            avv = "Bianco" if player < 0 else "Nero"
            print(f"{bcolors.OKGREEN}♚ SCACCO al Re {avv}!{bcolors.ENDC}")

        # ----------------------------------------------------------
        #  MAPPING  ➜  INDICE (array piatto con padding) → "e4", …
        # ----------------------------------------------------------


    def _flat_to_rc(self, idx):
            """Converte l’indice `idx` nell’array piatto *padded*
               in coordinate (row, col) 0‑based del vero 5×5."""
            cols = self.n + 2                # 7
            r, c = divmod(idx, cols)
            return r - 2, c - 1              # elimina i 2 pad-row e 1 pad‑col

    def idx_to_square(self, idx):
            """Restituisce la casella in notazione algebrica (es. 'e4')."""
            r, c = self._flat_to_rc(idx)
            if not (0 <= r < self.n and 0 <= c < self.n):
                return "??"
            # Rango 1 è il lato BIANCO (in basso nella configurazione iniziale)
            return f"{self.FILES[c]}{self.n - r}"

    def move_to_algebraic(self, move, capture=False, check=False, mate=False):
            """Converte una tripla (piece, src, dst) nella stringa algebrica.
               ▸ identifica automaticamente il pezzo (‘’, N, B, R, Q, K)
               ▸ mette ‘x’ se `capture` è True
               ▸ mette ‘+’ / ‘#’ se `check` / `mate` è True.
            """
            piece, src, dst = move
            letter = {self.KNIGHT: "N", self.BISHOP: "B",
                      self.ROOK: "R",  self.QUEEN:  "Q",
                      self.KING:  "K"}.get(piece, "")
            sep  = "x" if capture else "-"
            suf  = "#" if mate else ("+" if check else "")
            return f"{letter}{self.idx_to_square(src)}{sep}{self.idx_to_square(dst)}{suf}"
