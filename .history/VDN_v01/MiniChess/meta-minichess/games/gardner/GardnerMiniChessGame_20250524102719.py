from __future__ import print_function
import sys
from games import Game
from games.gardner.GardnerMiniChessLogic import Board
import numpy as np
import time
import hashlib
import random
 
class GardnerMiniChessGame(Game):
    RECURSION_LIMIT = 1000
    def __init__(self, n=5):
        self.n = n
        self.setAllActions()
 
    def getInitBoard(self):
        sys.setrecursionlimit(GardnerMiniChessGame.RECURSION_LIMIT)
        b = Board(self.n,[
            
            [-Board.ROOK, -Board.KNIGHT, -Board.BISHOP, -Board.QUEEN, -Board.KING],
            #[-Board.BLANK,-Board.BLANK,-Board.KING,-Board.BLANK,-Board.BLANK],
            [-Board.PAWN]*5,
            [Board.BLANK]*5,
            [Board.PAWN]*5,
            #[Board.BLANK,Board.BLANK,Board.KING,Board.BLANK,Board.BLANK]
            [Board.ROOK, Board.KNIGHT, Board.BISHOP, Board.QUEEN, Board.KING],
        ])
        return b.pieces_without_padding()
 
    def setAllActions(self):
        """
        Costruisce action_to_id/id_to_action mappando
        tutte le mosse pseudo‐legali di ogni pezzo, per entrambi i colori,
        inclusi *tutti* i cattura‐pedone possibili.
        """
        self.action_to_id = {}
        self.id_to_action = {}
        id = 0

        empty = [[Board.BLANK]*self.n for _ in range(self.n)]

        for player in [Board.PLAYER1, Board.PLAYER2]:
            for piece in [Board.PAWN, Board.KNIGHT, Board.BISHOP, Board.ROOK, Board.QUEEN, Board.KING]:
                # 1) board in cui c'è solo quel pezzo
                for r in range(self.n):
                    for c in range(self.n):
                        b = Board(self.n, [row[:] for row in empty])
                        b.set(r, c, piece * player)
                        for (p, s, e) in b._get_pseudo_legal_moves(player):
                            key = f"{abs(p)}:{s}:{e}"
                            if key not in self.action_to_id:
                                self.action_to_id[key] = id
                                self.id_to_action[id] = (abs(p), s, e)
                                id += 1

                # 2) per il pedone, seminare *tutti* i possibili “nemici” su TUTTE le altre celle
                if piece == Board.PAWN:
                    for r in range(self.n):
                        for c in range(self.n):
                            # piazzo il nostro pedone
                            board_template = [row[:] for row in empty]
                            board_template[r][c] = Board.PAWN * player

                            # per ogni altra cella (in cui potrebbe esserci un nemico):
                            for rr in range(self.n):
                                for cc in range(self.n):
                                    if rr==r and cc==c: 
                                        continue
                                    # metto un pedone avversario
                                    board_template[rr][cc] = -Board.PAWN * player

                                    b = Board(self.n, [row[:] for row in board_template])
                                    for (p, s, e) in b._get_pseudo_legal_moves(player):
                                        key = f"{abs(p)}:{s}:{e}"
                                        if key not in self.action_to_id:
                                            self.action_to_id[key] = id
                                            self.id_to_action[id] = (abs(p), s, e)
                                            id += 1

                                    # tolgo il pedone nemico e riprovo
                                    board_template[rr][cc] = Board.BLANK

        # azione “PASS” per stallo
        self.action_to_id["PASS:0:0"] = id
        self.id_to_action[id] = None

 
    def getBoardSize(self):
        return (self.n, self.n)
 
    def getActionSize(self):
        return len(self.action_to_id)
 
    def getNextState(self, board, player, action):
        """
        Se l'azione è PASS (id_to_action[action] è None), 
        rimaniamo sullo stesso board e cambiamo solo il player.
        """
        move = self.id_to_action[action]
        if move is None:
            # pass move: solo cambio di turno
            return (board, -player)
        b = Board(self.n, board)
        b.execute_move(move, player)
        return (b.pieces_without_padding(), -player)

    def getValidMoves(self, board, player):
        valids = [0.0]*self.getActionSize()
        b = Board(self.n, board)
        legal = b.get_legal_moves(player)
        if not legal:
            # solo “pass move” disponibile
            valids[-1] = 1.0
            return np.array(valids)
        for (p, s, e) in legal:
            key = f"{p}:{s}:{e}"
            valids[self.action_to_id[key]] = 1.0
        return np.array(valids)

 
    def getRandomMove(self, board, player):
        b = Board(self.n, board)
        legal = b.get_legal_moves(player)
        if not legal:
            return self.getActionSize() - 1
        p, s, e = random.choice(legal)
        return self.action_to_id[f"{p}:{s}:{e}"]

    def getGreedyMove(self, board, player):
        b = Board(self.n, board)
        flat = [c for row in board for c in row]
        best = None
        best_score = float('inf')
        for p, s, e in b.get_legal_moves(player):
            target = flat[e]
            score = abs(target) if target != Board.BLANK else 0
            if score < best_score:
                best_score, best = score, (p, s, e)
        if best is None:
            return self.getActionSize() - 1
        p, s, e = best
        return self.action_to_id[f"{p}:{s}:{e}"]

 
    def getGameEnded(self, board, player):
        b = Board(self.n, board)
        if b.is_checkmate(player):   return -1   # se tocca a player e non può uscire dallo scacco
        if b.is_in_check(player):    return 0    # scacco ma non matto
        if not b.has_legal_moves(player):
            return 1e-4               # patta per stallo
        if b.is_insufficient_material():  
            return 1e-4
        return 0

 
    def getCanonicalForm(self, board, player):
        return [[j*player for j in i] for i in board]
 
    def getSymmetries(self, board, pi):
        return [(board, pi)]
 
    def stringRepresentation(self, board):
        return hashlib.md5(np.array_str(np.array(board)).encode('utf-8')).hexdigest()
 
    def display(self, board, player):
        Board(self.n, board).display(player)
 
def display(game, board, player):
    Board(game.n, board).display(player)