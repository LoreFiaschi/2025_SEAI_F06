# -*- coding: utf-8 -*-
import sys
import os
import pygame
import numpy as np    # ← aggiunto
from games.gardner.minichess_state import MiniChessState

this_dir: str = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- PARAMETRI GRAFICI
TILE_SIZE   = 80
BOARD_SIZE  = 5
WINDOW_SIZE = TILE_SIZE * BOARD_SIZE

WHITE     = (255, 255, 255)
GRAY      = (170, 170, 170)
BLACK     = (0,   0,   0)
HIGHLIGHT = (255, 215, 0)

# --- mappa pezzi → unicode (raw definiti in BabyChessGame)
PIECE_UNICODE = {
     100: "♙",  280: "♘", 320: "♗", 479: "♖", 929: "♕", 60000: "♔",
    -100: "♟", -280: "♞", -320: "♝", -479: "♜", -929: "♛", -60000:"♚",
       0: ""
}

def is_in_check(board, color):
    """
    board : np.ndarray 5x5 con Bianco > 0, Nero < 0
    color : 1 (bianco) o -1 (nero) → Re da controllare
    """
    king_val = 60000 * color
    kr = kc = None
    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if board[r, c] == king_val:
                kr, kc = r, c
                break
        if kr is not None:
            break
    if kr is None:      # (Re catturato: tecnicamente è già matto)
        return True

    opp = -color

    # ---- pedoni
    pawn_dirs = [(-1, -1), (-1, 1)] if color == 1 else [(1, -1), (1, 1)]
    for dr, dc in pawn_dirs:
        r, c = kr + dr, kc + dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == opp * 100:
            return True

    # ---- cavalli
    N_MOVES = [(2, 1), (1, 2), (-1, 2), (-2, 1),
               (-2, -1), (-1, -2), (1, -2), (2, -1)]
    for dr, dc in N_MOVES:
        r, c = kr + dr, kc + dc
        if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and abs(board[r, c]) == 280 and board[r, c] * opp > 0:
            return True

    dirs_bishop = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    dirs_rook   = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # alfiere / donna diagonale
    for dr, dc in dirs_bishop:
        r, c = kr + dr, kc + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            piece = board[r, c]
            if piece != 0:
                if piece * opp > 0 and (abs(piece) in (320, 929)):   # ♝ o ♛
                    return True
                break
            r += dr; c += dc

    # torre / donna ortogonale
    for dr, dc in dirs_rook:
        r, c = kr + dr, kc + dc
        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
            piece = board[r, c]
            if piece != 0:
                if piece * opp > 0 and (abs(piece) in (479, 929)):   # ♜ o ♛
                    return True
                break
            r += dr; c += dc

    # ---- Re avversario adiacente (caso raro ma incluso)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == dc == 0:
                continue
            r, c = kr + dr, kc + dc
            if 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE and board[r, c] == opp * 60000:
                return True

    return False

def has_escape_moves(game, board, player):
    """
    True se il giocatore `player` dispone di almeno una mossa legale
    che lo tolga dallo scacco.
    """
    valids = game.getValidMoves(board.tolist(), player)

    for idx, v in enumerate(valids):
        if not v:
            continue
        tmp_board, _ = game.getNextState(board.tolist(), player, idx)
        tmp_board = np.array(tmp_board, dtype=int)
        if not is_in_check(tmp_board, player):
            return True
    return False

def draw_board(screen, board_view, selected_sq=None):
    """
    board_view: np.ndarray 5×5 già orientata con il Bianco in basso
    selected_sq: (row, col) coordinate schermo
    """
    board_arr = np.array(board_view, dtype=int)

    # usa un font che supporta i simboli degli scacchi
    font = pygame.font.SysFont("Segoe UI Symbol", TILE_SIZE - 10)

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            # disegna sfondo
            rect = pygame.Rect(c*TILE_SIZE, r*TILE_SIZE, TILE_SIZE, TILE_SIZE)
            color = WHITE if (r+c)%2==0 else GRAY
            pygame.draw.rect(screen, color, rect)
            if selected_sq == (r, c):
                pygame.draw.rect(screen, HIGHLIGHT, rect, 4)

            # disegna il pezzo come emoji/unicode glyph
            raw  = int(board_arr[r, c])
            char = PIECE_UNICODE.get(raw, "")
            if char:
                surf = font.render(char, True, BLACK)
                screen.blit(surf, surf.get_rect(center=rect.center))

def state_is_terminal(state: MiniChessState) -> bool:
    """
    Wrapper per controllare se lo stato corrente è terminale.
    """
    return state.is_terminal()