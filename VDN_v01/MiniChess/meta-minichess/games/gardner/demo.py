# demo.py
import os
import sys
import argparse
import torch
import pygame
import numpy as np
import time

# ── PATH SETUP ───────────────────────────────────────────────────────
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from games.gardner.GardnerMiniChessLogic import Board
from games.gardner.ChessFunctions import (
    draw_board, BOARD_SIZE, TILE_SIZE,
    is_in_check, has_escape_moves,
    state_is_terminal       # ← importiamo il nuovo helper
)
from config import DEVICE, HIDDEN_CHANNELS, iterations_MCTS

def load_model(path: str) -> ValueNetwork:
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True,      help="Checkpoint modello (.pth)")
    p.add_argument("--delay", type=float, default=0.5, help="Pausa tra le mosse AI")
    args = p.parse_args()

    # carica modello e MCTS
    model = load_model(args.ckpt)
    mcts  = MCTS(model, iterations=iterations_MCTS)

    # init Pygame
    pygame.init()
    window_size = TILE_SIZE * BOARD_SIZE
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("MiniChess: Tu vs AI")
    clock = pygame.time.Clock()

    # inizializza stato di gioco
    gm = GardnerMiniChessGame()
    state = MiniChessState(gm.getInitBoard(), player=1, turns=0)
    selected_sq = None

    # usiamo state_is_terminal nello while
    running = True
    while running and not state_is_terminal(state):
        board_np   = np.array(state.board(), dtype=int)
        board_view = board_np if state.current_player() == 1 else board_np[::-1]

        screen.fill((0,0,0))
        draw_board(screen, board_view, selected_sq)
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            # il tuo turno: gestione click
            if state.current_player() == 1 and ev.type == pygame.MOUSEBUTTONDOWN:
                mx, my = ev.pos
                col = mx // TILE_SIZE
                row = my // TILE_SIZE
                # coord su board_np
                br = row
                bc = col

                # primo click: selezione pezzo
                if selected_sq is None:
                    if board_np[br, bc] * state.current_player() > 0:
                        selected_sq = (br, bc)
                # secondo click: destinazione
                else:
                    br0, bc0 = selected_sq
                    # costruisci chiave azione come in ChessGame.py
                    start_idx = (br0 + 2) * 7 + (bc0 + 1)
                    end_idx   = (br  + 2) * 7 + (bc  + 1)
                    key = f"{abs(int(board_np[br0, bc0]))}:{start_idx}:{end_idx}"
                    idx = gm.action_to_id.get(key)
                    valid = gm.getValidMoves(board_np.tolist(), state.current_player())

                    if idx is not None and valid[idx]:
                        # applica mossa
                        nb, _ = gm.getNextState(board_np.tolist(), state.current_player(), idx)
                        nb = np.array(nb, dtype=int)
                        if not is_in_check(nb, state.current_player()):
                            state = MiniChessState(nb.tolist(), player=-1, turns=state._turns+1)
                    selected_sq = None

        # turno AI
        if running and state.current_player() == -1 and not state_is_terminal(state):
            # controlla mosse legali prima di invocare MCTS
            legal = state.legal_moves()
            if not legal:
                print("⚠️  AI non ha mosse legali, termino la partita.")
                break

            mv = mcts.search(state, temperature=0.0)
            if mv is None:
                print("⚠️  MCTS non ha restituito mossa, termino.")
                break

            state = state.next_state(mv)
            time.sleep(args.delay)

        clock.tick(30)

    # risultato usando ancora state_is_terminal se vuoi
    res = state.result()
    msg = "Hai vinto!" if res == 1 else "Hai perso!" if res == -1 else "Patta!"
    print(msg)
    font = pygame.font.SysFont(None, 48)
    surf = font.render(msg, True, (255,0,0))
    screen.blit(surf, surf.get_rect(center=(window_size//2, window_size//2)))
    pygame.display.flip()
    time.sleep(2)
    pygame.quit()

if __name__ == "__main__":
    main()