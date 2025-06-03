import os
import sys
import argparse
import torch
import pygame
import time

# ── PATH SETUP ───────────────────────────────────────────────────────
this_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessLogic import Board
from games.gardner.ChessFunctions import draw_board, BOARD_SIZE, TILE_SIZE
from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
from config import DEVICE, HIDDEN_CHANNELS, iterations_MCTS

def load_model(path: str) -> ValueNetwork:
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_white",  required=True, help="Checkpoint Bianco")
    p.add_argument("--ckpt_black",  required=True, help="Checkpoint Nero")
    p.add_argument("--delay",       type=float, default=0.5, help="Secondi di pausa tra le mosse")
    args = p.parse_args()

    # Carica modelli e MCTS
    model_w = load_model(args.ckpt_white)
    model_b = load_model(args.ckpt_black)
    mcts_w  = MCTS(model_w, iterations=iterations_MCTS)
    mcts_b  = MCTS(model_b, iterations=iterations_MCTS)

    # Pygame init
    pygame.init()
    window_size = TILE_SIZE * BOARD_SIZE
    screen = pygame.display.set_mode((window_size, window_size))
    pygame.display.set_caption("MiniChess AI vs AI")
    clock = pygame.time.Clock()

    # stato iniziale
    gm = GardnerMiniChessGame()
    state = MiniChessState(gm.getInitBoard(), player=1, turns=0)

    running = True
    while running and not state.is_terminal():
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

        # AI muove
        if state.current_player() == 1:
            mv = mcts_w.search(state, temperature=0.0)
        else:
            mv = mcts_b.search(state, temperature=0.0)

        state = state.next_state(mv)

        # Draw
        board_view = state.board() if state.current_player()==1 else state.board()[::-1]
        screen.fill((0,0,0))
        draw_board(screen, board_view, selected_sq=None)
        pygame.display.flip()

        time.sleep(args.delay)
        clock.tick(30)

    # risultato
    if state.is_terminal():
        res = state.result()
        msg = "Bianco vince!" if res==1 else "Nero vince!" if res==-1 else "Patta!"
        print(msg)
        # facciamo apparire il risultato sullo schermo
        font = pygame.font.SysFont(None, 48)
        text = font.render(msg, True, (255, 0, 0))
        screen.blit(text, text.get_rect(center=(window_size//2, window_size//2)))
        pygame.display.flip()
        time.sleep(3)

    pygame.quit()

if __name__ == "__main__":
    main()