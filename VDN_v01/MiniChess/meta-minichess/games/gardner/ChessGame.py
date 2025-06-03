import sys
import os
import pygame
import numpy as np

this_dir: str = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame

from ChessFunctions import *

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

def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
    pygame.display.set_caption("MiniChess Viewer")
    clock = pygame.time.Clock()

    game  = GardnerMiniChessGame()
    board = np.array(game.getInitBoard(), dtype=int)
    player = 1               # 1 = White, -1 = Black
    selected_sq = None       # coord. schermo (sr, sc)

    running = True
    while running:
        board_view = board if player == 1 else np.flipud(board)   # Bianco sempre in basso
        screen.fill(BLACK)
        draw_board(screen, board_view, selected_sq)
        pygame.display.flip()

        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False

            elif ev.type == pygame.MOUSEBUTTONDOWN:
                vc, vr = ev.pos[0] // TILE_SIZE, ev.pos[1] // TILE_SIZE   # col / row nel viewer

                # --- converte verso la matrice canonica ---
                bc = vc
                br = vr if player == 1 else BOARD_SIZE - 1 - vr
                raw = int(board[br, bc])

                # ---------- 1° click: selezione ----------
                if selected_sq is None:
                    if raw * player > 0:          # pezzo del lato al tratto
                        selected_sq = (vr, vc)

                # ---------- 2° click: tentativo di mossa ----------
                else:
                    vr0, vc0 = selected_sq
                    bc0 = vc0
                    br0 = vr0 if player == 1 else BOARD_SIZE - 1 - vr0

                    start_idx = (br0 + 2) * 7 + (bc0 + 1)
                    end_idx   = (br  + 2) * 7 + (bc  + 1)
                    key       = f"{abs(int(board[br0, bc0]))}:{start_idx}:{end_idx}"
                    
                    idx = game.action_to_id.get(key)

                    # posizione con il lato al tratto positivo (serve a BabyChessGame)
                    valid_board = board if player == 1 else -board
                    valid       = game.getValidMoves(valid_board.tolist(), player)

                    if idx is not None and valid[idx]:
                        tmp_board, _ = game.getNextState(board.tolist(), player, idx)
                        tmp_board = np.array(tmp_board, dtype=int)

                        if is_in_check(tmp_board, player):          # Re proprio sotto scacco
                            print("❌ Mossa illegale: il tuo Re resterebbe sotto scacco.")
                        else:
                            # ---- mossa applicata ----
                            board  = tmp_board
                            player = -player

                            in_check = is_in_check(board, player)   # Re avversario sotto scacco?

                            if in_check and not has_escape_moves(game, board, player):
                                print("♚  SCACCO MATTO!  Vince il", "Bianco" if -player==1 else "Nero")
                                running = False

                            elif in_check:
                                print("♔  SCACCO!")

                            elif not has_escape_moves(game, board, player):
                                print("½–½  PATTA per stallo.")
                                running = False

                            # cattura del Re (possibile in questa variante)
                            elif game.getGameEnded(board.tolist(), player) != 0:
                                print("✔️ Vittoria per cattura del Re!")
                                running = False
                    else:
                        print("❌ Mossa non valida.")

                    selected_sq = None

        clock.tick(30)

    pygame.quit()


if __name__ == "__main__":
    main()
