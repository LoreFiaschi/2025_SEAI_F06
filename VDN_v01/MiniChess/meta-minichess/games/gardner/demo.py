# demo.py
import os
import argparse
import torch
import sys
# ── PATH SETUP ───────────────────────────────────────────────────────
this_dir: str = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(this_dir, "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from games.gardner.value_network import ValueNetwork
from games.gardner.mcts_pt import MCTS
from games.gardner.minichess_state import MiniChessState
from games.gardner.GardnerMiniChessLogic import Board
from config import DEVICE, HIDDEN_CHANNELS, iterations_MCTS

def load_model(path: str) -> ValueNetwork:
    net = ValueNetwork(hidden_channels=HIDDEN_CHANNELS, output_dim=1).to(DEVICE)
    net.load_state_dict(torch.load(path, map_location=DEVICE))
    net.eval()
    return net

def main(): #per eseguire: python demo.py --ckpt "percorso checkpoint"
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="path al file .pth del modello")
    args = p.parse_args()

    model = load_model(args.ckpt)
    mcts  = MCTS(model, iterations=iterations_MCTS, rng=None)

    # inizializza partita
    game = MiniChessState
    state = MiniChessState(Board(5, Board(5, [[]]).pieces_without_padding()), player=1, turns=0)
    # in realtà per init usa GardnerMiniChessGame
    from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
    gm = GardnerMiniChessGame()
    state = MiniChessState(gm.getInitBoard(), player=1, turns=0)

    while not state.is_terminal():
        # stampa board
        b = Board(5, [list(r) for r in state.board()])
        b.display(state.current_player())

        if state.current_player() == 1:
            # muove l'utente (bianco)
            moves = state.legal_moves()
            print("\nLe tue mosse:")
            for i, mv in enumerate(moves):
                print(f"  [{i}] {b.move_to_algebraic(mv)}")

            # gestione input sicura
            while True:
                choice = input("Scegli indice mossa: ").strip()
                try:
                    idx = int(choice)
                    if 0 <= idx < len(moves):
                        mv = moves[idx]
                        break
                    else:
                        print(f"Inserisci un numero tra 0 e {len(moves)-1}.")
                except ValueError:
                    print("Per favore inserisci un numero valido.")
        else:
            # muove l'AI (nero)
            mv = mcts.search(state, temperature=0.0)
            print(f"\nAI muove: {b.move_to_algebraic(mv)}")

        state = state.next_state(mv)
        print()

    # partita finit
    res = state.result()   # +1 win white, 0 draw, -1 lose
    if res > 0:
        print("Ha vinto il Bianco!")
    elif res < 0:
        print("Ha vinto il Nero!")
    else:
        print("Patta!")

if __name__ == "__main__":
    main()