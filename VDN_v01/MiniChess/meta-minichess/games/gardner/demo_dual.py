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

def main():  # per eseguire: python demo_dual.py --ckpt_white "checkpoint_bianco.pth" --ckpt_black "checkpoint_nero.pth"
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt_white", required=True, help="Path al file .pth del modello per il Bianco")
    p.add_argument("--ckpt_black", required=True, help="Path al file .pth del modello per il Nero")
    args = p.parse_args()

    model_white = load_model(args.ckpt_white)
    model_black = load_model(args.ckpt_black)
    mcts_white = MCTS(model_white, iterations=iterations_MCTS, rng=None)
    mcts_black = MCTS(model_black, iterations=iterations_MCTS, rng=None)

    # Inizializza partita
    from games.gardner.GardnerMiniChessGame import GardnerMiniChessGame
    gm = GardnerMiniChessGame()
    state = MiniChessState(gm.getInitBoard(), player=1, turns=0)

    while not state.is_terminal():
        # Stampa board
        b = Board(5, [list(r) for r in state.board()])
        b.display(state.current_player())

        if state.current_player() == 1:
            # Mossa del modello per il Bianco
            mv = mcts_white.search(state, temperature=0.0)
            print(f"\nModel Bianco muove: {b.move_to_algebraic(mv)}")
        else:
            # Mossa del modello per il Nero
            mv = mcts_black.search(state, temperature=0.0)
            print(f"\nModel Nero muove: {b.move_to_algebraic(mv)}")
        
        state = state.next_state(mv)
        print()

    # Partita finita: mostra il risultato
    res = state.result()   # +1: vittoria Bianco; 0: patta; -1: vittoria Nero
    if res == 1:
        print("Ha vinto il Bianco!")
    elif res == -1:
        print("Ha vinto il Nero!")
    else:
        print("Patta!")

if __name__ == "__main__":
    main()