import torch

# ---------- Configurazione del gioco ----------

PIECE_VALUES = {
    100: 0.001,   # pedone
    280: 0.003,   # cavallo
    320: 0.003,   # alfiere
    479: 0.005,   # torre
    929: 0.011,   # regina
    60000: 0.0    # re (non conta ai fini del potenziale)
}
STEP_COST = 0.0001  # penalità al giocatore di turno per ogni mossa
ALPHA = 1.0         # scaling dello shaping
# ────────────────────────────────────────────────────────────────────────────

# ---------- Hyper‑parametri globali ----------
num_cycles        = 15          # cicli self‑play + training
arena_games       = 50         # partite deterministiche per l'arena
games_per_cycle   = 50          # partite self‑play per ciclo
max_buffer_size   = 10_000      # massimo numero di transizioni nel buffer
iterations_MCTS   = 500         # simulazioni MCTS per mossa

learning_rate   = 5e-3
weight_decay    = 0.0
batch_size      = 64
num_epochs      = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

INPUT_CHANNELS = 13  # 6 bianchi + 6 neri + turno = 13
HIDDEN_CHANNELS = 32  # numero di filtri del primo conv-layer

# ----MCTS Hyper-parameters----

PIECE_TO_IDX = {               # codici definiti in Board
    100: 0,    # Pawn
    280: 1,    # Knight
    320: 2,    # Bishop
    479: 3,    # Rook
    929: 4,    # Queen
    60000: 5,  # King
}
N_TYPES = 6                    # numero di tipi di pezzo
INPUT_CHANNELS = N_TYPES * 2 + 1   # 6 bianchi + 6 neri + turno = 13
# ------------------------------------------------------------------

# --- MiniChessState (o in un util condiviso) -----------------

MAX_TURNS = 200
DRAW_REPETITION = 3
REWARD_DRAW = 1e-4

# -------------------------------------------------------------------
