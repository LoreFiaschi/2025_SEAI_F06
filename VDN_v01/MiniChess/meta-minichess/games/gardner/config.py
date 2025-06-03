import torch

# ---------- Configurazione del gioco ----------

PIECE_VALUES = {
    100: 1,   # pedone
    280: 3,   # cavallo
    320: 3,   # alfiere
    479: 5,   # torre
    929: 11,   # regina
    60000: 0.0    # re (non conta ai fini del potenziale)
}
STEP_COST = 1e-4  # penalità al giocatore di turno per ogni mossa
ALPHA = 0.1        # scaling dello shaping
# ────────────────────────────────────────────────────────────────────────────

# ---------- Hyper‑parametri globali ----------
num_cycles        = 50          # cicli self‑play + training
arena_games       = 30         # partite deterministiche per l'arena
games_per_cycle   = 50         # partite self‑play per ciclo
max_buffer_size   = 16000      # massimo numero di transizioni nel buffer
iterations_MCTS   = 200         # simulazioni MCTS per mossa

learning_rate   = 1e-3
weight_decay    = 1e-6
batch_size      = 64
num_epochs      = 3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

HIDDEN_CHANNELS = 256  # numero di filtri del primo conv-layer

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
INPUT_CHANNELS = N_TYPES * 2   # 6 bianchi + 6 neri
# ------------------------------------------------------------------

# --- MiniChessState (o in un util condiviso) -----------------

MAX_TURNS = 150
DRAW_REPETITION = 3
REWARD_DRAW = 1e-4

# -------------------------------------------------------------------

GAMMA = 0.95 # fattore di sconto per il futuro