import torch

# =============================================================================
#   MiniChess Configuration
# =============================================================================

# Piece values used to compute the “material potential” φ(s).
# Keys are the piece identifiers (as in GardnerMiniChessGame), values are
# the signed contribution to the potential (positive for White, negative for Black).
PIECE_VALUES = {
    100:    1,    # pawn
    280:    3,    # knight
    320:    3,    # bishop
    479:    4,    # rook
    929:    7,    # queen
    60000:  0.0,  # king (ignored for potential-based shaping)
}

# Mapping from piece identifiers → channel index in the tensor encoding.
# Used by encode_state_as_tensor to place each piece on one of the 6 feature planes.
PIECE_TO_IDX = {
    100:    0,  # Pawn
    280:    1,  # Knight
    320:    2,  # Bishop
    479:    3,  # Rook
    929:    4,  # Queen
    60000:  5,  # King
}

# Maximum number of plies (half-moves) before forcing a draw.
MAX_TURNS = 150

# Number of repetitions of the same board state to declare a draw.
DRAW_REPETITION = 3


# =============================================================================
#   Reward Shaping Parameters (MCTS + Self-Play)
# =============================================================================

# Small penalty per move to encourage shorter games.
STEP_COST = 1e-5  

# Scaling factor α in the potential-based reward shaping:
#    r_shaping = α [ γ·φ(s') − φ(s) ]
ALPHA = 0.08        

# Reward assigned for a drawn game in self-play / arena matches.
REWARD_DRAW = 1e-4

# Discount factor γ applied to future shaping rewards.
GAMMA = 0.95        


# =============================================================================
#   Global Hyperparameters for Training & Evaluation
# =============================================================================

# Number of self-play + training cycles before termination.
num_cycles = 50            

# Number of deterministic arena games to evaluate promotion.
arena_games = 50           

# Number of self-play games generated per cycle.
games_per_cycle = 500       

# Maximum size of the replay buffer (oldest transitions are discarded).
max_buffer_size = 16000    

# Number of MCTS simulations per move.
iterations_MCTS = 800       

# Learning rate and weight decay for the AdamW optimizer.
learning_rate = 1e-3       
weight_decay    = 1e-6     

# Training batch size (number of (state, return) pairs per update).
batch_size = 256           

# Number of epochs to train the value network on the current replay buffer.
num_epochs = 10            

# Device selection: "cuda" if GPU is available, else "cpu".
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =============================================================================
#   Neural Network Architecture Parameters
# =============================================================================

# Number of feature planes in the first convolutional layer.
# We use one plane per piece-type-per-color: N_TYPES white + N_TYPES black.
N_TYPES        = 6             
INPUT_CHANNELS = N_TYPES * 2    

# Number of filters (output channels) in the first conv layer.
HIDDEN_CHANNELS = 256

