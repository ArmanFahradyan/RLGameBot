
class Config:
    # Self-play
    NUM_SELF_PLAY_GAMES = 50
    MCTS_SIMULATIONS = 100
    TEMPERATURE = 1.0

    # Training
    BATCH_SIZE = 64
    EPOCHS_PER_ITER = 5
    LEARNING_RATE = 1e-3
    REPLAY_BUFFER_SIZE = 10_000
    MODEL_SAVE_PATH = "tictactoe_az.pth"

    # MCTS
    C_PUCT = 1.0

    # Training loop
    NUM_ITERATIONS = 50

    # Device
    DEVICE = "cpu"
