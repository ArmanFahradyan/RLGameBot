
import torch
import torch.optim as optim
import sys

from src.config import Config
from src.game_factory import get_game_config
from src.network import PolicyValueNet
from src.replay import ReplayBuffer
from src.selfplay import self_play
from src.train import train_step


def main():
    # Prompt user to select game
    print("Available games:")
    print("  1. TicTacToe")
    print("  2. Connect Four")
    print("  3. Othello")
    
    choice = input("\nSelect game (1, 2, or 3): ").strip()
    
    if choice == "1":
        game_name = "tictactoe"
    elif choice == "2":
        game_name = "connect_four"
    elif choice == "3":
        game_name = "othello"
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    # Get game-specific configuration
    game_config = get_game_config(game_name)
    config = Config()
    
    print(f"\n{'='*60}")
    print(f"Starting AlphaZero training for {game_config.game_name.upper()}")
    print(f"{'='*60}")
    print(f"Board size: {game_config.board_size}")
    print(f"Action size: {game_config.action_size}")
    print(f"MCTS simulations: {game_config.mcts_simulations}")
    print(f"Training iterations: {game_config.num_iterations}")
    print(f"Model will be saved to: {game_config.model_save_path}")
    print(f"{'='*60}\n")
    
    # Initialize network with game-specific dimensions
    network = PolicyValueNet(
        board_size=game_config.board_size,
        action_size=game_config.action_size,
        hidden_size=game_config.hidden_size
    ).to(config.DEVICE)
    
    optimizer = optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    replay_buffer = ReplayBuffer(game_config.replay_buffer_size)

    for iteration in range(game_config.num_iterations):
        print(f"\nIteration {iteration + 1}/{game_config.num_iterations}")

        # ---- SELF PLAY ----
        network.eval()
        for _ in range(game_config.num_self_play_games):
            samples = self_play(game_config.game, network, game_config, config)
            replay_buffer.add(samples)

        print(f"Replay buffer size: {len(replay_buffer)}")

        # ---- TRAINING ----
        if len(replay_buffer) < game_config.batch_size:
            continue

        network.train()
        for epoch in range(game_config.epochs_per_iter):
            batch = replay_buffer.sample(game_config.batch_size)
            loss = train_step(network, optimizer, batch, config.DEVICE)

        print(f"Training loss: {loss:.4f}")

    print("\nTraining complete.")
    torch.save(network.state_dict(), game_config.model_save_path)
    print(f"Model saved to {game_config.model_save_path}")


if __name__ == "__main__":
    main()
