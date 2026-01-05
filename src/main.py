
import torch
import torch.optim as optim

from src.config import Config
from src.game import TicTacToe
from src.network import PolicyValueNet
from src.replay import ReplayBuffer
from src.selfplay import self_play
from src.train import train_step


def main():
    config = Config()

    game = TicTacToe()
    network = PolicyValueNet().to(config.DEVICE)
    optimizer = optim.Adam(network.parameters(), lr=config.LEARNING_RATE)
    replay_buffer = ReplayBuffer(config.REPLAY_BUFFER_SIZE)

    print("Starting AlphaZero Tic-Tac-Toe training")

    for iteration in range(config.NUM_ITERATIONS):
        print(f"\nIteration {iteration + 1}/{config.NUM_ITERATIONS}")

        # ---- SELF PLAY ----
        network.eval()
        for _ in range(config.NUM_SELF_PLAY_GAMES):
            samples = self_play(game, network, config)
            replay_buffer.add(samples)

        print(f"Replay buffer size: {len(replay_buffer)}")

        # ---- TRAINING ----
        if len(replay_buffer) < config.BATCH_SIZE:
            continue

        network.train()
        for epoch in range(config.EPOCHS_PER_ITER):
            batch = replay_buffer.sample(config.BATCH_SIZE)
            loss = train_step(network, optimizer, batch)

        print(f"Training loss: {loss:.4f}")

    print("\nTraining complete.")
    torch.save(network.state_dict(), config.MODEL_SAVE_PATH)
    print(f"Model saved to {config.MODEL_SAVE_PATH}")


if __name__ == "__main__":
    main()
