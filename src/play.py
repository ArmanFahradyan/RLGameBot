
import numpy as np
import torch
import sys
import os

from src.game_factory import get_game_config
from src.network import PolicyValueNet
from src.mcts import MCTS, MCTSNode
from src.config import Config
from src.connect_four import ConnectFour
from src.game import TicTacToe


def print_board_tictactoe(board):
    """Print TicTacToe board"""
    symbols = {1: "X", -1: "O", 0: "."}
    print("\n  0 1 2")
    for i in range(0, 9, 3):
        row_num = i // 3
        print(f"{row_num} " + " ".join(symbols[board[i + j]] for j in range(3)))
    print()


def print_board_connect_four(board):
    """Print Connect Four board"""
    board_2d = board.reshape(ConnectFour.ROWS, ConnectFour.COLS)
    symbols = {1: "X", -1: "O", 0: "."}
    
    print("\n  0 1 2 3 4 5 6")
    print("  " + "-" * 13)
    for row in range(ConnectFour.ROWS):
        print(f"{row}|" + " ".join(symbols[board_2d[row, col]] for col in range(ConnectFour.COLS)) + "|")
    print("  " + "-" * 13)
    print()


def human_move(game, state, game_name):
    """Get move from human player"""
    legal = game.legal_actions(state)
    
    if game_name == "connect_four":
        prompt = f"Your move (columns {legal}): "
    else:
        prompt = f"Your move (positions {legal}): "
    
    while True:
        try:
            move = int(input(prompt))
            if move in legal:
                return move
        except ValueError:
            pass
        print("Invalid move, try again.")


def ai_move(game, network, state, game_config):
    """Get move from AI using MCTS"""
    root = MCTSNode(state, prior=1.0)
    mcts = MCTS(game, network, game_config.c_puct)

    for _ in range(game_config.mcts_simulations):
        mcts.simulate(root)

    # Choose most visited action
    best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
    return best_action


def main():
    # Prompt user to select game
    print("Available games:")
    print("  1. TicTacToe")
    print("  2. Connect Four")
    
    choice = input("\nSelect game (1 or 2): ").strip()
    
    if choice == "1":
        game_name = "tictactoe"
    elif choice == "2":
        game_name = "connect_four"
    else:
        print("Invalid choice. Exiting.")
        sys.exit(1)
    
    # Get game-specific configuration
    game_config = get_game_config(game_name)
    config = Config()
    
    # Check if model file exists
    if not os.path.exists(game_config.model_save_path):
        print(f"\nError: Model file '{game_config.model_save_path}' not found!")
        print(f"Please train the model first by running: python -m src.main")
        sys.exit(1)
    
    # Load network
    network = PolicyValueNet(
        board_size=game_config.board_size,
        action_size=game_config.action_size,
        hidden_size=game_config.hidden_size
    )
    network.load_state_dict(torch.load(game_config.model_save_path, map_location="cpu"))
    network.eval()

    print(f"\nWelcome to AlphaZero {game_config.game_name.upper()}!")
    choice = input("Do you want to play as X (first)? [y/n]: ").lower()

    human_player = 1 if choice == "y" else -1
    state = game_config.game.initial_state()

    # Print initial board
    if game_name == "tictactoe":
        print_board_tictactoe(state[0])
    else:
        print_board_connect_four(state[0])

    # Game loop
    while not game_config.game.is_terminal(state):
        board, player = state

        if player == human_player:
            action = human_move(game_config.game, state, game_name)
        else:
            print("AI is thinking...")
            action = ai_move(game_config.game, network, state, game_config)
            if game_name == "connect_four":
                print(f"AI plays column {action}")
            else:
                print(f"AI plays position {action}")

        state = game_config.game.next_state(state, action)
        
        # Print updated board
        if game_name == "tictactoe":
            print_board_tictactoe(state[0])
        else:
            print_board_connect_four(state[0])

    # Show result
    result = game_config.game.outcome(state)

    if result == 0:
        print("Game ended in a draw.")
    elif result == human_player:
        print("🎉 You win!")
    else:
        print("🤖 AI wins!")


if __name__ == "__main__":
    main()
