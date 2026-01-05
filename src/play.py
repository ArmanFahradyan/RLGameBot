
import numpy as np
import torch

from src.game import TicTacToe
from src.network import PolicyValueNet
from src.mcts import MCTS, MCTSNode
from src.config import Config


def print_board(board):
    symbols = {1: "X", -1: "O", 0: "."}
    for i in range(0, 9, 3):
        print(" ".join(symbols[board[i + j]] for j in range(3)))
    print()


def human_move(game, state):
    legal = game.legal_actions(state)
    while True:
        try:
            move = int(input(f"Your move {legal}: "))
            if move in legal:
                return move
        except ValueError:
            pass
        print("Invalid move, try again.")


def ai_move(game, network, state, config):
    root = MCTSNode(state, prior=1.0)
    mcts = MCTS(game, network, config.C_PUCT)

    for _ in range(config.MCTS_SIMULATIONS):
        mcts.simulate(root)

    # Choose most visited action (tau -> 0)
    best_action = max(root.children.items(), key=lambda x: x[1].N)[0]
    return best_action


def main():
    config = Config()
    game = TicTacToe()

    network = PolicyValueNet()
    network.load_state_dict(torch.load("tictactoe_az.pth", map_location="cpu"))
    network.eval()

    print("Welcome to AlphaZero Tic-Tac-Toe!")
    choice = input("Do you want to play as X (first)? [y/n]: ").lower()

    human_player = 1 if choice == "y" else -1
    state = game.initial_state()

    print_board(state[0])

    while not game.is_terminal(state):
        board, player = state

        if player == human_player:
            action = human_move(game, state)
        else:
            print("AI is thinking...")
            action = ai_move(game, network, state, config)

        state = game.next_state(state, action)
        print_board(state[0])

    result = game.outcome(state)

    if result == 0:
        print("Game ended in a draw.")
    elif result == human_player:
        print("You win!")
    else:
        print("AI wins.")


if __name__ == "__main__":
    main()
