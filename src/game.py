
import numpy as np
from typing import List, Tuple

class TicTacToe:
    BOARD_SIZE = 9
    WINNING_LINES = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]

    def initial_state(self):
        board = np.zeros(self.BOARD_SIZE, dtype=int)
        player = 1  # +1 = X, -1 = O
        return board, player

    def legal_actions(self, state) -> List[int]:
        board, _ = state
        return [i for i in range(9) if board[i] == 0]

    def next_state(self, state, action: int):
        board, player = state
        next_board = board.copy()
        next_board[action] = player
        return next_board, -player

    def is_terminal(self, state) -> bool:
        board, _ = state
        for i,j,k in self.WINNING_LINES:
            if board[i] == board[j] == board[k] != 0:
                return True
        return not (board == 0).any()

    def outcome(self, state) -> int:
        board, player = state
        for i,j,k in self.WINNING_LINES:
            if board[i] == board[j] == board[k] != 0:
                return -player  # winner is previous player
        return 0
