
import numpy as np
from typing import List, Tuple

class ConnectFour:
    ROWS = 6
    COLS = 7
    BOARD_SIZE = ROWS * COLS  # 42
    ACTION_SIZE = COLS  # 7
    
    def initial_state(self):
        board = np.zeros(self.BOARD_SIZE, dtype=int)
        player = 1  # +1 = Player 1, -1 = Player 2
        return board, player
    
    def legal_actions(self, state) -> List[int]:
        """Returns list of columns that are not full"""
        board, _ = state
        board_2d = board.reshape(self.ROWS, self.COLS)
        # A column is legal if the top row has empty space
        return [col for col in range(self.COLS) if board_2d[0, col] == 0]
    
    def next_state(self, state, action: int):
        """Drop piece in column 'action'"""
        board, player = state
        board_2d = board.reshape(self.ROWS, self.COLS)
        
        # Find lowest empty row in this column
        for row in range(self.ROWS - 1, -1, -1):
            if board_2d[row, action] == 0:
                new_board = board.copy()
                new_board_2d = new_board.reshape(self.ROWS, self.COLS)
                new_board_2d[row, action] = player
                return new_board, -player
        
        # Should never reach here if action is legal
        raise ValueError(f"Column {action} is full!")
    
    def is_terminal(self, state) -> bool:
        board, _ = state
        board_2d = board.reshape(self.ROWS, self.COLS)
        
        # Check for winner
        if self._check_winner(board_2d) != 0:
            return True
        
        # Check for draw (board full)
        return not (board == 0).any()
    
    def outcome(self, state) -> int:
        """Returns winner from previous player's perspective"""
        board, player = state
        board_2d = board.reshape(self.ROWS, self.COLS)
        
        winner = self._check_winner(board_2d)
        if winner != 0:
            return -player  # winner is the previous player
        return 0  # draw
    
    def _check_winner(self, board_2d) -> int:
        """Check if there's a winner. Returns 1, -1, or 0"""
        
        # Check horizontal
        for row in range(self.ROWS):
            for col in range(self.COLS - 3):
                window = board_2d[row, col:col+4]
                if abs(window.sum()) == 4 and len(np.unique(window)) == 1:
                    return window[0]
        
        # Check vertical
        for row in range(self.ROWS - 3):
            for col in range(self.COLS):
                window = board_2d[row:row+4, col]
                if abs(window.sum()) == 4 and len(np.unique(window)) == 1:
                    return window[0]
        
        # Check diagonal (down-right)
        for row in range(self.ROWS - 3):
            for col in range(self.COLS - 3):
                window = [board_2d[row+i, col+i] for i in range(4)]
                if abs(sum(window)) == 4 and len(set(window)) == 1 and window[0] != 0:
                    return window[0]
        
        # Check diagonal (down-left)
        for row in range(self.ROWS - 3):
            for col in range(3, self.COLS):
                window = [board_2d[row+i, col-i] for i in range(4)]
                if abs(sum(window)) == 4 and len(set(window)) == 1 and window[0] != 0:
                    return window[0]
        
        return 0  # no winner

