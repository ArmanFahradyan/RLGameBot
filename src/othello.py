
import numpy as np
from typing import List, Tuple

class Othello:
    """Othello (Reversi) game implementation."""
    ROWS = 8
    COLS = 8
    BOARD_SIZE = ROWS * COLS  # 64
    ACTION_SIZE = BOARD_SIZE + 1  # 64 positions + 1 pass action
    
    # Directions for flipping: (row_delta, col_delta)
    DIRECTIONS = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    def initial_state(self):
        """Initialize board with starting position."""
        board = np.zeros(self.BOARD_SIZE, dtype=int)
        board_2d = board.reshape(self.ROWS, self.COLS)
        
        # Standard Othello starting position
        board_2d[3, 3] = -1  # White
        board_2d[3, 4] = 1   # Black
        board_2d[4, 3] = 1   # Black
        board_2d[4, 4] = -1  # White
        
        player = 1  # Black moves first
        return board, player
    
    def legal_actions(self, state) -> List[int]:
        """Returns list of legal moves (including pass if no moves available)."""
        board, player = state
        board_2d = board.reshape(self.ROWS, self.COLS)
        
        moves = []
        for row in range(self.ROWS):
            for col in range(self.COLS):
                if board_2d[row, col] == 0:
                    if self._would_flip(board_2d, row, col, player):
                        moves.append(row * self.COLS + col)
        
        # If no moves, must pass (action 64)
        if not moves:
            moves = [self.BOARD_SIZE]
        
        return moves
    
    def _would_flip(self, board_2d, row: int, col: int, player: int) -> bool:
        """Check if placing at (row, col) would flip any opponent pieces."""
        for dr, dc in self.DIRECTIONS:
            if self._count_flips_in_direction(board_2d, row, col, dr, dc, player) > 0:
                return True
        return False
    
    def _count_flips_in_direction(self, board_2d, row: int, col: int, dr: int, dc: int, player: int) -> int:
        """Count how many pieces would be flipped in a given direction."""
        r, c = row + dr, col + dc
        count = 0
        
        # Move in direction while finding opponent pieces
        while 0 <= r < self.ROWS and 0 <= c < self.COLS:
            if board_2d[r, c] == -player:
                count += 1
                r += dr
                c += dc
            elif board_2d[r, c] == player:
                return count  # Found our piece, flips are valid
            else:
                return 0  # Empty space, no flips
        
        return 0  # Reached edge without finding our piece
    
    def next_state(self, state, action: int):
        """Place piece and flip opponent pieces."""
        board, player = state
        new_board = board.copy()
        
        # Pass action
        if action == self.BOARD_SIZE:
            return new_board, -player
        
        board_2d = new_board.reshape(self.ROWS, self.COLS)
        row, col = action // self.COLS, action % self.COLS
        
        # Place the piece
        board_2d[row, col] = player
        
        # Flip pieces in all directions
        for dr, dc in self.DIRECTIONS:
            flips = self._count_flips_in_direction(board_2d, row, col, dr, dc, player)
            if flips > 0:
                r, c = row + dr, col + dc
                for _ in range(flips):
                    board_2d[r, c] = player
                    r += dr
                    c += dc
        
        return new_board, -player
    
    def is_terminal(self, state) -> bool:
        """Game ends when neither player can move."""
        board, player = state
        
        # Check if current player can move
        current_moves = self.legal_actions(state)
        if current_moves != [self.BOARD_SIZE]:
            return False
        
        # Check if opponent can move
        opponent_moves = self.legal_actions((board, -player))
        if opponent_moves != [self.BOARD_SIZE]:
            return False
        
        # Neither can move, game over
        return True
    
    def outcome(self, state) -> int:
        """Returns winner from current player's perspective."""
        board, player = state
        
        black_count = np.sum(board == 1)
        white_count = np.sum(board == -1)
        
        if black_count > white_count:
            winner = 1
        elif white_count > black_count:
            winner = -1
        else:
            return 0  # Draw
        
        # Return from current player's perspective
        return winner * player
    
    def get_score(self, state) -> Tuple[int, int]:
        """Returns (black_count, white_count) for display."""
        board, _ = state
        black = np.sum(board == 1)
        white = np.sum(board == -1)
        return black, white
