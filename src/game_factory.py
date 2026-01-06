
from src.game import TicTacToe
from src.connect_four import ConnectFour

class GameConfig:
    """Game-specific configuration"""
    def __init__(self, game_name):
        if game_name == "tictactoe":
            self.game = TicTacToe()
            self.board_size = TicTacToe.BOARD_SIZE
            self.action_size = TicTacToe.BOARD_SIZE
            self.mcts_simulations = 100
            self.num_self_play_games = 50
            self.batch_size = 64
            self.epochs_per_iter = 5
            self.replay_buffer_size = 10_000
            self.num_iterations = 50
            self.c_puct = 1.0
            self.hidden_size = 64
            self.model_save_path = "tictactoe_az.pth"
            self.temperature_threshold = 5
            
        elif game_name == "connect_four":
            self.game = ConnectFour()
            self.board_size = ConnectFour.BOARD_SIZE
            self.action_size = ConnectFour.ACTION_SIZE
            self.mcts_simulations = 200
            self.num_self_play_games = 50
            self.batch_size = 128
            self.epochs_per_iter = 10
            self.replay_buffer_size = 20_000
            self.num_iterations = 200
            self.c_puct = 1.4
            self.hidden_size = 256  # Increased for better capacity
            self.model_save_path = "connect_four_az.pth"
            self.temperature_threshold = 10
            
        else:
            raise ValueError(f"Unknown game: {game_name}")
        
        self.game_name = game_name


def get_game_config(game_name):
    """Factory function to get game configuration"""
    return GameConfig(game_name)

