
import numpy as np
from src.mcts import MCTS, MCTSNode

def self_play(game, network, game_config, config):
    """
    Run a single self-play game.
    
    Args:
        game: Game instance
        network: Neural network
        game_config: Game-specific configuration (GameConfig object)
        config: General training configuration (Config object)
    """
    data = []
    state = game.initial_state()
    move_count = 0

    while not game.is_terminal(state):
        root = MCTSNode(state, prior=1.0)
        mcts = MCTS(game, network, game_config.c_puct)

        for _ in range(game_config.mcts_simulations):
            mcts.simulate(root)

        # Create visit count array dynamically based on action size
        visits = np.zeros(game_config.action_size)
        for a, child in root.children.items():
            visits[a] = child.N

        policy = visits / visits.sum()
        data.append((state, policy))

        # Temperature-based action selection
        if move_count < game_config.temperature_threshold:
            action = np.random.choice(game_config.action_size, p=policy)
        else:
            action = np.argmax(policy)

        state = game.next_state(state, action)
        move_count += 1

    # Get final outcome and assign values
    z_final = game.outcome(state)

    samples = []
    for (board, player), policy in data:
        value = z_final * player
        samples.append(((board, player), policy, value))

    return samples
