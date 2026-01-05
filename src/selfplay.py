
import numpy as np
from src.mcts import MCTS, MCTSNode

def self_play(game, network, config):
    data = []
    state = game.initial_state()

    move_count = 0

    while not game.is_terminal(state):
        root = MCTSNode(state, prior=1.0)
        mcts = MCTS(game, network, config.C_PUCT)

        for _ in range(config.MCTS_SIMULATIONS):
            mcts.simulate(root)

        visits = np.zeros(9)
        for a, child in root.children.items():
            visits[a] = child.N

        policy = visits / visits.sum()
        data.append((state, policy))

        # action = np.random.choice(9, p=policy)
        if move_count < 5:
            action = np.random.choice(9, p=policy)
        else:
            action = np.argmax(policy)

        state = game.next_state(state, action)
        move_count += 1

    # z = game.outcome(state)

    # samples = []
    # for (s, p) in data:
    #     samples.append((s, p, z))
    #     z = -z

    # z_final = game.outcome(state)

    # samples = []
    # current_z = z_final
    # for (board, player), p in data:
    #     samples.append(((board, player), p, current_z))
    #     current_z = -current_z

    # z = game.outcome(state)

    # samples = []
    # for (board, player), policy in data:
    #     samples.append(((board, player), policy, z))

    z_final = game.outcome(state)

    samples = []
    for (board, player), policy in data:
        samples.append(((board, player), policy, z_final * player))



    return samples
