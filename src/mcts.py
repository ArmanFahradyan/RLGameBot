
import math
import numpy as np
import torch

class MCTSNode:
    def __init__(self, state, prior):
        self.state = state
        self.prior = prior
        self.children = {}
        self.N = 0
        self.W = 0.0

    @property
    def Q(self):
        return 0 if self.N == 0 else self.W / self.N


class MCTS:
    def __init__(self, game, network, c_puct=1.0):
        self.game = game
        self.network = network
        self.c_puct = c_puct

    def select(self, node):
        best_score = -float("inf")
        best_action = None

        for action, child in node.children.items():
            u = (
                child.Q
                + self.c_puct
                * child.prior
                * math.sqrt(node.N) / (1 + child.N)
            )
            if u > best_score:
                best_score = u
                best_action = action

        return best_action
    
    def expand(self, node):
        board, player = node.state
        x = torch.tensor(board * player, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            logits, value = self.network(x)
            policy = torch.softmax(logits, dim=1).squeeze(0).numpy()

        # legal = self.game.legal_actions(node.state)
        # for a in legal:
        #     node.children[a] = MCTSNode(
        #         self.game.next_state(node.state, a),
        #         prior=policy[a]
        #     )

        legal = self.game.legal_actions(node.state)
        policy_sum = sum(policy[a] for a in legal)

        for a in legal:
            node.children[a] = MCTSNode(
                self.game.next_state(node.state, a),
                prior=policy[a] / policy_sum
            )

        return value.item()

    # def backup(self, path, value):
    #     for node in reversed(path):
    #         node.N += 1
    #         node.W += value
    #         value = -value

    def backup(self, path, value):
        for node in reversed(path):
            node.N += 1
            node.W += value


    def simulate(self, root):
        node = root
        path = [node]

        while node.children:
            action = self.select(node)
            node = node.children[action]
            path.append(node)

        # if self.game.is_terminal(node.state):
        #     value = self.game.outcome(node.state)
        if self.game.is_terminal(node.state):
            board, player = node.state
            value = self.game.outcome(node.state)
            value *= player
        else:
            value = self.expand(node)

        self.backup(path, value)


