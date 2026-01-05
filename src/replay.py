
import random
import torch

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []

    def add(self, samples):
        for s in samples:
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            self.buffer.append(s)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)

        states, policies, values = zip(*batch)

        boards = []
        for (board, player) in states:
            boards.append(board * player)

        states = torch.tensor(boards, dtype=torch.float32)
        policies = torch.tensor(policies, dtype=torch.float32)
        values = torch.tensor(values, dtype=torch.float32)

        return states, policies, values

    def __len__(self):
        return len(self.buffer)
