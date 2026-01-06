
import random
import torch
import numpy as np

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

        # More efficient: convert to numpy array first, then to tensor
        boards = np.array([board * player for (board, player) in states], dtype=np.float32)
        policies_np = np.array(policies, dtype=np.float32)
        values_np = np.array(values, dtype=np.float32)

        states = torch.from_numpy(boards)
        policies = torch.from_numpy(policies_np)
        values = torch.from_numpy(values_np)

        return states, policies, values

    def __len__(self):
        return len(self.buffer)
