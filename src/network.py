
import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, board_size=9, action_size=9, hidden_size=128):
        super().__init__()
        self.fc1 = nn.Linear(board_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.policy_head = nn.Linear(hidden_size, action_size)
        self.value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        policy_logits = self.policy_head(x)
        value = torch.tanh(self.value_head(x))

        return policy_logits, value.squeeze(-1)
