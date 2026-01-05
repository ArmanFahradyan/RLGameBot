
import torch
import torch.nn.functional as F

def train_step(network, optimizer, batch):
    states, policies, values = batch

    logits, preds = network(states)

    policy_loss = -torch.mean(
        torch.sum(policies * F.log_softmax(logits, dim=1), dim=1)
    )

    value_loss = torch.mean((values - preds) ** 2)

    loss = policy_loss + value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
