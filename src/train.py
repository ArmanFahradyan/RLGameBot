
import torch
import torch.nn.functional as F

def train_step(network, optimizer, batch, device='cpu'):
    states, policies, values = batch
    
    # Move batch to same device as network
    states = states.to(device)
    policies = policies.to(device)
    values = values.to(device)

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
