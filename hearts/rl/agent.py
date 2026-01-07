import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class Experience:
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    action_mask: np.ndarray
    log_prob: float
    value: float


class RolloutBuffer:
    """Buffer to store experiences."""
    
    def __init__(self):
        self.clear()
    
    def clear(self):
        self.observations = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.action_masks = []
        self.log_probs = []
        self.values = []
    
    def add(self, exp: Experience):
        self.observations.append(exp.obs)
        self.actions.append(exp.action)
        self.rewards.append(exp.reward)
        self.dones.append(exp.done)
        self.action_masks.append(exp.action_mask)
        self.log_probs.append(exp.log_prob)
        self.values.append(exp.value)
    
    def __len__(self):
        return len(self.observations)


class ActorCritic(nn.Module):
    """Large Actor-Critic network with residual connections."""
    
    def __init__(self, obs_size: int, action_size: int, hidden_size: int = 1024):
        super().__init__()
        
        # Initial projection
        self.input_proj = nn.Linear(obs_size, hidden_size)
        
        # Deeper shared layers with residual connections
        # Deeper shared layers with residual connections
        self.shared_layers = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(0.1),
            ) for _ in range(3)
        ])
        
        self.final_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
        )
        
        # Actor head (policy)
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, action_size)
        )
        
        # Critic head (value)
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor, action_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initial projection
        features = self.input_proj(x)
        
        # Residual blocks
        for layer in self.shared_layers:
            features = features + layer(features)  # Residual connection
        
        features = self.final_proj(features)
        
        # Get logits and mask illegal actions
        logits = self.actor(features)
        logits = logits - (1 - action_mask) * 1e9
        
        value = self.critic(features)
        return logits, value


class PPOAgent:
    """PPO Agent with scheduling and larger batches."""
    
    def __init__(
        self,
        obs_size: int = 183,
        action_size: int = 52,
        hidden_size: int = 1024,
        lr: float = 1e-4,  # Lower learning rate for stability
        gamma: float = 0.99,  # Standard discount
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.15,
        entropy_coef: float = 0.1,  # High exploration for robustness
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,  # Allow larger gradients
        device: str = None
    ):
        self.obs_size = obs_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.initial_entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.lr = lr
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        self.network = ActorCritic(obs_size, action_size, hidden_size).to(self.device)
        self.optimizer = optim.AdamW(self.network.parameters(), lr=lr, weight_decay=1e-5, betas=(0.9, 0.999))
        self.scaler = GradScaler()
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3000, gamma=0.8)
        self.buffer = RolloutBuffer()
        
        self.update_count = 0
    
    def select_action(self, obs: np.ndarray, action_mask: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(action_mask).unsqueeze(0).to(self.device)
            
            logits, value = self.network(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), value.item()
    
    def select_action_batch(self, obs_batch: np.ndarray, mask_batch: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Select actions for a batch of observations."""
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs_batch).to(self.device)
            mask_t = torch.FloatTensor(mask_batch).to(self.device)
            
            logits, values = self.network(obs_t, mask_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
            return actions.cpu().numpy(), log_probs.cpu().numpy(), values.cpu().numpy()
    
    def get_value(self, obs: np.ndarray, mask: np.ndarray) -> float: #  mask parameter
        """Get value estimate for observation."""
        self.network.eval()
        with torch.no_grad():
            state = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            mask_t = torch.FloatTensor(mask).unsqueeze(0).to(self.device) # mask_t
            _, value = self.network(state, mask_t) # Pass mask_t to network
            return value.item()
    
    def compute_gae(self, rewards, values, next_value, dones):
        """Compute Generalized Advantage Estimation."""
        returns = []
        advantages = []
        gae = 0
        
        # Ensure values is a list or append correctly if numpy
        if isinstance(values, np.ndarray):
            values = np.append(values, next_value)
        else:
            values = values + [next_value]
        
        for i in reversed(range(len(rewards))):
            # Handle array of dones (VectorEnv) or single boolean (End of episode)
            if isinstance(dones, (list, np.ndarray)):
                mask = 1.0 - float(dones[i])
            else:
                mask = 1.0 if not (dones and i == len(rewards) - 1) else 0.0
            
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            gae = delta + self.gamma * self.gae_lambda * mask * gae
            advantages.insert(0, gae)
            returns.insert(0, gae + values[i])
            
        return returns, advantages
    
    def select_action_batch(self, states, masks):
        """Select actions for a batch of states (VectorEnv)."""
        self.network.eval()
        with torch.no_grad():
            states = torch.FloatTensor(np.array(states)).to(self.device)
            masks = torch.FloatTensor(np.array(masks)).to(self.device)
            
            logits, value = self.network(states, masks)
            
            # Mask invalid actions
            logits = logits.masked_fill(masks == 0, -1e9)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            
        return actions.cpu().numpy(), log_probs.cpu().numpy(), value.squeeze(-1).cpu().numpy()

    def get_value_batch(self, states, masks):
        """Get values for a batch of states."""
        self.network.eval()
        with torch.no_grad():
            states = torch.FloatTensor(np.array(states)).to(self.device)
            masks = torch.FloatTensor(np.array(masks)).to(self.device)
            _, values = self.network(states, masks)
        return values.squeeze(-1).cpu().numpy()

    def update(self, states, actions, old_log_probs, returns, advantages, masks, episode=0, max_episodes=30000):
        """Update policy using PPO loss."""
        self.network.train()
        
        # Linear entropy decay
        self.entropy_coef = self.initial_entropy_coef * (1 - (episode / max_episodes))
        self.entropy_coef = max(0.01, self.entropy_coef)  # Minimum entropy
        
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(np.array(actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(self.device)
        returns = torch.FloatTensor(np.array(returns)).to(self.device)
        advantages = torch.FloatTensor(np.array(advantages)).to(self.device)
        masks = torch.FloatTensor(np.array(masks)).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO Update Loop
        total_loss = 0
        n_updates = 4  # Less overfitting per batch
        batch_size = 128  # Larger batches for GPU
        indices = np.arange(len(states))
        
        for _ in range(n_updates):
            np.random.shuffle(indices)
            
            for start in range(0, len(states), batch_size):
                end = start + batch_size
                idx = indices[start:end]
                
                # Get new policy distribution
                with autocast():
                    logits, new_values = self.network(states[idx], masks[idx])
                    logits = logits.masked_fill(masks[idx] == 0, -1e9)
                    dist = torch.distributions.Categorical(logits=logits)
                    
                    new_log_probs = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()
                    
                    # Calculate ratio and surrogate loss
                    ratio = torch.exp(new_log_probs - old_log_probs[idx])
                    surr1 = ratio * advantages[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages[idx]
                    
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = 0.5 * ((new_values.squeeze() - returns[idx]) ** 2).mean()
                    
                    # Total loss with current entropy coefficient
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                total_loss += loss.item()
        
        self.scheduler.step()
        self.buffer.clear()
        n_batches = max(1, (len(states) // batch_size))
        
        # Log entropy for debugging
        # print(f"Entropy: {entropy.item():.4f} | Coef: {self.entropy_coef:.4f}")
        
        return {'loss': total_loss / (n_updates * n_batches), 'entropy': entropy.item()}
    
    def save(self, path: str):
        """Save model."""
        torch.save({
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'entropy_coef': self.entropy_coef,
            'update_count': self.update_count
        }, path)
    
    def load(self, path: str):
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'entropy_coef' in checkpoint:
            self.entropy_coef = checkpoint['entropy_coef']
        if 'update_count' in checkpoint:
            self.update_count = checkpoint['update_count']