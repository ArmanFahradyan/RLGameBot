import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from tqdm import tqdm
from collections import deque
import random
import gymnasium as gym
from gymnasium.vector import AsyncVectorEnv

from rl.environment import HeartsEnv, SmartBot, RandomBot
from rl.agent import PPOAgent

def collect_expert_data(n_games=2000):
    """Collect (state, action) pairs from SmartBot."""
    print(f"Collecting expert data from {n_games} games...")
    # Use 3 SmartBots to generate high-quality game states
    env = HeartsEnv(agent_player_id=0, opponents=[SmartBot(), SmartBot(), SmartBot()])
    expert_data = []
    
    for _ in tqdm(range(n_games), desc="Collecting data"):
        obs, info = env.reset()
        done = False
        smartbot = SmartBot()
        
        while not done:
            legal_moves = info['legal_moves']
            mask = info['action_mask']
            
            # SmartBot decision
            card = smartbot.select_card(legal_moves, info)
            action = card.to_index()
            
            expert_data.append((obs.copy(), action, mask.copy()))
            
            obs, reward, done, truncated, info = env.step(action)
            
    print(f"Collected {len(expert_data)} samples.")
    return expert_data

def train_pretraining(agent, expert_data, epochs=20, batch_size=256):
    """Train policy using Supervised Learning."""
    print(f"\nPhase 1: Pre-training ({epochs} epochs)")
    
    optimizer = optim.Adam(agent.network.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    n_samples = len(expert_data)
    indices = list(range(n_samples))
    
    for epoch in range(epochs):
        random.shuffle(indices)
        total_loss = 0
        correct = 0
        total = 0
        
        agent.network.train()
        
        for i in range(0, n_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            
            states = torch.FloatTensor(np.array([expert_data[j][0] for j in batch_indices])).to(agent.device)
            actions = torch.LongTensor(np.array([expert_data[j][1] for j in batch_indices])).to(agent.device)
            masks = torch.FloatTensor(np.array([expert_data[j][2] for j in batch_indices])).to(agent.device)
            
            # Forward pass
            logits, _ = agent.network(states, masks)
            logits = logits.masked_fill(masks == 0, -1e9)
            
            loss = criterion(logits, actions)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Accuracy
            preds = logits.argmax(dim=1)
            correct += (preds == actions).sum().item()
            total += len(batch_indices)
            
        acc = correct / total * 100
        n_batches = max(1, n_samples // batch_size)
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Accuracy: {acc:.1f}%")
        
    print("Pre-training complete!")
    return agent

def train_finetune(agent, n_episodes=30000, n_steps=512, min_buffer=64):
    """Phase 2: Fine-tuning against 3 SmartBots using Parallel Envs."""
    print(f"\nPhase 2: Fine-tuning ({n_episodes} episodes)")
    checkpoint_dir = 'checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Parallel Environments - Maximize CPU/GPU
    num_envs = 16
    
    # Faster curriculum: 0→3 SmartBots over 3000 eps (instead of 6000)
    def get_opponents(episode):
        num_smarts = min(3, episode // 1000)
        opps = [RandomBot() for _ in range(3)]
        for i in range(num_smarts):
            opps[i] = SmartBot()
        return opps

    def make_env(episode_idx):
        def _init():
            return HeartsEnv(agent_player_id=0, opponents=get_opponents(episode_idx))
        return _init

    # Initialize vector env
    env_fns = [make_env(0) for _ in range(num_envs)]
    envs = AsyncVectorEnv(env_fns)
    
    rewards_history = deque(maxlen=100)
    wins_history = deque(maxlen=100)
    best_win_rate = 0.0
    
    # Lower LR for fine-tuning
    current_lr = 1e-5
    for param_group in agent.optimizer.param_groups:
        param_group['lr'] = current_lr
    
    # Gentler clipping
    agent.clip_epsilon = 0.1
    
    obs, info = envs.reset()
    
    # Total updates approximation
    n_updates = n_episodes * 2
    total_steps = 0
    
    reductions = 0
    
    pbar = tqdm(range(n_updates))
    for update_step in pbar:
        # Collection phase
        batch_states, batch_actions, batch_log_probs, batch_rewards, batch_masks, batch_values = [], [], [], [], [], []
        batch_dones = []
        
        steps_per_env = n_steps // num_envs
        
        for _ in range(steps_per_env):
            action_masks = info['action_mask'] # (num_envs, 52)
            
            # Select actions
            action_idx, log_prob, value = agent.select_action_batch(obs, action_masks)
            
            next_obs, reward, terminated, truncated, next_info = envs.step(action_idx)
            done = np.logical_or(terminated, truncated)
            
            # Store transition
            batch_states.append(obs)
            batch_actions.append(action_idx)
            batch_log_probs.append(log_prob)
            batch_rewards.append(reward)
            batch_masks.append(action_masks)
            batch_values.append(value)
            batch_dones.append(done)
            
            obs = next_obs
            info = next_info
            total_steps += num_envs
            
            # Track stats
            if 'winner' in info:
                for i, winner in enumerate(info['winner']):
                    if winner != -1:  # -1 means no winner yet
                        is_win = 1 if winner == 0 else 0
                        wins_history.append(is_win)

        # Prepare data for update
        # Convert lists to arrays: (steps, num_envs, ...)
        states_arr = np.array(batch_states)
        actions_arr = np.array(batch_actions)
        log_probs_arr = np.array(batch_log_probs)
        rewards_arr = np.array(batch_rewards)
        masks_arr = np.array(batch_masks)
        values_arr = np.array(batch_values)
        dones_arr = np.array(batch_dones)
        
        # Get next values for GAE
        next_values = agent.get_value_batch(obs, info['action_mask'])
        
        # Compute GAE per environment
        # Transpose to (num_envs, steps)
        rewards_env = rewards_arr.T
        values_env = values_arr.T
        dones_env = dones_arr.T
        
        all_returns = np.zeros_like(rewards_env)
        all_advantages = np.zeros_like(rewards_env)
        
        for i in range(num_envs):
            # Compute GAE for env i
            next_val = next_values[i]
            env_returns, env_advantages = agent.compute_gae(
                rewards_env[i], 
                values_env[i], 
                next_val, 
                dones_env[i]
            )
            all_returns[i] = env_returns
            all_advantages[i] = env_advantages
            
        # Flatten everything for PPO update
        # (num_envs, steps) -> (num_envs * steps)
        states_flat = states_arr.transpose(1, 0, 2).reshape(-1, 183)
        actions_flat = actions_arr.transpose(1, 0).flatten()
        log_probs_flat = log_probs_arr.transpose(1, 0).flatten()
        masks_flat = masks_arr.transpose(1, 0, 2).reshape(-1, 52)
        returns_flat = all_returns.flatten()
        advantages_flat = all_advantages.flatten()
        
        # Update Agent
        loss_info = agent.update(
            states_flat, 
            actions_flat, 
            log_probs_flat, 
            returns_flat, 
            advantages_flat, 
            masks_flat, 
            update_step, 
            n_updates
        )
        
        if (update_step + 1) % 10 == 0:
            win_rate = np.mean(wins_history) * 100 if wins_history else 0.0
            pbar.set_description(f"Win: {win_rate:.1f}% | Loss: {loss_info.get('loss', 0):.4f}")
            
            # Save best model
            if win_rate > best_win_rate and update_step > 50:
                best_win_rate = win_rate
                agent.save(os.path.join(checkpoint_dir, 'hearts_rl_agent.pt'))
            
            # Adaptive Logic
            if update_step > 200 and win_rate < 25.0 and best_win_rate > 30.0:
                 # Halve LR
                current_lr *= 0.5
                for param_group in agent.optimizer.param_groups:
                    param_group['lr'] = current_lr
                
                # Boost entropy
                agent.entropy_coef = 0.05
                
                reductions += 1
                if reductions > 5:
                    print("Too many adjustments; stopping early.")
                    break
                
                wins_history.clear()

            if (update_step + 1) % 1000 == 0:
                agent.save(os.path.join(checkpoint_dir, f'hearts_superhuman_{update_step+1}.pt'))
    
    envs.close()

def main():
    # Create PPO Agent
    agent = PPOAgent(
        obs_size=183,
        hidden_size=1024,
        action_size=52,
        lr=3e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Phase 1: Pre-training
    expert_data = collect_expert_data(n_games=2000)
    agent = train_pretraining(agent, expert_data, epochs=15)
    
    # Save pre-trained model
    os.makedirs('checkpoints', exist_ok=True)
    agent.save('checkpoints/hearts_pretrained.pt')
    
    # Phase 2: Fine-tuning
    if torch.cuda.is_available():
        print(f"🚀 Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️ Training on CPU - this will be slow!")
        
    train_finetune(agent, n_episodes=30000)

if __name__ == '__main__':
    main()
