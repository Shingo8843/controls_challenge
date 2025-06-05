from . import BaseController
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_dim * 2)  # mean + log_std
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            nn.init.constant_(m.bias, 0)

    def forward(self, state):
        params = self.actor(state)
        mean, log_std = torch.chunk(params, 2, dim=-1)
        log_std = torch.clamp(log_std, -4, 1)  # safer range
        std = torch.exp(log_std)
        value = self.critic(state)
        return mean, std, value


class Controller(BaseController):
    def __init__(self, load_model=True, model_path='models/rl/ppo_ep900.pt'):
        self.state_dim = 5
        self.action_dim = 1
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.ppo_epochs = 5
        self.clip_param = 0.2
        self.learning_rate = 3e-4

        self.actor_critic = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.learning_rate)

        self.episode_buffer = []
        self.training = not load_model
        self.episode_count = 0
        self.best_cost = float('inf')

        if load_model and model_path:
            try:
                self.load_checkpoint(model_path)
                print(f"Loaded PPO model from {model_path}")
            except Exception as e:
                print(f"Failed to load PPO model: {e}")

        self.prev_lataccel = None

    def _get_state(self, target_lataccel, current_lataccel, state, future_plan):
        return np.array([
            current_lataccel / 5.0,
            target_lataccel / 5.0,
            state.v_ego / 30.0,
            state.a_ego / 5.0,
            state.roll_lataccel / 5.0
        ], dtype=np.float32)

    def _compute_reward(self, target_lataccel, current_lataccel):
        err = target_lataccel - current_lataccel
        jerk = 0.0 if self.prev_lataccel is None else abs(current_lataccel - self.prev_lataccel)
        self.prev_lataccel = current_lataccel
        reward = - (err**2) - 0.05 * (jerk**2)
        return reward

    def update(self, target_lataccel, current_lataccel, state, future_plan):
        state_vec = self._get_state(target_lataccel, current_lataccel, state, future_plan)
        state_tensor = torch.FloatTensor(state_vec).unsqueeze(0).to(device)

        with torch.no_grad():
            mean, std, value = self.actor_critic(state_tensor)
            dist = Normal(mean, std)
            if self.training:
                action = dist.sample()
            else:
                action = mean
            action = torch.clamp(action, -2.0, 2.0)

        if self.training:
            log_prob = dist.log_prob(action).sum(-1)
            reward = self._compute_reward(target_lataccel, current_lataccel)

            self.episode_buffer.append({
                "state": state_vec,
                "action": action.cpu().numpy().flatten(),
                "value": value.item(),
                "log_prob": log_prob.item(),
                "reward": reward
            })

        return action.item()

    def _process_trajectory(self):
        buffer = self.episode_buffer
        # Pre-allocate numpy arrays
        states = np.array([t['state'] for t in buffer], dtype=np.float32)
        actions = np.array([t['action'] for t in buffer], dtype=np.float32)
        values = np.array([t['value'] for t in buffer], dtype=np.float32)
        log_probs = np.array([t['log_prob'] for t in buffer], dtype=np.float32)
        rewards = np.array([t['reward'] for t in buffer], dtype=np.float32)

        # Convert to tensors in a single operation
        states = torch.from_numpy(states).to(device)
        actions = torch.from_numpy(actions).to(device)
        values = torch.from_numpy(values).to(device)
        log_probs = torch.from_numpy(log_probs).to(device)

        # Compute advantages using GAE
        returns = []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value - values[i].item()
            gae = delta + self.gamma * self.gae_lambda * gae
            returns.insert(0, gae + values[i].item())
            next_value = values[i].item()
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        advantages = returns - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return states, actions, log_probs, returns, advantages

    def train_on_episode(self):
        if not self.episode_buffer:
            return

        states, actions, log_probs_old, returns, advantages = self._process_trajectory()

        for _ in range(self.ppo_epochs):
            mean, std, values = self.actor_critic(states)
            dist = Normal(mean, std)
            log_probs = dist.log_prob(actions).sum(-1)
            entropy = dist.entropy().sum(-1)

            ratios = torch.exp(log_probs - log_probs_old)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(values.squeeze(-1), returns)
            entropy_loss = -entropy.mean()

            total_loss = actor_loss + 0.5 * critic_loss + 0.01 * entropy_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

        self.episode_buffer = []

    def save_checkpoint(self, path):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'episode_count': self.episode_count,
            'best_cost': self.best_cost
        }, path)

    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_count = checkpoint.get('episode_count', 0)
        self.best_cost = checkpoint.get('best_cost', float('inf'))

    def set_training(self, training: bool):
        self.training = training
        self.actor_critic.train(training)

    def reset(self):
        """Call this at the beginning of every episode"""
        self.episode_buffer = []
        self.prev_lataccel = None
