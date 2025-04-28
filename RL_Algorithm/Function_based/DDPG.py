import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.nn.functional import mse_loss
from RL_Algorithm.RL_base_function import BaseAlgorithm

class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=1e-4):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.tanh(self.fc3(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, learning_rate=1e-4):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

        self.relu = nn.ReLU()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.zeros_(m.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

class DDPG(BaseAlgorithm):
    def __init__(self, 
                 device=None,
                 n_observations=12,
                 n_actions=1,
                 hidden_dim=256,
                 action_range=[-1.0, 1.0],
                 learning_rate=1e-4,
                 tau=0.005,
                 discount_factor=0.99,
                 buffer_size=100000,
                 batch_size=64):
        super(DDPG, self).__init__(
            num_of_action=n_actions,
            action_range=action_range,
            learning_rate=learning_rate,
            discount_factor=discount_factor,
            buffer_size=buffer_size,
            batch_size=batch_size
        )

        self.device = (torch.device(device) if device else
                       torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        self.actor = Actor(n_observations, hidden_dim, n_actions).to(self.device)
        self.actor_target = Actor(n_observations, hidden_dim, n_actions).to(self.device)
        self.critic = Critic(n_observations, n_actions, hidden_dim).to(self.device)
        self.critic_target = Critic(n_observations, n_actions, hidden_dim).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=learning_rate)

        self.tau = tau
        self.gamma = discount_factor
        self.mse = nn.MSELoss()
        self.batch_size = batch_size

    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.actor(state)
        action = action.cpu().data.numpy().flatten()

        if noise != 0.0:
            action += noise * np.random.randn(*action.shape)

        action = np.clip(action, -1.0, 1.0)
        discrete_action = 1 if action[0] > 0 else 0
        return discrete_action

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device).squeeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        actions = actions.unsqueeze(1)  # Ensure action shape is (batch, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            q_next = self.critic_target(next_states, next_actions)
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q_val = self.critic(states, actions)
        critic_loss = mse_loss(q_val, q_target)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

    def _soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def learn(self, env, max_steps=1000, noise_scale=0.1, noise_decay=0.995):
        state, _ = env.reset()
        episode_reward = 0

        for step in range(max_steps):
            action = self.select_action(state, noise=noise_scale)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            self.memory.add(state, [action], reward, next_state, done)

            self.update()

            state = next_state
            episode_reward += reward
            noise_scale *= noise_decay

            if done:
                break

        return episode_reward
