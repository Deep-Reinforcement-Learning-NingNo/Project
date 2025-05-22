import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mamba_ssm import Mamba2

class PolicyNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim, act_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, act_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.softmax(self.action_head(x))

class ValueNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, obs):
        x = self.relu(self.fc1(obs))
        x = self.relu(self.fc2(x))
        return self.value_head(x)

class MambaWorldModel(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim):
        super().__init__()
        self.input_layer = nn.Linear(obs_dim + act_dim, hidden_dim)
        self.mamba = Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
        self.output_obs = nn.Linear(hidden_dim, obs_dim)
        self.output_reward = nn.Linear(hidden_dim, 1)

    def forward(self, obs, action):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)
        x = torch.cat([obs, action], dim=-1).unsqueeze(1)
        x = self.input_layer(x)
        x = self.mamba(x).squeeze(1)
        next_obs = self.output_obs(x)
        reward = self.output_reward(x).squeeze(1)
        return next_obs, reward

class Mamba2ModelBasedRL:
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=256, lr=3e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.policies = [PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device) for _ in range(n_agents)]
        self.values = [ValueNetwork(obs_dim, hidden_dim).to(self.device) for _ in range(n_agents)]
        self.models = [MambaWorldModel(obs_dim, act_dim, hidden_dim).to(self.device) for _ in range(n_agents)]

        self.policy_optimizers = [optim.Adam(policy.parameters(), lr=lr) for policy in self.policies]
        self.value_optimizers = [optim.Adam(value.parameters(), lr=lr) for value in self.values]
        self.model_optimizers = [optim.Adam(model.parameters(), lr=lr) for model in self.models]

        self.states = [[] for _ in range(n_agents)]
        self.actions = [[] for _ in range(n_agents)]
        self.next_states = [[] for _ in range(n_agents)]
        self.rewards = [[] for _ in range(n_agents)]

    def train_model(self, agent_idx, states, actions, next_states, rewards, batch_size=32):
        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.nn.functional.one_hot(torch.LongTensor(actions), num_classes=self.act_dim).float().to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        r = torch.FloatTensor(np.array(rewards)).to(self.device)

        model = self.models[agent_idx]
        optimizer = self.model_optimizers[agent_idx]

        num_batches = len(states) // batch_size
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = (batch_idx + 1) * batch_size

            batch_states = s[start_idx:end_idx]
            batch_actions = a[start_idx:end_idx]
            batch_next_states = ns[start_idx:end_idx]
            batch_rewards = r[start_idx:end_idx]

            pred_ns, pred_r = model(batch_states, batch_actions)
            loss_obs = nn.MSELoss()(pred_ns, batch_next_states)
            loss_r = nn.MSELoss()(pred_r, batch_rewards)
            loss = loss_obs + loss_r

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def imagined_rollout(self, agent_idx, start_obs, horizon=5):
        obs = torch.FloatTensor(start_obs).unsqueeze(0).to(self.device)
        imagined_obs, imagined_actions, imagined_rewards = [], [], []

        policy = self.policies[agent_idx]
        model = self.models[agent_idx]

        for _ in range(horizon):
            with torch.no_grad():
                probs = policy(obs)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                action_scalar = action.squeeze().item()
                action_onehot = torch.nn.functional.one_hot(
                    torch.tensor([action_scalar], device=self.device), num_classes=self.act_dim
                ).float()
                next_obs, reward = model(obs, action_onehot)

            imagined_obs.append(obs)
            imagined_actions.append(torch.tensor(action_scalar, device=self.device))
            imagined_rewards.append(reward.squeeze())
            obs = next_obs.detach()

        return imagined_obs, imagined_actions, imagined_rewards

    def update_policy(self, agent_idx, imagined_obs, imagined_actions, imagined_rewards, episode, batch_size=32):
        returns = []
        R = 0
        for r in reversed(imagined_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        obs_tensor = torch.cat(imagined_obs).to(self.device)
        act_tensor = torch.stack(imagined_actions).reshape(-1).to(self.device)
        ret_tensor = torch.tensor(returns).unsqueeze(1).float().to(self.device)

        policy = self.policies[agent_idx]
        value = self.values[agent_idx]
        policy_opt = self.policy_optimizers[agent_idx]
        value_opt = self.value_optimizers[agent_idx]

        probs = policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(act_tensor)

        values = value(obs_tensor)
        advantages = ret_tensor - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        entropy_weight = max(0.1 - (episode / 10000) * 0.1, 0.01)
        policy_loss = -(log_probs * advantages.squeeze()).mean() - entropy_weight * dist.entropy().mean()
        value_loss = nn.MSELoss()(values, ret_tensor)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        return policy_loss.item(), value_loss.item()

    def train_on_imagined(self, states_n, actions_n, next_states_n, rewards_n, batch_size=32, global_episode=0):
        agent_total_rewards = [0.0 for _ in range(self.n_agents)]
        total_policy_loss, total_value_loss, total_steps = 0, 0, 0

        for i in range(self.n_agents):
            num_batches = len(states_n[i]) // batch_size
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = (batch_idx + 1) * batch_size

                batch_states = states_n[i][start_idx:end_idx]
                batch_actions = actions_n[i][start_idx:end_idx]
                batch_next_states = next_states_n[i][start_idx:end_idx]
                batch_rewards = rewards_n[i][start_idx:end_idx]

                self.train_model(i, batch_states, batch_actions, batch_next_states, batch_rewards)

                start_obs = batch_states[np.random.randint(0, len(batch_states))]
                obs, actions, rewards = self.imagined_rollout(i, start_obs)
                policy_loss, value_loss = self.update_policy(i, obs, actions, rewards, episode=global_episode)

                agent_total_rewards[i] += sum([r.item() if isinstance(r, torch.Tensor) else r for r in rewards])
                total_steps += len(rewards)
                total_policy_loss += policy_loss
                total_value_loss += value_loss

        avg_reward = sum(agent_total_rewards) / self.n_agents
        return agent_total_rewards, total_policy_loss / self.n_agents, total_value_loss / self.n_agents, total_steps // self.n_agents


    def learn(self, env, max_steps=1000, batch_size=32, global_episode=0):
        obs_n = env.reset()[0]
        self.states = [[] for _ in range(self.n_agents)]
        self.actions = [[] for _ in range(self.n_agents)]
        self.next_states = [[] for _ in range(self.n_agents)]
        self.rewards = [[] for _ in range(self.n_agents)]
        step_count = 0

        for _ in range(max_steps):
            action_n = []
            for i in range(self.n_agents):
                obs_i = obs_n[i]
                obs_tensor = torch.FloatTensor(obs_i).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    probs = self.policies[i](obs_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
                action_n.append(action)

            next_obs_n, reward_n, terminated_n, truncated_n, _ = env.step(action_n)

            for i in range(self.n_agents):
                self.states[i].append(obs_n[i])
                self.actions[i].append(action_n[i])
                self.next_states[i].append(next_obs_n[i])
                self.rewards[i].append(reward_n[i])

            step_count += 1
            obs_n = next_obs_n
            done = all(t or tr for t, tr in zip(terminated_n, truncated_n))
            if done:
                break

        result = self.train_on_imagined(
            self.states, self.actions, self.next_states, self.rewards,
            batch_size=batch_size, global_episode=global_episode
        )
        return (*result, step_count)

