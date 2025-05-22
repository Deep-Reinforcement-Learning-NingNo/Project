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

class Mamba2SingleAgent:
    def __init__(self, obs_dim, act_dim, hidden_dim=256, lr=3e-4, gamma=0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = gamma

        self.policy = PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device)
        self.value = ValueNetwork(obs_dim, hidden_dim).to(self.device)
        self.model = MambaWorldModel(obs_dim, act_dim, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

    def learn(self, env, max_steps=1000):
        obs = env.reset()[0]
        done = False
        step_count = 0

        states, actions, next_states, rewards = [], [], [], []

        for _ in range(max_steps):
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            with torch.no_grad():
                probs = self.policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample().item()

            next_obs, reward, terminated, truncated, _ = env.step(action)

            states.append(obs)
            actions.append(action)
            next_states.append(next_obs)
            rewards.append(reward)

            step_count += 1
            obs = next_obs
            done = terminated or truncated
            if done:
                break

        if len(states) == 0:
            return (0.0, 0.0, 0.0), step_count

        return self.train_on_imagined(states, actions, next_states, rewards), step_count

    def train_on_imagined(self, states, actions, next_states, rewards):
        if len(states) == 0:
            return 0.0, 0.0, 0.0

        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.nn.functional.one_hot(torch.LongTensor(actions), num_classes=self.act_dim).float().to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)
        r = torch.FloatTensor(np.array(rewards)).to(self.device)

        pred_ns, pred_r = self.model(s, a)
        loss_obs = nn.MSELoss()(pred_ns, ns)
        loss_r = nn.MSELoss()(pred_r, r)
        model_loss = loss_obs + loss_r

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        obs = torch.FloatTensor(states[np.random.randint(0, len(states))]).unsqueeze(0).to(self.device)
        imagined_obs, imagined_actions, imagined_rewards = [], [], []

        for _ in range(5):
            with torch.no_grad():
                probs = self.policy(obs)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                action_onehot = torch.nn.functional.one_hot(action, num_classes=self.act_dim).float().to(self.device)
                next_obs, reward = self.model(obs, action_onehot)

            imagined_obs.append(obs)
            imagined_actions.append(action)
            imagined_rewards.append(reward.squeeze())
            obs = next_obs.detach()

        return self.update_policy(imagined_obs, imagined_actions, imagined_rewards)

    def update_policy(self, imagined_obs, imagined_actions, imagined_rewards):
        R = 0
        returns = []
        for r in reversed(imagined_rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        obs_tensor = torch.cat(imagined_obs)
        act_tensor = torch.stack(imagined_actions).reshape(-1).to(self.device)
        ret_tensor = torch.tensor(returns).unsqueeze(1).float().to(self.device)

        probs = self.policy(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(act_tensor)

        values = self.value(obs_tensor)
        advantages = ret_tensor - values.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        policy_loss = -(log_probs * advantages.squeeze()).mean() - 0.05 * dist.entropy().mean()
        value_loss = nn.MSELoss()(values, ret_tensor)

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        return sum([r.item() for r in imagined_rewards]), policy_loss.item(), value_loss.item()