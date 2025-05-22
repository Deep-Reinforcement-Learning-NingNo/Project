
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mamba_ssm import Mamba2

# --- Policy Network ---
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.action_head = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.softmax(self.action_head(x))


# --- Value Network ---
class ValueNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.value_head(x)


# --- Mamba2 Dynamics Model ---
class MambaDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.input_layer = nn.Linear(state_dim + action_dim, hidden_dim)
        self.mamba_block = Mamba2(d_model=hidden_dim, d_state=64, d_conv=4, expand=2)
        self.output_layer = nn.Linear(hidden_dim, state_dim)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1).unsqueeze(1)
        x = self.input_layer(x)
        x = self.mamba_block(x)
        x = self.output_layer(x.squeeze(1))
        return x


# --- PPO with Mamba2 (Dyna-PPO) ---
class DynaPPO:
    def __init__(self, n_observations, n_actions, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, clip_epsilon=0.2, epochs=4, batch_size=64):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = PolicyNetwork(n_observations, hidden_dim, n_actions).to(self.device)
        self.policy_old = PolicyNetwork(n_observations, hidden_dim, n_actions).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value = ValueNetwork(n_observations, hidden_dim).to(self.device)
        self.model = MambaDynamicsModel(n_observations, 1, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.gamma = gamma
        self.batch_size = batch_size

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_old(state)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item()

    def compute_returns(self, rewards, dones):
        returns = []
        R = 0
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    def collect_trajectory(self, env, max_steps):
        state, _ = env.reset()
        states, actions, log_probs, rewards, dones, next_states = [], [], [], [], [], []
        total_reward = 0
        step_count = 0

        for _ in range(max_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)
            next_states.append(next_state)

            total_reward += reward
            state = next_state
            step_count += 1

            if done:
                break

        return states, actions, log_probs, rewards, dones, next_states, total_reward, step_count

    def train_mamba_model(self, states, actions, next_states):
        s = torch.FloatTensor(np.array(states)).to(self.device)
        a = torch.FloatTensor(actions).unsqueeze(1).to(self.device)
        ns = torch.FloatTensor(np.array(next_states)).to(self.device)

        pred_next = self.model(s, a)
        model_loss = nn.MSELoss()(pred_next, ns)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

    def update_policy(self, states, actions, old_log_probs, returns):
        for _ in range(self.epochs):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            state_values = self.value(states)
            advantages = returns - state_values.detach()
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
            value_loss = nn.MSELoss()(state_values, returns)

            self.policy_optimizer.zero_grad()
            policy_loss.backward()
            self.policy_optimizer.step()

            self.value_optimizer.zero_grad()
            value_loss.backward()
            self.value_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return policy_loss.item(), value_loss.item()

    def learn(self, env, max_steps=1000):
        # Step 1: collect real trajectory
        ( states,
          actions, 
          log_probs, 
          rewards, 
          dones, 
          next_states, 
          ep_reward, 
          step_count
        ) = self.collect_trajectory(env, max_steps)

        # Step 2: train dynamics model (Mamba2)
        self.train_mamba_model(states, actions, next_states)

        # Step 3: PPO policy update
        s_tensor = torch.FloatTensor(np.array(states)).to(self.device)
        a_tensor = torch.LongTensor(actions).to(self.device)
        logp_tensor = torch.FloatTensor(log_probs).to(self.device)
        ret_tensor = torch.FloatTensor(self.compute_returns(rewards, dones)).unsqueeze(1).to(self.device)

        policy_loss, value_loss = self.update_policy(s_tensor, a_tensor, logp_tensor, ret_tensor)
        
        return ep_reward, policy_loss, value_loss, step_count