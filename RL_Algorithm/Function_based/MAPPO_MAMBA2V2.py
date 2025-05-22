
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from mamba_ssm import Mamba2

# --- Shared Policy Network ---
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


# --- Shared Value Network ---
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


# --- Mamba2 World Model ---
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


# --- Multi-Agent Dyna-PPO ---
class MultiAgentDynaPPO:
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=256, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4, batch_size=64, lambda_=0.95):
        self.n_agents = n_agents
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device)
        self.policy_old = PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value = ValueNetwork(obs_dim, hidden_dim).to(self.device)
        self.model = MambaDynamicsModel(obs_dim, 1, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=lr)
        self.model_optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.lambda_ = lambda_

    def select_action(self, obs_batch):
        obs_tensor = torch.FloatTensor(obs_batch).to(self.device)
        with torch.no_grad():
            probs = self.policy_old(obs_tensor)
        dist = torch.distributions.Categorical(probs)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        return actions.cpu().numpy(), log_probs.cpu().numpy()

    def compute_gae(self, rewards, values, dones):
        advantages, returns = [], []
        for r, v, d in zip(reversed(rewards), reversed(values), reversed(dones)):
            adv = 0
            ret = v[-1]
            advs, rets = [], []
            for t in reversed(range(len(r))):
                delta = r[t] + self.gamma * v[t+1] * (1 - d[t]) - v[t]
                adv = delta + self.gamma * self.lambda_ * (1 - d[t]) * adv
                ret = r[t] + self.gamma * (1 - d[t]) * ret
                advs.insert(0, adv)
                rets.insert(0, ret)
            advantages.append(advs)
            returns.append(rets)
        return advantages, returns

    def collect_trajectory(self, env, max_steps):
        obs, _ = env.reset()
        obs = np.array(obs)
        all_data = {k: [[] for _ in range(self.n_agents)] for k in ['obs', 'obs_next', 'actions', 'log_probs', 'rewards', 'dones', 'values']}
        episode_reward = [0 for _ in range(self.n_agents)]
        step_count = 0

        for _ in range(max_steps):
            actions, log_probs = self.select_action(obs)
            with torch.no_grad():
                values = self.value(torch.FloatTensor(obs).to(self.device)).cpu().numpy()

            next_obs, rewards, terminated, truncated, _ = env.step(actions)
            done_flags = np.logical_or(terminated, truncated)
            # print(f"[Step {step_count + 1}] Rewards:")
            # for i in range(self.n_agents):
            #     print(f"  Agent {i}: reward = {rewards[i]:.4f}")

            for i in range(self.n_agents):
                all_data['obs'][i].append(obs[i])
                all_data['obs_next'][i].append(next_obs[i])
                all_data['actions'][i].append(actions[i])
                all_data['log_probs'][i].append(log_probs[i])
                all_data['rewards'][i].append(rewards[i])
                all_data['dones'][i].append(done_flags[i])
                all_data['values'][i].append(values[i][0])
                episode_reward[i] += rewards[i]

            obs = np.array(next_obs)
            step_count += 1
            if all(done_flags):
                break

        # append final value estimate
        final_values = self.value(torch.FloatTensor(obs).to(self.device)).detach().cpu().numpy()
        for i in range(self.n_agents):
            all_data['values'][i].append(final_values[i][0])

        return all_data, episode_reward, step_count

    def train_mamba_model(self, all_obs, all_actions, all_next_obs):
        s = torch.FloatTensor(np.vstack(all_obs)).to(self.device)
        a = torch.FloatTensor(np.vstack(all_actions)).to(self.device)
        ns = torch.FloatTensor(np.vstack(all_next_obs)).to(self.device)

        if a.dim() == 1:
            a = a.unsqueeze(1)
        elif a.dim() == 3:
            a = a.squeeze(2)

        pred = self.model(s, a)
        loss = nn.MSELoss()(pred, ns)

        self.model_optimizer.zero_grad()
        loss.backward()
        self.model_optimizer.step()

    def update_policy(self, states, actions, old_log_probs, advantages, returns):
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.epochs):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            state_values = self.value(states)
            ratios = torch.exp(new_log_probs - old_log_probs)

            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
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
        data, ep_rewards, step_count = self.collect_trajectory(env, max_steps)

        flat = lambda l: [x for agent in l for x in agent]
        all_obs = flat(data['obs'])
        all_actions = flat(data['actions'])
        # all_next_obs = flat(data['obs'][i+1:] for i in range(len(data['obs'])-1))  # approximation
        all_next_obs = flat(data['obs_next'])


        self.train_mamba_model(all_obs, all_actions, all_next_obs)

        advantages, returns = self.compute_gae(data['rewards'], data['values'], data['dones'])

        obs_tensor = torch.FloatTensor(np.array(flat(data['obs']))).to(self.device)
        act_tensor = torch.LongTensor(np.array(flat(data['actions']))).to(self.device)
        logp_tensor = torch.FloatTensor(flat(data['log_probs'])).to(self.device)
        adv_tensor = torch.FloatTensor(flat(advantages)).to(self.device)
        ret_tensor = torch.FloatTensor(flat(returns)).unsqueeze(1).to(self.device)

        pol_loss, val_loss = self.update_policy(obs_tensor, act_tensor, logp_tensor, adv_tensor, ret_tensor)

        return ep_rewards, pol_loss, val_loss, step_count