
# MAPPO version of your PPO implementation (shared policy, centralized critic)
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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


class CentralizedValueNetwork(nn.Module):
    def __init__(self, total_obs_dim, hidden_dim):
        super(CentralizedValueNetwork, self).__init__()
        self.fc1 = nn.Linear(total_obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.value_head = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.value_head(x)


class MAPPO:
    def __init__(self, n_agents, obs_dim, act_dim, hidden_dim=256, lr=3e-4,
                 gamma=0.99, clip_eps=0.2, epochs=4):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.total_obs_dim = n_agents * obs_dim

        self.policy = PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device)
        self.policy_old = PolicyNetwork(obs_dim, hidden_dim, act_dim).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.critic = CentralizedValueNetwork(self.total_obs_dim, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

        self.clip_eps = clip_eps
        self.epochs = epochs
        self.gamma = gamma

    def select_action(self, obs):
        obs = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.policy_old(obs)
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
        obs = env.reset()[0]
        agent_data = [
            {"observations": [], "actions": [], "log_probs": [], "rewards": [], "dones": []}
            for _ in range(self.n_agents)
        ]

        step_count = 0
        is_multi = self.n_agents > 1

        for _ in range(max_steps):
            actions = []
            log_probs = []

            for i in range(self.n_agents):
                obs_i = obs[i] if is_multi else obs
                obs_tensor = torch.FloatTensor(obs_i).unsqueeze(0).to(self.device)
                probs = self.policy_old(obs_tensor)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                actions.append(action.item())
                log_probs.append(log_prob.item())

                agent_data[i]["observations"].append(obs_i)
                agent_data[i]["actions"].append(action.item())
                agent_data[i]["log_probs"].append(log_prob.item())

            next_obs, rewards, terminateds, truncateds, _ = env.step(actions if is_multi else actions[0])

            if isinstance(terminateds, bool):
                done_flags = [terminateds or truncateds]
            else:
                done_flags = [t or tr for t, tr in zip(terminateds, truncateds)]

            for i in range(self.n_agents):
                reward_i = rewards[i] if is_multi else rewards
                agent_data[i]["rewards"].append(reward_i)
                agent_data[i]["dones"].append(done_flags[i] if is_multi else done_flags[0])

            obs = next_obs
            step_count += 1

            if all(done_flags):
                break

        return agent_data, step_count

    def update_policy(self, agent_data):
        for _ in range(self.epochs):
            for agent_id in range(self.n_agents):
                obs = torch.FloatTensor(agent_data[agent_id]["observations"]).to(self.device)
                actions = torch.LongTensor(agent_data[agent_id]["actions"]).to(self.device)
                old_log_probs = torch.FloatTensor(agent_data[agent_id]["log_probs"]).to(self.device)
                returns = torch.FloatTensor(self.compute_returns(agent_data[agent_id]["rewards"],
                                                                 agent_data[agent_id]["dones"])).unsqueeze(1).to(self.device)

                all_obs = [torch.FloatTensor(agent_data[i]["observations"]).to(self.device) for i in range(self.n_agents)]
                centralized_obs = torch.cat(all_obs, dim=-1)

                probs = self.policy(obs)
                dist = torch.distributions.Categorical(probs)
                new_log_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                state_values = self.critic(centralized_obs)
                advantages = returns - state_values.detach()
                ratios = torch.exp(new_log_probs - old_log_probs)

                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
                policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy
                value_loss = nn.MSELoss()(state_values, returns)

                self.policy_optimizer.zero_grad()
                policy_loss.backward()
                self.policy_optimizer.step()

                self.critic_optimizer.zero_grad()
                value_loss.backward()
                self.critic_optimizer.step()

        self.policy_old.load_state_dict(self.policy.state_dict())
        return policy_loss.item(), value_loss.item()

    def learn(self, env, max_steps=1000):
        agent_data, step_count = self.collect_trajectory(env, max_steps)

        agent_rewards = [sum(agent_data[i]["rewards"]) for i in range(self.n_agents)]
        policy_loss, value_loss = self.update_policy(agent_data)

        return agent_rewards, policy_loss, value_loss, step_count