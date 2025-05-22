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


class PPO:
    def __init__(self, n_observations, n_actions, hidden_dim=256, learning_rate=3e-4,
                 gamma=0.99, clip_epsilon=0.2, epochs=4):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy = PolicyNetwork(n_observations, hidden_dim, n_actions).to(self.device)
        self.policy_old = PolicyNetwork(n_observations, hidden_dim, n_actions).to(self.device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.value = ValueNetwork(n_observations, hidden_dim).to(self.device)

        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=learning_rate)

        self.clip_epsilon = clip_epsilon
        self.epochs = epochs
        self.gamma = gamma

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
        states, actions, log_probs, rewards, dones = [], [], [], [], []
        state, _ = env.reset()
        episode_reward = 0
        step_count = 0  
        # print(state)
        for _ in range(max_steps):
            action, log_prob = self.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            dones.append(done)

            episode_reward += reward
            state = next_state
            step_count += 1  

            if done:
                break

        return states, actions, log_probs, rewards, dones, episode_reward, step_count

    def update_policy(self, states, actions, old_log_probs, returns):
        for _ in range(self.epochs):
            probs = self.policy(states)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()

            state_values = self.value(states)
            advantages = returns - state_values.detach() # differrnt between Return and Expected Return
            ratios = torch.exp(new_log_probs - old_log_probs) #rt = exp(log[new policy[at|st]]-log[old policy[at|st]])

            surr1 = ratios * advantages # rt*At
            surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages # clip(rt,1-e,1+e)*At
            policy_loss = -torch.min(surr1, surr2).mean() - 0.01 * entropy #Lclip
            # policy_loss = -torch.min(surr1, surr2).mean()
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
        # 1. Collect rollout
        (
            states,
            actions,
            log_probs,
            rewards,
            dones,
            episode_reward,
            step_count  
        ) = self.collect_trajectory(env, max_steps)

        # 2. Prepare tensors
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        returns = torch.FloatTensor(self.compute_returns(rewards, dones)).unsqueeze(1).to(self.device)

        # 3. Update policy & value networks
        policy_loss, value_loss = self.update_policy(states, actions, old_log_probs, returns)

        return episode_reward, policy_loss, value_loss, step_count

