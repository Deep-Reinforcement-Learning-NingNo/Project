import torch
import numpy as np
import random
import os
from tqdm import tqdm
import wandb
import argparse

from mpe2 import simple_spread_v3
import supersuit as ss
from RL_Algorithm.Function_based.MAMBA2_batch import Mamba2ModelBasedRL

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="No render")
args = parser.parse_args()

# --- Config ---
N = 3
local_ratio = 1
max_cycles = 150
continuous_actions = False
batch_size = 32

base_env = simple_spread_v3.parallel_env(
    N=N,
    local_ratio=local_ratio,
    max_cycles=max_cycles,
    continuous_actions=continuous_actions
)
n_agents = len(base_env.possible_agents)

for agent in base_env.unwrapped.world.agents:
    agent.size = 0.05

for landmark in base_env.unwrapped.world.landmarks:
    landmark.size = 0.05
    landmark.movable = False
    landmark.collide = False
    landmark.state.p_vel = np.zeros(base_env.unwrapped.world.dim_p)

env = ss.pettingzoo_env_to_vec_env_v1(base_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="gymnasium")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0] if continuous_actions else env.action_space.n

# --- Hyperparams ---
learning_rate = 5e-5
hidden_dim = 256
discount_factor = 0.90
n_episodes = 10000
max_steps_per_episode = max_cycles
save_interval = 100
random_landmark = 1

save_path = (
    f"./saved_models/MAMBA2_mpe_simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}-"
    f"random_landmark{random_landmark}"
)

wandb_name = (
    f"MAMBA2-MPE-simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}-"
    f"random_landmark{random_landmark}"
)

os.makedirs(save_path, exist_ok=True)
set_seed(42)

wandb.init(
    project="mpe-multi-agent",
    name=wandb_name,
    config={
        "env": "simple_spread_v3",
        "n_agents": n_agents,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "gamma": discount_factor,
    }
)

agent = Mamba2ModelBasedRL(
    n_agents=n_agents,
    obs_dim=n_observations,
    act_dim=n_actions,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    gamma=discount_factor
)

total_reward, total_policy_loss, total_value_loss, total_step = 0, 0, 0, 0
max_step = 10000

states_batch = [[] for _ in range(n_agents)]
actions_batch = [[] for _ in range(n_agents)]
next_states_batch = [[] for _ in range(n_agents)]
rewards_batch = [[] for _ in range(n_agents)]

for episode in tqdm(range(n_episodes)):
    for landmark in base_env.unwrapped.world.landmarks:
        landmark.state.p_pos = np.random.uniform(-1.0, 1.0, size=base_env.unwrapped.world.dim_p)
        landmark.state.p_vel = np.zeros(base_env.unwrapped.world.dim_p)

    agent_rewards, policy_loss, value_loss, avg_steps, step_count = agent.learn(
        env=env,
        max_steps=max_steps_per_episode,
        batch_size=batch_size,
        global_episode=episode
    )

    for i in range(n_agents):
        states_batch[i].extend(agent.states[i])
        actions_batch[i].extend(agent.actions[i])
        next_states_batch[i].extend(agent.next_states[i])
        rewards_batch[i].extend(agent.rewards[i])

    if all(len(states_batch[i]) >= batch_size for i in range(n_agents)):
        agent_rewards, policy_loss, value_loss, step_count = agent.train_on_imagined(
            states_batch, actions_batch, next_states_batch, rewards_batch, batch_size=batch_size
        )
        states_batch = [[] for _ in range(n_agents)]
        actions_batch = [[] for _ in range(n_agents)]
        next_states_batch = [[] for _ in range(n_agents)]
        rewards_batch = [[] for _ in range(n_agents)]

    agent_rewards = agent_rewards[:n_agents]
    avg_reward = sum(agent_rewards) / n_agents

    # Log to WandB every episode (like DynaMAPPO)
    wandb_log = {
        "avg_reward": avg_reward,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "step_count": step_count,
        "episode": episode
    }
    for i, r in enumerate(agent_rewards):
        wandb_log[f"agent_{i}_reward"] = r
    wandb.log(wandb_log)

    print(f"[Ep {episode+1}] " + " | ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(agent_rewards)]) +
          f" | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Steps: {step_count}")

    total_reward += avg_reward
    total_policy_loss += policy_loss
    total_value_loss += value_loss
    total_step += step_count
    max_step = max(max_step, step_count)

    if (episode + 1) % save_interval == 0:
        for i in range(n_agents):
            torch.save(agent.policies[i].state_dict(), os.path.join(save_path, f"policy_agent{i}_ep{episode+1}.pt"))
            torch.save(agent.values[i].state_dict(), os.path.join(save_path, f"value_agent{i}_ep{episode+1}.pt"))
            torch.save(agent.models[i].state_dict(), os.path.join(save_path, f"model_agent{i}_ep{episode+1}.pt"))

        print(f"[INFO] Saved model checkpoints for all agents at episode {episode+1}")

        total_reward = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_step = 0
        max_step = 0

env.close()
wandb.finish()
