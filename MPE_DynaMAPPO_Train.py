import torch
import numpy as np
import random
import os
from tqdm import tqdm
import wandb
import argparse

from mpe2 import simple_spread_v3
import supersuit as ss

from RL_Algorithm.Function_based.MAPPO_MAMBA import DynaMAPPO  

# --- Utils ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed()

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="No render")
args = parser.parse_args()

# --- Config ---
N = 3
local_ratio = 0.5
max_cycles = 100
continuous_actions = False

# --- Create base_env for metadata ---
base_env = simple_spread_v3.parallel_env(
    N=N,
    local_ratio=local_ratio,
    max_cycles=max_cycles,
    continuous_actions=continuous_actions
)
base_env.reset(seed=42)
n_agents = len(base_env.possible_agents)

# --- Supersuit VecEnv ---
env = ss.pettingzoo_env_to_vec_env_v1(base_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="gymnasium")

# --- Dimensions ---
n_observations = env.observation_space.shape[0]  # <- Fixed to use per-agent obs dim
n_actions = env.action_space.n

# --- Hyperparameters ---
learning_rate = 1e-4
hidden_dim = 256
discount_factor = 0.99
clip_epsilon = 0.2
epochs = 6
n_episodes = 50000
max_steps_per_episode = 1000
save_interval = 100
save_path = f"./saved_models/dyna_mappo_mpe"
wandb_name = f"DynaMAPPO-MPE-simple_spread-v3"

os.makedirs(save_path, exist_ok=True)

# --- WandB Init ---
wandb.init(
    project="mpe-dynamappo",
    name=wandb_name,
    config={
        "algorithm": "DynaMAPPO",
        "n_agents": n_agents,
        "n_observations": n_observations,
        "n_actions": n_actions,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "gamma": discount_factor,
        "clip_epsilon": clip_epsilon,
        "epochs": epochs,
    }
)

# --- Init DynaMAPPO Agent ---
agent = DynaMAPPO(
    n_agents=n_agents,
    obs_dim=n_observations,
    act_dim=n_actions,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    gamma=discount_factor,
    clip=clip_epsilon,
    epochs=epochs
)

# --- Training Loop ---
total_reward = 0
total_policy_loss = [0 for _ in range(n_agents)]
total_value_loss = [0 for _ in range(n_agents)]
total_step = 0
max_step = 0

for episode in tqdm(range(n_episodes)):
    reward_total, reward_per_agent, policy_losses, value_losses, step_count = agent.learn(
        env=env,
        max_steps=max_steps_per_episode
    )

    max_step = max(max_step, step_count)
    total_reward += reward_total
    total_step += step_count

    for i in range(n_agents):
        total_policy_loss[i] += policy_losses[i]
        total_value_loss[i] += value_losses[i]

    # --- log ทุก episode ---
    wandb.log({
        "reward": reward_total,
        "step_count": step_count,
        "episode": episode,
        **{f"agent_{i}/reward": reward_per_agent[i] for i in range(n_agents)},
        **{f"agent_{i}/policy_loss": policy_losses[i] for i in range(n_agents)},
        **{f"agent_{i}/value_loss": value_losses[i] for i in range(n_agents)},
    })

    # --- log summary ทุก 100 episodes ---
    if (episode + 1) % 100 == 0:
        avg_policy_loss = [pl / 100 for pl in total_policy_loss]
        avg_value_loss = [vl / 100 for vl in total_value_loss]

        wandb.log({
            "avg_reward": total_reward / 100,
            "avg_step_count": total_step / 100,
            "max_step": max_step,
            **{f"agent_{i}/avg_policy_loss": avg_policy_loss[i] for i in range(n_agents)},
            **{f"agent_{i}/avg_value_loss": avg_value_loss[i] for i in range(n_agents)},
            "episode": episode + 1
        })

        print(f"[Episode {episode+1}] Avg Reward: {total_reward/100:.2f} | " +
              " | ".join([f"Agent {i}: πL={avg_policy_loss[i]:.4f}, VL={avg_value_loss[i]:.4f}"
                          for i in range(n_agents)]) +
              f" | Max Step: {max_step}")

        total_reward = 0
        total_policy_loss = [0 for _ in range(n_agents)]
        total_value_loss = [0 for _ in range(n_agents)]
        total_step = 0
        max_step = 0

    # --- Save checkpoint ---
    if (episode + 1) % save_interval == 0:
        for i in range(n_agents):
            path = os.path.join(save_path, f"dynamappo_agent{i}_ep{episode+1}.pt")
            torch.save(agent.policies[i].state_dict(), path)
        print(f"[INFO] Saved DynaMAPPO policy checkpoint at episode {episode+1}")

env.close()
wandb.finish()
