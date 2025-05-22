import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
import argparse
import wandb
import random
import numpy as np

from RL_Algorithm.Function_based.MAMBA2_flappy import Mamba2SingleAgent

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- Config ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without rendering")
args = parser.parse_args()

set_seed(42)
lidar_flag = 1
render_mode = None if args.headless else "human"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=lidar_flag)
env.reset(seed=42)
env.action_space.seed(42)
env.observation_space.seed(42)

n_observations = 180 if lidar_flag else 12
n_actions = 2
learning_rate = 1e-4
hidden_dim = 256
discount_factor = 0.99
n_episodes = 50000
max_steps_per_episode = 1000
save_interval = 100

save_path = f"./saved_models_real/mamba2({'lidar' if lidar_flag else 'normal'})-lr{learning_rate:.0e}-dis{discount_factor}-n_eps{n_episodes}"
os.makedirs(save_path, exist_ok=True)

agent = Mamba2SingleAgent(
    obs_dim=n_observations,
    act_dim=n_actions,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    gamma=discount_factor
)

wandb.init(
    project="flappy-bird-rl-real",
    name=f"MAMBA2({'lidar' if lidar_flag else 'normal'})-lr{learning_rate:.0e}-dis{discount_factor}-n_eps{n_episodes}",
    config={
        "algorithm": "MAMBA2-SingleAgent",
        "n_observations": n_observations,
        "n_actions": n_actions,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "gamma": discount_factor,
    }
)

total_reward, total_policy_loss, total_value_loss, total_step, max_step = 0, 0, 0, 0, 0

for episode in tqdm(range(n_episodes)):
    (episode_reward, policy_loss, value_loss), step_count = agent.learn(env=env, max_steps=max_steps_per_episode)

    wandb.log({
        "reward": episode_reward,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "step_count": step_count,
        "episode": episode
    })

    total_reward += episode_reward
    total_policy_loss += policy_loss
    total_value_loss += value_loss
    total_step += step_count
    max_step = max(max_step, step_count)

    if (episode + 1) % 100 == 0:
        wandb.log({
            "avg_reward": total_reward / 100,
            "avg_policy_loss": total_policy_loss / 100,
            "avg_value_loss": total_value_loss / 100,
            "avg_step_count": total_step / 100,
            "max_step": max_step,
            "episode": episode + 1
        })

        print(f"[Episode {episode + 1}] Avg Reward: {total_reward / 100:.2f} | "
              f"Policy Loss: {total_policy_loss / 100:.4f} | Value Loss: {total_value_loss / 100:.4f} | "
              f"Max Step: {max_step}")

        total_reward, total_policy_loss, total_value_loss, total_step, max_step = 0, 0, 0, 0, 0

    if (episode + 1) % save_interval == 0:
        torch.save(agent.policy.state_dict(), os.path.join(save_path, f"mamba2_policy_ep{episode+1}.pt"))
        print(f"[INFO] Saved policy checkpoint at episode {episode+1}")

env.close()
wandb.finish()
