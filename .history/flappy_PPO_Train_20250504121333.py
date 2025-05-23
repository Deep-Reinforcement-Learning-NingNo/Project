
import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
import argparse
import wandb
import random
import numpy as np

from RL_Algorithm.Function_based.PPO import PPO 

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 42

# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without rendering")
args = parser.parse_args()

# --- Environment setup ---
set_seed(seed)
render_mode = None if args.headless else "human"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=True)
env.reset(seed=seed)
env.action_space.seed(seed)
env.observation_space.seed(seed)

# --- Hyperparameters ---
num_of_action = 2
n_observations =  12 # normal
n_observations =  180 # lidar
learning_rate = 1e-4
hidden_dim = 256
discount_factor = 0.99
clip_epsilon = 0.1
epochs = 6
batch_size = 32
buffer_size = 5000
n_episodes = 50000
max_steps_per_episode = 1000
save_interval = 100
save_path = f"./saved_models/ppo(lidar)-lr{learning_rate:.0e}-bs{batch_size}-clip{clip_epsilon}-ep{epochs}-n_eps{n_episodes}"
os.makedirs(save_path, exist_ok=True)

# --- Init PPO agent ---
agent = PPO(
    n_observations=n_observations,
    n_actions=num_of_action,
    hidden_dim=hidden_dim,
    learning_rate=learning_rate,
    gamma=discount_factor,
    clip_epsilon=clip_epsilon,
    epochs=epochs,
    buffer_size=buffer_size,
    batch_size=batch_size,
)

# --- Init WandB ---
wandb.init(
    project="flappy-bird-rl",
    name=f"PPO-lr{learning_rate:.0e}-bs{batch_size}-clip{clip_epsilon}-ep{epochs}-n_eps{n_episodes}",
    config={
        "algorithm": "PPO",
        "n_observations": n_observations,
        "n_actions": num_of_action,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "gamma": discount_factor,
        "clip_epsilon": clip_epsilon,
        "epochs": epochs,
        "batch_size": batch_size,
    }
)

# Temporary accumulators
total_reward = 0
total_policy_loss = 0
total_value_loss = 0
total_step = 0
max_step = 0

# --- Training Loop ---
for episode in tqdm(range(n_episodes)):
    episode_reward, policy_loss, value_loss, step_count = agent.learn(
        env=env,
        max_steps=max_steps_per_episode
    )

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
        avg_reward = total_reward / 100
        avg_policy_loss = total_policy_loss / 100
        avg_value_loss = total_value_loss / 100
        avg_step = total_step / 100

        wandb.log({
            "avg_reward": avg_reward,
            "avg_policy_loss": avg_policy_loss,
            "avg_value_loss": avg_value_loss,
            "avg_max_step": avg_step,
            "max_step": max_step,
            "episode": episode + 1
        })

        print(f"[Episode {episode + 1}] Avg Reward: {avg_reward:.2f} | "
              f"Policy Loss: {avg_policy_loss:.4f} | Value Loss: {avg_value_loss:.4f} | "
              f"Max Step: {max_step}")

        total_reward = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_step = 0
        max_step = 0

    if (episode + 1) % save_interval == 0:
        policy_path = os.path.join(save_path, f"ppo_policy_ep{episode+1}.pt")
        torch.save(agent.policy.state_dict(), policy_path)
        print(f"[INFO] Saved PPO policy checkpoint at {policy_path}")

env.close()
wandb.finish()
