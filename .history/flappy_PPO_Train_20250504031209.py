import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
import argparse
import wandb
import random

from RL_Algorithm.Function_based.PPO import PPO  # ← Make sure you saved PPO.py

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
# --- Parse args ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without rendering")
args = parser.parse_args()

# --- Environment setup ---
render_mode = None if args.headless else "human"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)

# --- Hyperparameters ---
num_of_action = 2  # PPO uses discrete action space → [0, 1]
n_observations = 12
learning_rate = 3e-4
hidden_dim = 256
discount_factor = 0.99
clip_epsilon = 0.2
epochs = 4
batch_size = 64
buffer_size = 5000
n_episodes = 10000
max_steps_per_episode = 1000
save_interval = 100
save_path = "./saved_models_ppo_flappy"
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
    name=f"PPO-run-{wandb.util.generate_id()}",
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

# --- Training Loop ---
for episode in tqdm(range(n_episodes)):
    episode_reward, policy_loss, value_loss = agent.learn(
        env=env,
        max_steps=max_steps_per_episode
    )

    wandb.log({
        "episode": episode,
        "reward": episode_reward,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
    })

    print(f"Episode {episode} | Reward: {episode_reward:.2f}")

    if (episode + 1) % save_interval == 0:
        policy_path = os.path.join(save_path, f"ppo_policy_ep{episode+1}.pt")
        torch.save(agent.policy.state_dict(), policy_path)
        print(f"[INFO] Saved PPO policy checkpoint at {policy_path}")

env.close()
wandb.finish()
