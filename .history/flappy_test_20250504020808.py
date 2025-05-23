import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
from RL_Algorithm.Function_based.DDPG import DDPG  # <<<<<< DDPG agent ที่เราเขียนไว้แล้ว
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run without rendering")
args = parser.parse_args()
# --- Set up environment ---
render_mode = None if args.headless else "human"
env = gym.make("FlappyBird-v0", render_mode=render_mode, use_lidar=False)

num_of_action = 1
action_range = [-1.0, 1.0]  
n_observations = 12
learning_rate = 0.001 #ทดลองปรับ
hidden_dim = 256  #ทดลองปรับ
tau = 0.005 
buffer_size = 5000 
batch_size = 64 #ทดลองปรับ
discount_factor = 0.99 #ทดลองปรับ
noise_scale_init = 0.2 #ทดลองปรับ
noise_decay  = 0.5 #ทดลองปรับ 
n_episodes = 10000
max_steps_per_episode = 1000

# --- Set up DDPG agent ---
agent = DDPG(
    n_observations=n_observations,  # From your obs size
    n_actions=num_of_action,
    hidden_dim=hidden_dim,
    action_range=action_range,
    learning_rate=learning_rate,
    tau=tau,
    discount_factor=discount_factor,
    buffer_size=buffer_size,
    batch_size=batch_size ,
)

# --- Hyperparameters ---
n_episodes = 10000
max_steps = 10000
save_interval = 100
save_path = "./saved_models_ddpg_flappy"
os.makedirs(save_path, exist_ok=True)

# --- Training Loop ---
for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    episode_reward = 0
    # noise = max(start_noise * (noise_decay ** episode), min_noise)

    for step in range(max_steps):
        ep_reward,a_loss,c_loss = agent.learn(
           env=env,
                max_steps=max_steps_per_episode,
                num_agents=num_agents,
                noise_scale=noise_scale_init,
                noise_decay=noise_decay,
            )

        if done:
            break

    print(f"Episode {episode} | Reward: {episode_reward:.2f} | Noise: {noise:.3f}")

    # Save model
    if (episode + 1) % save_interval == 0:
        actor_path = os.path.join(save_path, f"ddpg_actor_ep{episode+1}.pt")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f"[INFO] Saved actor checkpoint at {actor_path}")

env.close()
