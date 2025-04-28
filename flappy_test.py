import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
from RL_Algorithm.Function_based.DDPG import DDPG  # <<<<<< DDPG agent ที่เราเขียนไว้แล้ว

# --- Set up environment ---
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# --- Set up DDPG agent ---
agent = DDPG(
    n_observations=12,  # From your obs size
    n_actions=1,
    hidden_dim=256,
    action_range=[-1.0, 1.0],
    learning_rate=1e-4,
    tau=0.005,
    discount_factor=0.99,
    buffer_size=50000,
    batch_size=64,
)

# --- Hyperparameters ---
n_episodes = 1000
max_steps = 1000
start_noise = 0.2
noise_decay = 0.995
min_noise = 0.01
save_interval = 100
save_path = "./saved_models_ddpg_flappy"
os.makedirs(save_path, exist_ok=True)

# --- Training Loop ---
for episode in tqdm(range(n_episodes)):
    obs, _ = env.reset()
    episode_reward = 0
    noise = max(start_noise * (noise_decay ** episode), min_noise)

    for step in range(max_steps):
        action = agent.select_action(obs, noise=noise)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # Store in Replay Buffer
        agent.memory.add(obs, [action], reward, next_obs, done)

        # Update policy
        agent.update()

        obs = next_obs
        episode_reward += reward

        if done:
            break

    print(f"Episode {episode} | Reward: {episode_reward:.2f} | Noise: {noise:.3f}")

    # Save model
    if (episode + 1) % save_interval == 0:
        actor_path = os.path.join(save_path, f"ddpg_actor_ep{episode+1}.pt")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f"[INFO] Saved actor checkpoint at {actor_path}")

env.close()
