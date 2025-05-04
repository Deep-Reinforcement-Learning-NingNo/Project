import flappy_bird_gymnasium
import gymnasium as gym
import torch
import os
from tqdm import tqdm
from RL_Algorithm.Function_based.DDPG import DDPG  
import argparse
import wandb

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
noise_scale_init = 0.05 #ทดลองปรับ
noise_decay  = 0.99 #ทดลองปรับ 
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

wandb.init(
    project="flappy-bird-rl",       
    name=f"DDPG-run-{wandb.util.generate_id()}",  
    config={
        "algorithm": "DDPG",
        "n_observations": n_observations,
        "n_actions": num_of_action,
        "hidden_dim": hidden_dim,
        "learning_rate": learning_rate,
        "tau": tau,
        "discount_factor": discount_factor,
        "buffer_size": buffer_size,
        "batch_size": batch_size,
        "noise_scale": noise_scale_init,
        "noise_decay": noise_decay,
    }
)
# --- Accumulators ---
total_reward = 0
total_actor_loss = 0
total_critic_loss = 0
valid_loss_count = 0  # in case some losses are None

# --- Training Loop ---
for episode in tqdm(range(n_episodes)):
    episode_reward, a_loss, c_loss = agent.learn(
        env=env,
        max_steps=max_steps_per_episode,
        noise_scale=noise_scale_init,
        noise_decay=noise_decay,
    )
            wandb.log({
            "episode": episode + 1,
            "avg_reward": avg_reward,
            "avg_actor_loss": avg_actor_loss,
            "avg_critic_loss": avg_critic_loss
        })

    total_reward += episode_reward
    if a_loss is not None and c_loss is not None:
        total_actor_loss += a_loss
        total_critic_loss += c_loss
        valid_loss_count += 1

    # Log every 100 episodes
    if (episode + 1) % 100 == 0:
        avg_reward = total_reward / 100
        if valid_loss_count > 0:
            avg_actor_loss = total_actor_loss / valid_loss_count
            avg_critic_loss = total_critic_loss / valid_loss_count
        else:
            avg_actor_loss = avg_critic_loss = 0.0

        wandb.log({
            "episode": episode + 1,
            "avg_reward": avg_reward,
            "avg_actor_loss": avg_actor_loss,
            "avg_critic_loss": avg_critic_loss
        })

        print(f"[Episode {episode + 1}] Avg Reward: {avg_reward:.2f} | "
              f"Actor Loss: {avg_actor_loss:.4f} | Critic Loss: {avg_critic_loss:.4f}")

        total_reward = 0
        total_actor_loss = 0
        total_critic_loss = 0
        valid_loss_count = 0

    # Save model
    if (episode + 1) % save_interval == 0:
        actor_path = os.path.join(save_path, f"ddpg_actor_ep{episode+1}.pt")
        torch.save(agent.actor.state_dict(), actor_path)
        print(f"[INFO] Saved actor checkpoint at {actor_path}")


env.close()
wandb.finish()
