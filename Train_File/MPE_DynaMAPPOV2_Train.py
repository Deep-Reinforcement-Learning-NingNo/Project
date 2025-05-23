
# import torch
# import numpy as np
# import random
# import os
# from tqdm import tqdm
# import wandb
# import argparse

# from mpe2 import simple_spread_v3
# import supersuit as ss

# from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# # --- Utils ---
# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)

# # --- Args ---
# parser = argparse.ArgumentParser()
# parser.add_argument("--headless", action="store_true", help="No render")
# args = parser.parse_args()

# # --- Config ---
# N = 3
# local_ratio = 1
# max_cycles = 150
# continuous_actions = False

# # --- Create base_env for metadata --
# base_env = simple_spread_v3.parallel_env(
#     N=N,
#     local_ratio=local_ratio,
#     max_cycles=max_cycles,
#     continuous_actions=continuous_actions
# )
# n_agents = len(base_env .possible_agents)

# for agent in base_env .unwrapped.world.agents:
#     agent.size = 0.05  # or whatever value you used during training

# for i, landmark in enumerate(base_env .unwrapped.world.landmarks):
#     landmark.size = 0.05
#     landmark.movable = False      
#     landmark.collide = False
#     # landmark.fixed_pos = np.array([i * 0.5 - 0.5, 0.0])  # 🔧 ensure fixed_pos is re-initialized
#     # landmark.state.p_pos = landmark.fixed_pos
#     landmark.state.p_vel = np.zeros(base_env .unwrapped.world.dim_p)

# # --- Supersuit Vectorized Env ---
# env = ss.pettingzoo_env_to_vec_env_v1(base_env )
# env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="gymnasium")

# # --- Dimensions ---
# n_observations = env.observation_space.shape[0]
# n_actions = env.action_space.shape[0] if continuous_actions else env.action_space.n

# # --- Hyperparameters ---
# learning_rate = 1e-4
# hidden_dim = 256
# discount_factor = 0.95 # เดิม 0.99 
# clip_epsilon = 0.1 #เดิม 0.2 
# epochs = 10 #เดิม 6
# n_episodes = 5000
# max_steps_per_episode = max_cycles
# save_interval = 100
# save_path = (
#     f"./saved_models/dynappo_mpe_simple_spread-"
#     f"lr{learning_rate:.0e}-"
#     f"hdim{hidden_dim}-"
#     f"gamma{discount_factor}-"
#     f"clip{clip_epsilon}-"
#     f"epochs{epochs}-"
#     f"eps{n_episodes}-"
#     f"maxstep{max_steps_per_episode}-"
#     f"local{local_ratio}-"
#     f"cycles{max_cycles}"
# )
# wandb_name = (
#     f"DynaPPO-MPE-simple_spread-"
#     f"lr{learning_rate:.0e}-"
#     f"hdim{hidden_dim}-"
#     f"gamma{discount_factor}-"
#     f"clip{clip_epsilon}-"
#     f"epochs{epochs}-"
#     f"eps{n_episodes}-"
#     f"maxstep{max_steps_per_episode}-"
#     f"local{local_ratio}-"
#     f"cycles{max_cycles}"
# )

# os.makedirs(save_path, exist_ok=True)
# set_seed(42)

# # --- WandB Init ---
# wandb.init(
#     project="mpe-multi-agent",
#     name=wandb_name,
#     config={
#         "env": "simple_spread_v3",
#         "n_agents": n_agents,
#         "n_observations": n_observations,
#         "n_actions": n_actions,
#         "hidden_dim": hidden_dim,
#         "learning_rate": learning_rate,
#         "gamma": discount_factor,
#         "clip_epsilon": clip_epsilon,
#         "epochs": epochs,
#     }
# )

# # --- Agent Init ---
# agent = MultiAgentDynaPPO(
#     n_agents=n_agents,
#     obs_dim=n_observations,
#     act_dim=n_actions,
#     hidden_dim=hidden_dim,
#     lr=learning_rate,
#     gamma=discount_factor,
#     clip_eps=clip_epsilon,
#     epochs=epochs,
# )

# # --- Train Loop ---
# total_reward, total_policy_loss, total_value_loss, total_step = 0, 0, 0, 0
# max_step = 10000

# for episode in tqdm(range(n_episodes)):
#     ep_rewards, policy_loss, value_loss, step_count = agent.learn(env=env, max_steps=max_steps_per_episode)

#     print(f"[DEBUG] Landmark positions at episode {episode+1}:")
#     for i, landmark in enumerate(base_env.unwrapped.world.landmarks):
#         print(f"  Landmark {i}: x = {landmark.state.p_pos[0]:.2f}, y = {landmark.state.p_pos[1]:.2f}")

#     print(f"[DEBUG] Agent positions at episode {episode+1}:")
#     for i, a in enumerate(base_env.unwrapped.world.agents):
#         print(f"  Agent {i}: x = {a.state.p_pos[0]:.2f}, y = {a.state.p_pos[1]:.2f}")

#     wandb_log = {
#         "avg_reward": np.mean(ep_rewards),
#         "policy_loss": policy_loss,
#         "value_loss": value_loss,
#         "step_count": step_count,
#         "episode": episode
#     }

#     for i, r in enumerate(ep_rewards):
#         wandb_log[f"agent_{i}_reward"] = r

#     wandb.log(wandb_log)

#     agent_reward_str = " | ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(ep_rewards)])
#     print(f"[Ep {episode+1}] {agent_reward_str} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Steps: {step_count}")

#     total_reward += np.mean(ep_rewards)
#     total_policy_loss += policy_loss
#     total_value_loss += value_loss
#     total_step += step_count
#     max_step = max(max_step, step_count)

#     if (episode + 1) % 100 == 0:
#         print(f"[SUMMARY] Ep {episode+1} | Avg Reward: {total_reward/100:.2f} | Avg Steps: {total_step/100:.1f} | Max Step: {max_step}")
#         total_reward, total_policy_loss, total_value_loss, total_step, max_step = 0, 0, 0, 0, 0

#     if (episode + 1) % save_interval == 0:
#         path = os.path.join(save_path, f"dyna_ppo_policy_ep{episode+1}.pt")
#         torch.save(agent.policy.state_dict(), path)
#         print(f"[INFO] Saved checkpoint at {path}")

# env.close()
# wandb.finish()




import torch
import numpy as np
import random
import os
from tqdm import tqdm
import wandb
import argparse

from mpe2 import simple_spread_v3
import supersuit as ss

from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# --- Utils ---
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# --- Args ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="No render")
args = parser.parse_args()

# --- Config ---
N = 3
local_ratio = 1
max_cycles = 150
continuous_actions = False

# --- Create base_env for metadata --
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

# --- Supersuit Vectorized Env ---
env = ss.pettingzoo_env_to_vec_env_v1(base_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="gymnasium")

# --- Dimensions ---
n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0] if continuous_actions else env.action_space.n

# --- Hyperparameters ---
random_landmark = 1 #1 = random 0 = ไม่ random
learning_rate = 1e-4
hidden_dim = 256
discount_factor = 0.95
clip_epsilon = 0.1
epochs = 10
n_episodes = 5000
max_steps_per_episode = max_cycles
save_interval = 100

save_path = (
    f"./saved_models/dynappo_mpe_simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"clip{clip_epsilon}-"
    f"epochs{epochs}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}"
    f"random_landmark{random_landmark}"
)
wandb_name = (
    f"DynaPPO-MPE-simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"clip{clip_epsilon}-"
    f"epochs{epochs}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}"
    f"random_landmark{random_landmark}"
)

os.makedirs(save_path, exist_ok=True)
set_seed(42)

# --- WandB Init ---
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
        "clip_epsilon": clip_epsilon,
        "epochs": epochs,
    }
)

# --- Agent Init ---
agent = MultiAgentDynaPPO(
    n_agents=n_agents,
    obs_dim=n_observations,
    act_dim=n_actions,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    gamma=discount_factor,
    clip_eps=clip_epsilon,
    epochs=epochs,
)

# --- Train Loop ---
total_reward, total_policy_loss, total_value_loss, total_step = 0, 0, 0, 0
max_step = 10000

for episode in tqdm(range(n_episodes)):
    # Randomize landmark positions at start of each episode
    for landmark in base_env.unwrapped.world.landmarks:
        landmark.state.p_pos = np.random.uniform(-1.0, 1.0, size=base_env.unwrapped.world.dim_p)
        landmark.state.p_vel = np.zeros(base_env.unwrapped.world.dim_p)

    ep_rewards, policy_loss, value_loss, step_count = agent.learn(env=env, max_steps=max_steps_per_episode)

    print(f"[DEBUG] Landmark positions at episode {episode+1}:")
    for i, landmark in enumerate(base_env.unwrapped.world.landmarks):
        print(f"  Landmark {i}: x = {landmark.state.p_pos[0]:.2f}, y = {landmark.state.p_pos[1]:.2f}")

    print(f"[DEBUG] Agent positions at episode {episode+1}:")
    for i, a in enumerate(base_env.unwrapped.world.agents):
        print(f"  Agent {i}: x = {a.state.p_pos[0]:.2f}, y = {a.state.p_pos[1]:.2f}")

    wandb_log = {
        "avg_reward": np.mean(ep_rewards),
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "step_count": step_count,
        "episode": episode
    }

    for i, r in enumerate(ep_rewards):
        wandb_log[f"agent_{i}_reward"] = r

    wandb.log(wandb_log)

    agent_reward_str = " | ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(ep_rewards)])
    print(f"[Ep {episode+1}] {agent_reward_str} | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Steps: {step_count}")

    total_reward += np.mean(ep_rewards)
    total_policy_loss += policy_loss
    total_value_loss += value_loss
    total_step += step_count
    max_step = max(max_step, step_count)

    if (episode + 1) % 100 == 0:
        print(f"[SUMMARY] Ep {episode+1} | Avg Reward: {total_reward/100:.2f} | Avg Steps: {total_step/100:.1f} | Max Step: {max_step}")
        total_reward, total_policy_loss, total_value_loss, total_step, max_step = 0, 0, 0, 0, 0

    if (episode + 1) % save_interval == 0:
        path = os.path.join(save_path, f"dyna_ppo_policy_ep{episode+1}.pt")
        torch.save(agent.policy.state_dict(), path)
        print(f"[INFO] Saved checkpoint at {path}")

env.close()
wandb.finish()
