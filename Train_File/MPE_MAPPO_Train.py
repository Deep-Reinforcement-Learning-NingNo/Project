
import torch
import numpy as np
import random
import os
from tqdm import tqdm
import wandb
import argparse

from mpe2 import simple_spread_v3
import supersuit as ss

from RL_Algorithm.Function_based.MAPPO import MAPPO  # ปรับ path ตามจริง

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="No render")
args = parser.parse_args()

N = 3
local_ratio = 0.5
max_cycles = 100
continuous_actions = False

base_env = simple_spread_v3.parallel_env(
    N=N,
    local_ratio=local_ratio,
    max_cycles=max_cycles,
    continuous_actions=continuous_actions
)
n_agents = len(base_env.possible_agents)

env = ss.pettingzoo_env_to_vec_env_v1(base_env)
env = ss.concat_vec_envs_v1(env, num_vec_envs=1, num_cpus=1, base_class="gymnasium")

n_observations = env.observation_space.shape[0]
n_actions = env.action_space.shape[0] if continuous_actions else env.action_space.n

learning_rate = 1e-4
hidden_dim = 256
discount_factor = 0.99
clip_epsilon = 0.2
epochs = 6
n_episodes = 50000
max_steps_per_episode = 1000
save_interval = 100
save_path = (
    f"./saved_models/mappo_mpe_simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"clip{clip_epsilon}-"
    f"epochs{epochs}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}"
)

wandb_name = (
    f"MAPPO-MPE-simple_spread-"
    f"lr{learning_rate:.0e}-"
    f"hdim{hidden_dim}-"
    f"gamma{discount_factor}-"
    f"clip{clip_epsilon}-"
    f"epochs{epochs}-"
    f"eps{n_episodes}-"
    f"maxstep{max_steps_per_episode}-"
    f"local{local_ratio}-"
    f"cycles{max_cycles}"
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
        "clip_epsilon": clip_epsilon,
        "epochs": epochs,
    }
)

agent = MAPPO(
    n_agents=n_agents,
    obs_dim=n_observations,
    act_dim=n_actions,
    hidden_dim=hidden_dim,
    lr=learning_rate,
    gamma=discount_factor,
    clip_eps=clip_epsilon,
    epochs=epochs,
)

total_reward, total_policy_loss, total_value_loss, total_step = 0, 0, 0, 0
max_step = 0

for episode in tqdm(range(n_episodes)):
    agent_rewards, policy_loss, value_loss, step_count = agent.learn(env=env, max_steps=max_steps_per_episode)

    avg_reward = sum(agent_rewards) / n_agents
    log_dict = {
        "avg_reward": avg_reward,
        "policy_loss": policy_loss,
        "value_loss": value_loss,
        "step_count": step_count,
        "episode": episode
    }

    for i, r in enumerate(agent_rewards):
        log_dict[f"agent_{i}_reward"] = r

    wandb.log(log_dict)

    print(f"[Ep {episode+1}] " + " | ".join([f"Agent {i}: {r:.2f}" for i, r in enumerate(agent_rewards)]) +
          f" | Policy Loss: {policy_loss:.4f} | Value Loss: {value_loss:.4f} | Steps: {step_count}")

    total_reward += avg_reward
    total_policy_loss += policy_loss
    total_value_loss += value_loss
    total_step += step_count
    max_step = max(max_step, step_count)

    if (episode + 1) % save_interval == 0:
        avg_r100 = total_reward / save_interval
        avg_pol100 = total_policy_loss / save_interval
        avg_val100 = total_value_loss / save_interval
        avg_step100 = total_step / save_interval

        wandb.log({
            "avg_reward_100": avg_r100,
            "avg_policy_loss_100": avg_pol100,
            "avg_value_loss_100": avg_val100,
            "avg_step_count_100": avg_step100,
            "max_step_100": max_step,
            "episode": episode + 1
        })

        print(f"[SUMMARY] Ep {episode+1} | AvgReward100: {avg_r100:.2f} | "
              f"PolicyLoss100: {avg_pol100:.4f} | ValueLoss100: {avg_val100:.4f} | "
              f"AvgSteps100: {avg_step100:.1f} | MaxStep100: {max_step}")

        total_reward = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_step = 0
        max_step = 0

        path = os.path.join(save_path, f"mappo_policy_ep{episode+1}.pt")
        torch.save(agent.policy.state_dict(), path)
        print(f"[INFO] Saved checkpoint at {path}")

env.close()
wandb.finish()