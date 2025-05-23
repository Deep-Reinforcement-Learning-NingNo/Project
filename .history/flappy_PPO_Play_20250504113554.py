import torch
import gymnasium as gym
import flappy_bird_gymnasium
from RL_Algorithm.Function_based.PPO import PolicyNetwork
import random
import numpy as np
# --- Config ---
MODEL_PATH = "saved_models_ppo_flappy/ppo_policy_ep1000.pt"  # <<<<<< Set your model path here
OBS_DIM = 12
ACTION_DIM = 2
HIDDEN_DIM = 256

# def set_seed(seed=42):
#     random.seed(seed)
#     np.random.seed(seed)
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)  # for multi-GPU
#     torch.backends.cudnn.deterministic = True
#     torch.backends.cudnn.benchmark = False
    
# set_seed(42)
# --- Env Setup ---
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyNetwork(OBS_DIM, HIDDEN_DIM, ACTION_DIM).to(device)
policy.load_state_dict(torch.load(MODEL_PATH, map_location=device))
policy.eval()

# --- Run the agent ---
while True:
    obs, _ = env.reset(seed=seed)
    done = False
    total_reward = 0

    while not done:
        state = torch.FloatTensor(obs).unsqueeze(0).to(device)
        with torch.no_grad():
            probs = policy(state)
        action = torch.argmax(probs, dim=-1).item()

        obs, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward

    print(f"[INFO] Episode finished with reward: {total_reward:.2f}")
