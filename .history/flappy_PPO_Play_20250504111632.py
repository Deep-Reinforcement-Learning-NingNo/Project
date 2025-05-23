import torch
import gymnasium as gym
import flappy_bird_gymnasium
import argparse
from RL_Algorithm.Function_based.PPO import PolicyNetwork

# --- Arguments ---
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True, help="Path to saved PPO policy .pt file")
args = parser.parse_args()

# --- Env Setup ---
env = gym.make("FlappyBird-v0", render_mode="human", use_lidar=False)
obs_dim = 12     # depends on your environment
action_dim = 2   # PPO uses discrete action: 0 or 1
hidden_dim = 256 # must match your training config

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy = PolicyNetwork(obs_dim, hidden_dim, action_dim).to(device)
policy.load_state_dict(torch.load(args.model, map_location=device))
policy.eval()

# --- Run the agent ---
while True:
    obs, _ = env.reset()
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
