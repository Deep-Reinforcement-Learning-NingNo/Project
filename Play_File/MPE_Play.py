# import torch
# import numpy as np
# import time
# from mpe2 import simple_spread_v3
# from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# # --- Parameters ---
# N = 3
# local_ratio = 0.5
# max_cycles = 100
# continuous_actions = False
# hidden_dim = 256
# model_path = "saved_models/dynappo_mpe_simple_spread-lr1e-04-hdim256-gamma0.99-clip0.2-epochs6-eps5000-maxstep100-local0.5-cycles100/dyna_ppo_policy_ep1700.pt"  # <== adjust this if needed
# n_episodes = 5000

# # --- Create env ---
# env = simple_spread_v3.parallel_env(
#     N=N,
#     local_ratio=local_ratio,
#     max_cycles=max_cycles,
#     continuous_actions=continuous_actions,
#     render_mode="human"  
# )
# env.reset()
# # env.render_mode = "human"

# for agent in env.unwrapped.world.agents:
#     agent.size = 0.05  # or whatever value you used during training

# for i, landmark in enumerate(env.unwrapped.world.landmarks):
#     landmark.size = 0.05
#     landmark.movable = False      
#     landmark.collide = False
#     landmark.fixed_pos = np.array([i * 1.0 - 1.0, 0.0])  # 🔧 ensure fixed_pos is re-initialized
#     landmark.state.p_pos = landmark.fixed_pos
#     landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

# # --- Get metadata ---
# agent_names = env.agents
# observations, _ = env.reset()
# sample_obs = observations[agent_names[0]]
# obs_dim = sample_obs.shape[0]
# act_dim = env.action_space(agent_names[0]).n

# # --- Load trained agent ---
# agent = MultiAgentDynaPPO(
#     n_agents=N,
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     hidden_dim=hidden_dim,
# )
# agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
# agent.policy.eval()

# # --- Play loop ---
# for ep in range(n_episodes):
#     observations, _ = env.reset()
#     total_rewards = {agent_name: 0.0 for agent_name in agent_names}
    
#     for step in range(max_cycles):
#         env.render()
#         # time.sleep(0.1)  # slow down to see movement

#         # convert observations to array
#         obs_array = np.array([observations[agent_name] for agent_name in agent_names])
#         obs_tensor = torch.FloatTensor(obs_array).to(agent.device)

#         with torch.no_grad():
#             probs = agent.policy(obs_tensor)
#             dist = torch.distributions.Categorical(probs)
#             actions_array = dist.sample().cpu().numpy()

#         # convert actions to dict
#         actions = {agent_name: actions_array[i] for i, agent_name in enumerate(agent_names)}

#         # step environment
#         observations, rewards, terminations, truncations, _ = env.step(actions)

#         for name in agent_names:
#             total_rewards[name] += rewards[name]

#         if all(terminations.values()) or all(truncations.values()):
#             break

#     print(f"[Episode {ep+1}] Total reward: {[f'{name}: {r:.2f}' for name, r in total_rewards.items()]}")

# env.close()

# import torch
# import numpy as np
# import time
# from mpe2 import simple_spread_v3
# from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# # --- Parameters ---
# N = 3
# local_ratio = 0.5
# max_cycles = 100
# continuous_actions = False
# hidden_dim = 256
# model_path = "saved_models/dynappo_mpe_simple_spread-lr1e-04-hdim256-gamma0.99-clip0.2-epochs6-eps5000-maxstep100-local0.5-cycles100/dyna_ppo_policy_ep1700.pt"
# n_episodes = 5000

# # --- Create environment ---
# env = simple_spread_v3.parallel_env(
#     N=N,
#     local_ratio=local_ratio,
#     max_cycles=max_cycles,
#     continuous_actions=continuous_actions,
#     render_mode="human",
#     dynamic_rescaling = False
# )

# _ = env.reset()
# agent_names = env.agents

# reset_output = env.reset()
# if isinstance(reset_output, tuple):
#     observations, _ = reset_output
# else:
#     observations = reset_output

# # --- Fix landmark properties ---
# for i, landmark in enumerate(env.unwrapped.world.landmarks):
#     landmark.size = 0.05
#     landmark.movable = False
#     landmark.collide = False
#     landmark.fixed_pos = np.array([i * 1.0 - 1.0, 0.0])
#     landmark.state.p_pos = landmark.fixed_pos
#     landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

# # --- Fix agent size (visual only) ---
# for agent in env.unwrapped.world.agents:
#     agent.size = 0.05

# # --- Observation/action dimensions ---
# sample_obs = observations[agent_names[0]]
# obs_dim = sample_obs.shape[0]
# act_dim = env.action_space(agent_names[0]).n

# # --- Load trained agent ---
# agent = MultiAgentDynaPPO(
#     n_agents=N,
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     hidden_dim=hidden_dim,
# )
# agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
# agent.policy.eval()

# # --- Play loop ---
# for ep in range(n_episodes):
#     reset_output = env.reset()
#     if isinstance(reset_output, tuple):
#         observations, _ = reset_output
#     else:
#         observations = reset_output

#     # ✅ Reset landmark position manually to fixed_pos
#     for i, landmark in enumerate(env.unwrapped.world.landmarks):
#         landmark.state.p_pos = landmark.fixed_pos
#         landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

#     # 🖨️ Print landmark positions
#     print(f"[Episode {ep+1}] Landmark positions:")
#     for i, landmark in enumerate(env.unwrapped.world.landmarks):
#         pos = landmark.state.p_pos
#         print(f"  Landmark {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

#     # 🖨️ Print agent positions
#     print(f"[Episode {ep+1}] Agent positions:")
#     for i, a in enumerate(env.unwrapped.world.agents):
#         pos = a.state.p_pos
#         print(f"  Agent {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

#     total_rewards = {agent_name: 0.0 for agent_name in agent_names}

#     for step in range(max_cycles):
#         env.render()
#         # time.sleep(0.1)

#         obs_array = np.array([observations[name] for name in agent_names])
#         obs_tensor = torch.FloatTensor(obs_array).to(agent.device)

#         with torch.no_grad():
#             probs = agent.policy(obs_tensor)
#             dist = torch.distributions.Categorical(probs)
#             actions_array = dist.sample().cpu().numpy()

#         actions = {name: actions_array[i] for i, name in enumerate(agent_names)}
#         observations, rewards, terminations, truncations, _ = env.step(actions)

#         for name in agent_names:
#             total_rewards[name] += rewards[name]

#         if all(terminations.values()) or all(truncations.values()):
#             break

#     print(f"[Episode {ep+1}] Total reward: {[f'{name}: {r:.2f}' for name, r in total_rewards.items()]}")
#     print("-" * 60)

# env.close()


##########################   โค้ดนี้ใช้ได้ดี 

import torch
import numpy as np
import time
from mpe2 import simple_spread_v3
from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# --- Parameters ---
N = 3
local_ratio = 1.0
max_cycles = 100
continuous_actions = False
hidden_dim = 256
model_path = "/home/nigo/Desktop/Project/saved_models/dynappo_mpe_simple_spread-lr1e-04-hdim256-gamma0.95-clip0.1-epochs10-eps5000-maxstep150-local1-cycles150/dyna_ppo_policy_ep5000.pt"
n_episodes = 5000

# --- Create environment ---
env = simple_spread_v3.parallel_env(
    N=N,
    local_ratio=local_ratio,
    max_cycles=max_cycles,
    continuous_actions=continuous_actions,
    render_mode="human",
    dynamic_rescaling=False
)

_ = env.reset()
agent_names = env.agents

reset_output = env.reset()
if isinstance(reset_output, tuple):
    observations, _ = reset_output
else:
    observations = reset_output

# --- Fix landmark properties ---
for i, landmark in enumerate(env.unwrapped.world.landmarks):
    landmark.size = 0.05
    landmark.movable = False
    landmark.collide = False
    # landmark.fixed_pos = np.array([i * 0.5 - 0.5, 0.0]) 
    # landmark.state.p_pos = landmark.fixed_pos
    landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

# --- Fix agent visual size ---
for agent in env.unwrapped.world.agents:
    agent.size = 0.05

# --- Observation/action dimensions ---
sample_obs = observations[agent_names[0]]
obs_dim = sample_obs.shape[0]
act_dim = env.action_space(agent_names[0]).n

# --- Load trained agent ---
agent = MultiAgentDynaPPO(
    n_agents=N,
    obs_dim=obs_dim,
    act_dim=act_dim,
    hidden_dim=hidden_dim,
)
agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
agent.policy.eval()

# --- Play loop ---
for ep in range(n_episodes):
    reset_output = env.reset()
    if isinstance(reset_output, tuple):
        observations, _ = reset_output
    else:
        observations = reset_output

    # Reset landmark positions manually
    for i, landmark in enumerate(env.unwrapped.world.landmarks):
        landmark.state.p_pos = landmark.fixed_pos
        landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

    # 🖨️ Print landmark positions
    print(f"[Episode {ep+1}] Landmark positions:")
    for i, landmark in enumerate(env.unwrapped.world.landmarks):
        pos = landmark.state.p_pos
        print(f"  Landmark {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

    total_rewards = {agent_name: 0.0 for agent_name in agent_names}

    for step in range(max_cycles):
        env.render()
        # time.sleep(0.05)  # เพิ่มถ้าอยากดูช้าลง

        # 🖨️ Real-time agent positions
        for i, a in enumerate(env.unwrapped.world.agents):
            pos = a.state.p_pos
            print(f"[Ep {ep+1} | Step {step+1}] Agent {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

        # Clamp agent position to keep in [-1, 1]
        # for a in env.unwrapped.world.agents:
        #     a.state.p_pos = np.clip(a.state.p_pos, -1.0, 1.0)

        obs_array = np.array([observations[name] for name in agent_names])
        obs_tensor = torch.FloatTensor(obs_array).to(agent.device)

        with torch.no_grad():
            probs = agent.policy(obs_tensor)
            dist = torch.distributions.Categorical(probs)
            actions_array = dist.sample().cpu().numpy()

        actions = {name: actions_array[i] for i, name in enumerate(agent_names)}
        observations, rewards, terminations, truncations, _ = env.step(actions)

        for name in agent_names:
            total_rewards[name] += rewards[name]

        if all(terminations.values()) or all(truncations.values()):
            break

    print(f"[Episode {ep+1}] Total reward: {[f'{name}: {r:.2f}' for name, r in total_rewards.items()]}")
    print("-" * 60)

env.close()



# import torch
# import numpy as np
# import time
# from mpe2 import simple_spread_v3
# from RL_Algorithm.Function_based.MAPPO_MAMBA2V2 import MultiAgentDynaPPO

# # --- Parameters ---
# N = 3
# local_ratio = 1
# max_cycles = 150
# continuous_actions = False
# hidden_dim = 256
# model_path = "./saved_models/dynappo_mpe_simple_spread-lr1e-04-hdim256-gamma0.95-clip0.1-epochs10-eps5000-maxstep150-local1-cycles150random_landmark1/dyna_ppo_policy_ep3700.pt"
# n_episodes = 5000

# # --- Create environment ---
# env = simple_spread_v3.parallel_env(
#     N=N,
#     local_ratio=local_ratio,
#     max_cycles=max_cycles,
#     continuous_actions=continuous_actions,
#     render_mode="human",
#     dynamic_rescaling=False
# )

# _ = env.reset()
# agent_names = env.agents

# reset_output = env.reset()
# if isinstance(reset_output, tuple):
#     observations, _ = reset_output
# else:
#     observations = reset_output

# # --- Fix landmark properties ---
# for i, landmark in enumerate(env.unwrapped.world.landmarks):
#     landmark.size = 0.05
#     landmark.movable = False
#     landmark.collide = False
#     landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

# # --- Fix agent visual size ---
# for agent in env.unwrapped.world.agents:
#     agent.size = 0.05

# # --- Observation/action dimensions ---
# sample_obs = observations[agent_names[0]]
# obs_dim = sample_obs.shape[0]
# act_dim = env.action_space(agent_names[0]).n

# # --- Load trained agent ---
# agent = MultiAgentDynaPPO(
#     n_agents=N,
#     obs_dim=obs_dim,
#     act_dim=act_dim,
#     hidden_dim=hidden_dim,
# )
# agent.policy.load_state_dict(torch.load(model_path, map_location=agent.device))
# agent.policy.eval()

# # --- Play loop ---
# for ep in range(n_episodes):
#     reset_output = env.reset()
#     if isinstance(reset_output, tuple):
#         observations, _ = reset_output
#     else:
#         observations = reset_output

#     # Randomize landmark positions each episode (เหมือน training)
#     for landmark in env.unwrapped.world.landmarks:
#         landmark.state.p_pos = np.random.uniform(-1.0, 1.0, size=env.unwrapped.world.dim_p)
#         landmark.state.p_vel = np.zeros(env.unwrapped.world.dim_p)

#     # Print landmark positions
#     print(f"[Episode {ep+1}] Landmark positions:")
#     for i, landmark in enumerate(env.unwrapped.world.landmarks):
#         pos = landmark.state.p_pos
#         print(f"  Landmark {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

#     total_rewards = {agent_name: 0.0 for agent_name in agent_names}

#     for step in range(max_cycles):
#         env.render()
#         # time.sleep(0.05)  # ใช้เพิ่มความช้าในการดูถ้าต้องการ

#         # Print agent positions real-time
#         for i, a in enumerate(env.unwrapped.world.agents):
#             pos = a.state.p_pos
#             print(f"[Ep {ep+1} | Step {step+1}] Agent {i}: x = {pos[0]:.2f}, y = {pos[1]:.2f}")

#         obs_array = np.array([observations[name] for name in agent_names])
#         obs_tensor = torch.FloatTensor(obs_array).to(agent.device)

#         with torch.no_grad():
#             probs = agent.policy(obs_tensor)
#             dist = torch.distributions.Categorical(probs)
#             actions_array = dist.sample().cpu().numpy()

#         actions = {name: actions_array[i] for i, name in enumerate(agent_names)}
#         observations, rewards, terminations, truncations, _ = env.step(actions)

#         for name in agent_names:
#             total_rewards[name] += rewards[name]

#         if all(terminations.values()) or all(truncations.values()):
#             break

#     print(f"[Episode {ep+1}] Total reward: {[f'{name}: {r:.2f}' for name, r in total_rewards.items()]}")
#     print("-" * 60)

# env.close()
