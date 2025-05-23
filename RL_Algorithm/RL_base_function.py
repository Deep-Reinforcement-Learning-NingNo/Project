import numpy as np
from collections import defaultdict, namedtuple, deque
import random
from enum import Enum
import os
import json
import torch
import torch.nn as nn


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
    
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size = 1):
        """
        Initializes the replay buffer.

        Args:
            buffer_size (int): Maximum number of experiences the buffer can hold.
            batch_size (int): Number of experiences to sample per batch.
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        """
        Adds an experience to the replay buffer.

        Args:
            state (Tensor): The current state of the environment.
            action (Tensor): The action taken at this state.
            reward (Tensor): The reward received after taking the action.
            next_state (Tensor): The next state resulting from the action.
            done (bool): Whether the episode has terminated.
        """

        # append a tuple of ...
        self.memory.append((state, action, reward, next_state, done))

    def sample(self):
        """
        Samples a batch of experiences from the replay buffer.

        Returns:
            - state_batch: Batch of states.
            - action_batch: Batch of actions.
            - reward_batch: Batch of rewards.
            - next_state_batch: Batch of next states.
            - done_batch: Batch of terminal state flags.
        """

        # pick up without replacement from memory up to batch_size
        experiences = random.sample(self.memory, k=self.batch_size)

        # Unpack the experiences
        states, actions, rewards, next_states, dones = zip(*experiences)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        Returns the current size of the replay buffer.

        Returns:
            int: The number of stored experiences.
        """
        return len(self.memory)


class BaseAlgorithm():
    """
    Base class for reinforcement learning algorithms.

    Attributes:
        num_of_action (int): Number of discrete actions available.
        action_range (list): Scale for continuous action mapping.
        discretize_state_scale (list): Scale factors for discretizing states.
        lr (float): Learning rate for updates.
        epsilon (float): Initial epsilon value for epsilon-greedy policy.
        epsilon_decay (float): Rate at which epsilon decays.
        final_epsilon (float): Minimum epsilon value allowed.
        discount_factor (float): Discount factor for future rewards.
        q_values (dict): Q-values for state-action pairs.
        n_values (dict): Count of state-action visits (for Monte Carlo method).
        training_error (list): Stores training errors for analysis.
    """

    def __init__(
        self,
        num_of_action: int = 2,
        action_range: list = [-2.0, 2.0],
        learning_rate: float = 1e-3,
        initial_epsilon: float = 1.0,
        epsilon_decay: float = 1e-3,
        final_epsilon: float = 0.001,
        discount_factor: float = 0.95,
        buffer_size: int = 1000,
        batch_size: int = 1,
    ):
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.num_of_action = num_of_action
        self.action_range = action_range  # [action_min, action_max]

        self.q_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.n_values = defaultdict(lambda: np.zeros(self.num_of_action))
        self.training_error = []

        self.w = np.zeros((4, num_of_action))
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def q(self, obs, a=None):
        """Returns the linearly-estimated Q-value for a given state and action."""
        # ========= put your code here ========= #
        if isinstance(obs, dict) and "policy" in obs:
            obs = obs["policy"]
        if isinstance(obs, torch.Tensor):
            obs = obs.detach().cpu().numpy()

        obs = np.array(obs, dtype=np.float32)

        if a is not None:
            return float(obs.dot(self.w[:, a]))
        else:
            return obs.dot(self.w)
        # ====================================== #
        
    
    # def scale_action(self, action):
    #     """
    #     Maps a discrete action in range [0, n] to a continuous value in [action_min, action_max].

    #     Args:
    #         action (int): Discrete action in range [0, n].
    #         n (int): Number of discrete actions (inclusive range from 0 to n).
        
    #     Returns:
    #         torch.Tensor: Scaled action tensor.
    #     """
    #     # ========= put your code here ========= #
    #     action_min, action_max = self.action_range
    #     # Number of intervals is (num_of_action - 1)
    #     # Fraction in [0, 1] if action is in [0, num_of_action-1]
    #     fraction = action / (self.num_of_action - 1) if self.num_of_action > 1 else 0.0
    #     scaled = action_min + fraction * (action_max - action_min)
    #     return scaled
    
    #     # ====================================== #
    def scale_action(self, action):
        """
        แปลง action ดิจิทัล (index) → เป็น tensor 2D shape = [num_envs, action_dim]
        รองรับ IsaacLab ที่ใช้ shape [N, D]
        """
        action_min, action_max = self.action_range
        fraction = action / (self.num_of_action - 1) if self.num_of_action > 1 else 0.0
        scaled = action_min + fraction * (action_max - action_min)

        # ✅ Return shape [1, 1] → [num_envs, action_dim]
        return torch.tensor([[scaled]], dtype=torch.float32).to("cuda")  # หรือ .to(self.device)


    
    def decay_epsilon(self, time_step):
        """
        Decay epsilon value to reduce exploration over time.
        """
        # ========= put your code here ========= #
        epsilon_decrease = (1.0 - self.final_epsilon) / time_step # Calculate how much to decrease each step
        self.epsilon = max(self.final_epsilon, self.epsilon - epsilon_decrease)
        # ====================================== #

    def save_w(self, path, filename):
        """
        Save weight parameters.
        """
        # ========= put your code here ========= #
        full_path = os.path.join(path, filename)
        np.save(full_path, self.w)
        print(f"Weights saved to {full_path}.")
        # ====================================== #
            
    def load_w(self, path, filename):
        """
        Load weights for agents: support both .npy (Linear Q) and .pt (DQN, AC, etc.)
        """
        import os
        import numpy as np
        import torch

        full_path = os.path.join(path, filename)
        if not os.path.exists(full_path):
            print(f"❌ File {full_path} does not exist.")
            return

        if filename.endswith(".npy"):
            self.w = np.load(full_path)
            print(f"[LinearQ] Weights loaded from {full_path}")

        elif filename.endswith(".pt"):
            state_dict = torch.load(full_path, map_location=self.device)

            # Case: DQN or MC_REINFORCE (has policy_net)
            if hasattr(self, "policy_net") and isinstance(state_dict, dict):
                self.policy_net.load_state_dict(state_dict)
                print(f"[DQN / MC_REINFORCE] policy_net loaded from {full_path}")

            # Case: Actor-Critic (has actor and critic)
            elif hasattr(self, "actor"):
                self.actor.load_state_dict(state_dict)
                print(f"[Actor-Critic] actor loaded from {full_path}")

            else:
                print("Error: Agent structure doesn't match expected keys.")

        else:
            print("Unsupported file format. Only .npy and .pt are supported.")

