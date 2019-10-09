import numpy as np
import random
from collections import namedtuple, deque
import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, agent_config):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.agent_config = agent_config
        self.action_size = action_size
        self.memory = deque(maxlen=agent_config.memory_size)  # internal memory (deque)
        self.probability_picked = deque(maxlen=agent_config.memory_size)
        self.batch_size = agent_config.batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(agent_config.seed)

    def add(self, state, action, reward, next_state, done, probability_picked=0):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

        # Prioritized experience replay - store the (last td error + epsilom) ^ alpha to later be used when sampling from memory
        if self.agent_config.per_active == True:
            p = (np.power(np.abs(probability_picked)[0][0] + self.agent_config.per_epsilon, self.agent_config.per_alpha))
            self.probability_picked.append(p)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""

        # Prioritized experience replay
        if self.agent_config.per_active == True:

            # probabilities
            sum_probabilities = np.sum(self.probability_picked)
            sampling_probabilities = [p / sum_probabilities for p in self.probability_picked]
            
            # sampling
            sampled_indexes = np.random.choice(np.arange(len(self.memory)), size=self.batch_size, replace=False, p=sampling_probabilities)

            # experiences 
            experiences = [self.memory[ii] for ii in sampled_indexes]

            # weights ( 1 / N * 1 / P(i)) ^ beta
            one_over_memory_length = 1 / len(self.memory)
            weights = [np.power(one_over_memory_length * ( 1 / sampling_probabilities[ii]), self.agent_config.per_beta) for ii in sampled_indexes]
            replay_weights = torch.from_numpy(np.vstack([w for w in weights])).float().to(device)
        
        # No replay
        else:
            experiences = random.sample(self.memory, k=self.batch_size)
            replay_weights = None

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones), replay_weights

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)   