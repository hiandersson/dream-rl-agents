import torch
import torch.nn as nn
import torch.nn.functional as F

# Vanilla DQN network
class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.
        Params
        ======
        """
        
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Dueling DQN network https://arxiv.org/pdf/1511.06581.pdf
class QDuelingNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc_units):
        """Initialize parameters and build model.

        Params
        ======
        """

        super(QDuelingNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.feature = nn.Sequential(
            nn.Linear(state_size, fc_units),
            nn.ReLU()
        )
        
        self.advantage = nn.Sequential(
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, action_size)
        )
        
        self.value = nn.Sequential(
            nn.Linear(fc_units, fc_units),
            nn.ReLU(),
            nn.Linear(fc_units, 1)
        )

    def forward(self, state):
        """Build a network that maps state -> action values based on the dueling network architecture."""
        x = self.feature(state)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
