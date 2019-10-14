import torch
import torch.nn as nn
import torch.nn.functional as F

# Vanilla DQN network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc_units)
        self.fc2 = nn.Linear(fc_units, fc_units)
        self.fc3 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Vanilla DQN network convolutional
class QNetworkConvolutional(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units, device):
        super(QNetworkConvolutional, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.device = device

        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(1, 4, kernel_size=6, stride=2, bias=False)

        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=1936

        """
        self.conv1 = nn.Conv2d(1, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.size = 32 * 5 * 5
        """
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)

    def forward(self, state):
        state = state.to(self.device)
        x = F.relu(self.conv1(state))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Dueling DQN networs
class QDuelingNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, fc_units):
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
        x = self.feature(state)
        advantage = self.advantage(x)
        value     = self.value(x)
        return value + advantage  - advantage.mean()
