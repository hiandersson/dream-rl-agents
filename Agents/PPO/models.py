import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class Policy(nn.Module):

    def __init__(self, fc_units):
        super(Policy, self).__init__()
        
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)

        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, fc_units)
        self.fc2 = nn.Linear(fc_units, 1)

        # Sigmoid to 
        self.sig = nn.Sigmoid()
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        return self.sig(self.fc2(x))

    
# set up a convolutional neural net
# the output is the probability of moving right
# P(left) = 1-P(right)
class PolicyTwo(nn.Module):

    def __init__(self, fc_units, action_size=2, device=None):
        super(PolicyTwo, self).__init__()
        
        # 80x80x2 to 38x38x4
        # 2 channel from the stacked frame
        self.conv1 = nn.Conv2d(2, 4, kernel_size=6, stride=2, bias=False)

        # 38x38x4 to 9x9x32
        self.conv2 = nn.Conv2d(4, 16, kernel_size=6, stride=4)
        self.size=9*9*16
        
        # two fully connected layer
        self.fc1 = nn.Linear(self.size, fc_units)
        self.fc2 = nn.Linear(fc_units, action_size)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(-1,self.size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

    def act(self, state):
        #state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        """
        print("state x = {}".format(state.shape))
        print("state isnan = {}".format(torch.isnan(state).any()))
        print("state isinf = {}".format(torch.isinf(state).any()))
        """
        state = torch.from_numpy(state.numpy())
        # print("state = {}".format(torch.isnan(state).any()))
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        # print("m = {}".format(m.probs))
        action = m.sample()
        # print("m.log_prob(action) = {}".format(m.log_prob(action)))
        
        action = m.sample()
        #print("action = {}".format(action))
        return action.numpy(), -m.log_prob(action)
