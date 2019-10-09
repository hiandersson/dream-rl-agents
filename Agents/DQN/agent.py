import numpy as np
import random
from collections import namedtuple, deque

from Agents.DQN.model import QNetwork
from Agents.DQN.model import QDuelingNetwork
from Agents.Common.memory import ReplayBuffer
from Agents.Common.nn_updates import soft_update

import torch
import torch.nn.functional as F
import torch.optim as optim

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("GPU/CPU device: {}".format(device))

class DQNAgent():
    """Interacts with and learns from the environment."""

    def __init__(self, env, config):
        """Initialize an Agent object.
        
        Params
        ======
        """

        # ------------------- store configs and variables  ------------ #

        self.env = env
        self.config = config
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]
        self.qvalue_prev = 0
        self.last_td_error = torch.tensor([[0.0]])
        self.t_step = 0
        self.epsilon = config.epsilon_start
        
        # ------------------- create networks  ------------------------ #

        if self.config.deepq_dueling_networks == True:
            self.qnetwork_local = QDuelingNetwork(self.state_size , self.action_size, self.config.seed, self.config.fc1_units).to(device)
            self.qnetwork_target = QDuelingNetwork(self.state_size , self.action_size, self.config.seed, self.config.fc1_units).to(device)
        else:
            self.qnetwork_local = QNetwork(self.state_size , self.action_size, self.config.seed, self.config.fc1_units).to(device)
            self.qnetwork_target = QNetwork(self.state_size , self.action_size, self.config.seed, self.config.fc1_units).to(device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.config.learning_rate)

        # ------------------- create replay memory  ------------------- #

        self.memory = ReplayBuffer(self.action_size, config)

    def post_episode(self):

        self.epsilon = max(self.config.epsilon_end, self.config.epsilon_decay * self.epsilon) 
    
    def reset(self):

        pass

    def save(self, filepath):

        checkpoint = {
            'local_state_dict': self.qnetwork_local.state_dict(),
            'target_state_dict': self.qnetwork_target.state_dict()}

        torch.save(checkpoint, filepath)

    def load(self, filepath):

        checkpoint = torch.load(filepath)

        self.qnetwork_local.load_state_dict(checkpoint['local_state_dict'])
        self.qnetwork_target.load_state_dict(checkpoint['target_state_dict'])

    def step(self, state, action, reward, next_state, done):

        # ------------------- save experience in replay memory  ------- #

        self.memory.add(state, action, reward, next_state, done, self.last_td_error)

        # ------------------- learn every update_every step  ---------- #
        
        self.t_step = (self.t_step + 1) % self.config.update_every
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.config.batch_size:
                experiences, per_weights = self.memory.sample()
                self.learn(experiences, per_weights)

    def act(self, state, epsilon=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        """

        # ------------------- select action from local network  ------- #

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # ------------------- epsilon-greedy action selection  -------- #

        if random.random() > epsilon:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def act_no_training(self, state):

        # ------------------- select action from local network  ------- #

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        # ------------------- action selection  ---------------------- #

        return np.argmax(action_values.cpu().data.numpy())

    def learn(self, experiences, per_weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
        """

        # ------------------- unpack experiences  --------------------- #

        states, actions, rewards, next_states, dones = experiences

        # ------------------- calculate q expected and target  -------- #

        # Get max predicted Q values (for next states) from target model
        if self.config.deepq_double_learning == True:
            # In double Q learning, select max q value from local network but use target network for evaluation
            Q_targets_next = torch.gather(self.qnetwork_target(next_states), index=torch.argmax(self.qnetwork_local(next_states), dim=1, keepdim=True), dim=1)        
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Store last TD error for prioritized experience replay
        self.last_td_error = (Q_targets_next - self.qvalue_prev).detach().numpy()
        self.qvalue_prev = Q_targets_next

        # Update target
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions.long())

        # ------------------- calculate and minimize the loss --------- #

        if self.config.per_active == True:

            # Prioritized experience replay loss - multiple gradient with weights 
            loss = (per_weights * ((Q_expected - Q_targets) ** 2)).sum() / Q_expected.data.nelement()
        else:

            # Compute straight mse loss
            loss = F.mse_loss(Q_expected, Q_targets)

        # back propagate the error
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #

        soft_update(self.qnetwork_local, self.qnetwork_target, self.config.tau)                     
