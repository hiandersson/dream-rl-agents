# External
import numpy as np
import random
import copy
from collections import namedtuple, deque, defaultdict
import os
import logging
import json
from datetime import datetime

# Pytorch
import torch
import torch.nn.functional as F
import torch.optim as optim

# Code
from Agents.DDPG.models import Actor, Critic
from Agents.Common.memory import ReplayBuffer
from Agents.Common.noise import OUNoise
from Agents.Common.nn_updates import soft_update
from Agents.Common.nn_updates import copy_weights

# map device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
logger = logging.getLogger('ddpg')
logger.debug('Device Info:{}'.format(device))

class DDPGAgent():
    
    def __init__(self, env, config):

        # ------------------- store configs and variables  ------------ #

        action_size = env.action_space.shape[0]
        state_size = env.observation_space.shape[0]
        self.config = config
        self.env = env
        self.learn_step = 0
        self.qvalue_prev = 0
        self.last_td_error = torch.tensor([[0.0]])
        self.seed = random.seed(config.seed)      

        # ------------------- create networks  ------------------------ #

        self.actor_local = Actor(state_size, action_size, self.config.seed, self.config.fc1_units, self.config.fc2_units, self.config.batch_norm).to(device)
        self.actor_target = Actor(state_size, action_size, self.config.seed, self.config.fc1_units, self.config.fc2_units, self.config.batch_norm).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), self.config.actor_learning_rate)

        self.critic_local = Critic(state_size, action_size, self.config.seed, self.config.fc1_units, self.config.fc2_units, self.config.batch_norm).to(device)
        self.critic_target = Critic(state_size, action_size, self.config.seed, self.config.fc1_units, self.config.fc2_units, self.config.batch_norm).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), self.config.critic_learning_rate, weight_decay=self.config.weight_decay)

        # ------------------- create action noise  -------------------- #

        self.noise = OUNoise(action_size, self.config.seed, sigma=self.config.ou_noise_sigma)

        # ------------------- create replay memory  ------------------- #

        self.memory = ReplayBuffer(action_size, config)
    
    def save(self, filepath):

        checkpoint = {
            'actor_local': self.actor_local.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_local': self.critic_local.state_dict(),
            'critic_target': self.critic_target.state_dict()}

        torch.save(checkpoint, filepath)

    def load(self, filepath):

        checkpoint = torch.load(filepath)

        self.actor_local.load_state_dict(checkpoint['actor_local'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_local.load_state_dict(checkpoint['critic_local'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])

    def reset(self):

        self.noise.reset()

    def step(self, state, action, reward, next_state, done):

        # ------------------- save experience in replay memory  ------- #

        state_reshaped = state.reshape(1, state.shape[0])   
        next_state_reshaped = next_state.reshape(1, next_state.shape[0])   

        self.memory.add(state_reshaped, action, reward, next_state_reshaped, done, self.last_td_error)

        # ------------------- learn every update_every step  ---------- #
        
        # Learn, if enough samples are available in memory
        if len(self.memory) > self.config.batch_size:
            experiences, per_weights = self.memory.sample()
            self.learn(experiences, per_weights)

    def post_episode(self):

        pass

    def act(self, state):

        # ------------------- select action from local network  ------- #

        state_reshaped = state.reshape(1, state.shape[0])   
        state_act = torch.from_numpy(state_reshaped).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_act).cpu().data.numpy()
        self.actor_local.train()

        # ------------------- select action from local network  ------- #

        if self.config.ou_noise_active == True:
            action += self.noise.sample()

        # ------------------- clip action outputs -1 to 1  ------------ #

        return np.clip(action, -1, 1).flatten()

    def act_no_training(self, state):

        # ------------------- select action from local network  ------- #

        state_reshaped = state.reshape(1, state.shape[0])   
        state_act = torch.from_numpy(state_reshaped).float().to(device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state_act).cpu().data.numpy()

        # ------------------- clip action outputs -1 to 1  ------------ #

        return np.clip(action, -1, 1).flatten()

    def learn(self, experiences, per_weights):

        # ------------------- unpack experiences  --------------------- #

        states, actions, rewards, next_states, dones = experiences
        self.learn_step += 1

        # ------------------- train critic ---------------------------- #
        
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)

        # Store last TD error for prioritized experience replay
        self.last_td_error = (Q_targets_next - self.qvalue_prev).detach().numpy()
        self.qvalue_prev = Q_targets_next

        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (self.config.gamma * Q_targets_next * (1 - dones))

        # Expected Q
        Q_expected = self.critic_local(states, actions.float())

        # Critic loss
        if self.config.per_active == True:
            # Prioritized experience replay loss - multiple gradient with weights 
            critic_loss = (per_weights * ((Q_expected - Q_targets) ** 2)).sum() / Q_expected.data.nelement()
        else:
            # Compute straight mse loss
            critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # -------------------- train actor -------------------------- #
        
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()

        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------- update target networks --------------- #
        
        # Copy weights on step 1
        if self.learn_step == 1:
            copy_weights(self.critic_local, self.critic_target)
            copy_weights(self.actor_local, self.actor_target)
        else:
            # Soft update weights on target frequency
            if self.learn_step % self.config.update_every == 0:
                soft_update(self.critic_local, self.critic_target, self.config.tau)
                soft_update(self.actor_local, self.actor_target, self.config.tau)    
