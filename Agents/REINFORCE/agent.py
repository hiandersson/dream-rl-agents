# External
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

# Internal
from Agents.REINFORCE.models import *

class REINFORCEAgent():
    
    def __init__(self, config):

        # ------------------- store configs and variables  -------------- #

        self.env = config.env
        self.config = config
        self.init()
    
    def init(self):

        # ------------------- create networks  -------------------------- #

        action_size = self.env.action_space.n
        state_size = self.env.observation_space.shape[0]

        self.policy = Policy(state_size=state_size, action_size=action_size, fc1_units=self.config.fc1_units).to(self.config.device) 
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.config.learning_rate)

    def get_checkpoint(self):

        checkpoint = {
            'policy': self.policy.state_dict(),
            'agent_config': self.config.get_dict(),
        }

        return checkpoint

    def save(self, filepath):

        torch.save(self.get_checkpoint(), filepath)

    def load(self, filepath):

        checkpoint = torch.load(filepath)

        self.config = self.config.from_dict_to_config(checkpoint['agent_config'])

        self.init()

        self.policy.load_state_dict(checkpoint['policy'])

    def act(self, state):
        """Returns actions for given state as per current policy.
        
        Params
        ======
        """

        # ------------------- select action from local network  ------- #

        action, log_probability = self.policy.act(state, self.config.device)

        return action, log_probability

    def act_no_training(self, state):

        # ------------------- select action from local network  ------- #

        action, log_probability = self.policy.act(state, self.config.device)

        return action

    def learn(self, saved_log_probabilities, saved_rewards):

        # ------------------- calculate discountds and rewards  --------- #

        discounts = [self.config.gamma**i for i in range(len(saved_rewards)+1)]
        cumulative_rewards = sum([discount*reward for discount,reward in zip(discounts, saved_rewards)])

        # ------------------- calculate the policy loss ----------------- #
            
        policy_loss = []

        for log_probability in saved_log_probabilities:
            policy_loss.append(-log_probability * cumulative_rewards) # - for gradient ascent

        policy_loss = torch.cat(policy_loss).sum()

        # ------------------- perform gradient ascent ------------------- #
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
