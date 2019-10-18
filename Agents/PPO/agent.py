# External
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import logging

# Internal
from Agents.PPO.models import *

class PPOAgent():
    
    def __init__(self, config):

        # ------------------- store configs and variables  -------------- #

        self.envs = config.parallelEnv
        self.config = config
        self.init()

    def init(self):
    
        self.epsilon = self.config.ppo_epsilon
        self.beta = self.config.ppo_beta

        # ------------------- create networks  -------------------------- #

        #self.policy = Policy(self.config.fc1_units).to(self.config.device) 
        self.policy = PolicyTwo(self.config.fc1_units,action_size=2,device=self.config.device).to(self.config.device) 
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

        checkpoint = torch.load(filepath, map_location=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

        self.config = self.config.from_dict_to_config(checkpoint['agent_config'])

        self.init()

        self.policy.load_state_dict(checkpoint['policy'])

    def act_no_training(self, state):

        # ------------------- select action from local network  ------- #

        new_probabilities = self.states_to_probabilities(policy, state)

        return action

    def states_to_probabilities(self, policy, states):

        # ------------------- convert states to policies  --------------- #

        states = torch.stack(states)
        policy_input = states.view(-1,*states.shape[-3:])

        return policy(policy_input).view(states.shape[:-3])

    def clipped_surrogate(self, policy, old_probabilities, states, actions, rewards, discount=0.995, epsilon=0.1, beta=0.01):

        discount = discount**np.arange(len(rewards))
        rewards = np.asarray(rewards)*discount[:,np.newaxis]
        
        # ------------------- convert rewards to future rewards  -------- #

        rewards_future = rewards[::-1].cumsum(axis=0)[::-1]

        # ------------------- normalize rewards  ------------------------ #
        
        mean = np.mean(rewards_future, axis=1)
        standard_devitation = np.std(rewards_future, axis=1) + 1.0e-10
        rewards_normalized = (rewards_future - mean[:,np.newaxis])/standard_devitation[:,np.newaxis]
        
        # ------------------- convert to pytorch tensors  --------------- #

        actions = torch.tensor(actions, dtype=torch.int8, device=self.config.device)
        old_probabilities = torch.tensor(old_probabilities, dtype=torch.float, device=self.config.device)
        rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=self.config.device)

        # ------------------- convert states to probabilities  ---------- #

        #states = torch.tensor(states, dtype=torch.float, device=self.config.device)
        #print(states)

        # print("\n -------- \n")

        states = torch.stack(states)

        # print("states {}".format(states.shape))

        policy_input = states.view(-1,*states.shape[-3:])

        # print("policy_input {}".format(policy_input.shape))

        new_probabilities = torch.tensor(np.zeros((len(policy_input), 1)))

        for i in range(len(policy_input)):
            action, new_prob = policy.act(policy_input[0].unsqueeze(0))
            new_probabilities[i] = new_prob

        new_probabilities = new_probabilities.view(states.shape[:-3])

        """
        new_probabilities = self.states_to_probabilities(policy, states)
        new_probabilities = torch.where(actions == self.config.RIGHT, new_probabilities, 1.0-new_probabilities)
        """

        """
        print("\n -------- \n")

        print("new_probabilities[0] {}".format(new_probabilities[0]))
        print("new_probabilities {} type {}".format(new_probabilities.shape, type(new_probabilities)))

        print("\n -------- \n")

        print("old_probabilities[0] {}".format(old_probabilities[0]))
        print("old_probabilities {} type {}".format(old_probabilities.shape, type(old_probabilities)))
        """

        # ------------------- ratio for clipping  ----------------------- #

        ratio = new_probabilities/old_probabilities

        """
        print("\n -------- \n")

        print("ratio[0] {}".format(ratio[0]))
        print("ratio {} type {}".format(ratio.shape, type(ratio)))

        print("\n -------- \n")
        """

        # ------------------- clip the surrogate function  -------------- #

        clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
        clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

        """
        print("clipped_surrogate[0] {}".format(clipped_surrogate[0]))
        print("clipped_surrogate {} type {}".format(clipped_surrogate.shape, type(clipped_surrogate)))

        print("\n -------- \n")
        """


        # ------------------- include a regularization term  ------------ #

        """
        # this steers new_policy towards 0.5 and add in 1.e-10 to avoid log(0) which gives nan
        print("new_probabilities = {}".format(new_probabilities))
        print("old_probabilities = {}".format(old_probabilities))
        print("torch.log(old_probabilities+1.e-10) = {}".format(torch.log(old_probabilities)))
        print("torch.log(1.0-old_probabilities+1.e-9) = {}".format(torch.log(1.0-old_probabilities+1.e-9)))
        """

        entropy = -(new_probabilities*torch.log(old_probabilities+1.e-9)+ (1.0-new_probabilities)*torch.log(1.0-old_probabilities+1.e-9))

        """
        print("entropy[0] {}".format(entropy[0]))
        print("entropy {} type {}".format(entropy.shape, type(entropy)))

        print("\n -------- \n")
        """

        # ------------------- return an average of all entries ----------- #

        # effective computing L_sur^clip / T, averaged over time-step and number of trajectories
        # this is desirable because we have normalized our rewards.

        """
        print("clipped_surrogate = {}".format(clipped_surrogate))
        print("entropy = {}".format(entropy))
        print("beta*entropy = {}".format(beta*entropy))
        """

        L = torch.mean(clipped_surrogate + beta*entropy)

        """
        print("L[0] {}".format(L.item()))
        print("L {} type {}".format(L.shape, type(L)))

        print("\n -------- \n")

        ac(0)
        """



        """ WORKING
            -------- 

            new_probabilities[0] tensor([0.4975], grad_fn=<SelectBackward>)
            new_probabilities torch.Size([500, 1]) type <class 'torch.Tensor'>

            -------- 

            old_probabilities[0] tensor([0.4975])
            old_probabilities torch.Size([500, 1]) type <class 'torch.Tensor'>

            -------- 

            ratio[0] tensor([1.], grad_fn=<SelectBackward>)
            ratio torch.Size([500, 1]) type <class 'torch.Tensor'>

            -------- 

            clipped_surrogate[0] tensor([0.], grad_fn=<SelectBackward>)
            clipped_surrogate torch.Size([500, 1]) type <class 'torch.Tensor'>

            -------- 

            entropy[0] tensor([0.6931], grad_fn=<SelectBackward>)
            entropy torch.Size([500, 1]) type <class 'torch.Tensor'>

            -------- 

            L[0] 0.006931341253221035
            L torch.Size([]) type <class 'torch.Tensor'>

            -------- 
        """

        return L

    def learn(self, old_probabilities, states, actions, rewards):

        # ------------------- gradient ascent step  ---------------------- #

        for _ in range(self.config.ppo_gradientdescent_steps_per_epoch):
            
            # gradient ascent of the clipped surrogate function
            L = -self.clipped_surrogate(self.policy, old_probabilities, states, actions, rewards, epsilon=self.epsilon, beta=self.config.ppo_beta)

            self.optimizer.zero_grad()
            L.backward()
            self.optimizer.step()
            del L

        # ------------------- adjust epsilon and beta over time  --------- #
        
        # the clipping parameter reduces as time goes on
        self.epsilon*=.999
        
        # the regulation term also reduces
        # this reduces exploration in later runs
        self.beta*=.995