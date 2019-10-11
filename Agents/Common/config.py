import random
import torch
import numpy as np
from dict2obj import Dict2Obj

class AgentConfig():
    def __init__(self, 
                env=None,
                device=None,
                parallelEnv=None,
                seed=0,
                target_average=10, 
                n_episodes=500, 
                max_t=1000, 
                epsilon_start=1.0, 
                epsilon_end=0.01, 
                epsilon_decay=0.995,
                memory_size=int(1e4),
                batch_size=64,
                gamma=0.99,
                tau=1e-3,
                learning_rate=5e-4,
                update_every=4,
                per_active=False,
                per_epsilon=0.0001,
                per_alpha=0.6,
                per_beta=0.4,
                deepq_double_learning=True,
                deepq_dueling_networks=True,
                fc1_units=128,
                fc2_units=128,
                batch_norm=True,
                actor_learning_rate=1e-3,
                critic_learning_rate=1e-3,
                weight_decay=0,
                ou_noise_sigma=0.01,
                ou_noise_active=True,
                ppo_discount_rate = .99,
                ppo_epsilon = 0.1,
                ppo_beta = .01,
                ppo_tmax = 320,
                ppo_gradientdescent_steps_per_epoch = 4
                ):

        # environments
        self.env = env
        self.parallelEnv = parallelEnv

        # device
        self.device = device
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # seed for reproducibility
        self.seed = seed
        if env != None:
            env.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # target
        self.target_average=target_average
        self.n_episodes=n_episodes
        self.max_t=max_t

        # epislon
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # agent
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        self.learning_rate = learning_rate
        self.update_every = update_every
        
        # network
        self.fc1_units = fc1_units
        self.fc2_units = fc2_units
        self.batch_norm = batch_norm
        self.weight_decay = weight_decay

        # prioritized experience replay
        self.per_active = per_active
        self.per_epsilon = per_epsilon
        self.per_alpha = per_alpha
        self.per_beta = per_beta

        # deepq
        self.deepq_double_learning = deepq_double_learning
        self.deepq_dueling_networks = deepq_dueling_networks

        # actor critic
        self.actor_learning_rate = actor_learning_rate
        self.critic_learning_rate = critic_learning_rate

        # ou noise 
        self.ou_noise_active = ou_noise_active
        self.ou_noise_sigma = ou_noise_sigma

        # ppo
        self.ppo_discount_rate = ppo_discount_rate
        self.ppo_epsilon = ppo_epsilon
        self.ppo_beta = ppo_beta
        self.ppo_tmax = ppo_tmax
        self.ppo_gradientdescent_steps_per_epoch = ppo_gradientdescent_steps_per_epoch

    def get_dict(self):

        agent_config_dict = vars(self).copy()

        del agent_config_dict['parallelEnv']
        del agent_config_dict['env']
        del agent_config_dict['device']

        return agent_config_dict

    def from_dict_to_config(self, dict_object):

        agent_config_new = Dict2Obj(dict_object)
        agent_config_new.env = self.env
        agent_config_new.parallelEnv = self.parallelEnv
        agent_config_new.device = self.device

        return agent_config_new