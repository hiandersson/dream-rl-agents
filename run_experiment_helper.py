# External
import gym
import sys
import torch
from threading import Thread, Lock
main_lock = Lock()

# Internal
from Agents.Common import Runner
from Agents.Common import AgentConfig
from Agents.Common import Runner
from Agents.Utils.parallelEnv import parallelEnv
from Evolve import Evolver

class RunExperiment():
    """Interacts with and learns from the environment."""

    def __init__(self, gym_name, parallelEnvs=False):
        self.gym_name = gym_name
        self.parallelEnvs = parallelEnvs

    def random_agent(self):

        env = gym.make(self.gym_name) # Make the gym
        agent = self.get_agent(AgentConfig(env=env)) # Create the agent
        runner = Runner(agent) # Create a runner that runs the agent in the environment
        runner.random() # Run the agent
        
        env.close()

    def enjoy_agent(self, agent_type):

        env = gym.make(self.gym_name) # Make the gym
        agent = self.get_agent(AgentConfig(env=env)) # Create the agent
        runner = Runner(agent) # Create a runner that runs the agent in the environment    

        if agent_type == "evolve":
            self.enjoy_checkpoint(runner, 'Checkpoints/{}_evo.ch'.format(self.gym_name.lower()))
        elif agent_type == "train":
            self.enjoy_checkpoint(runner, 'Checkpoints/{}_train.ch'.format(self.gym_name.lower()))

        env.close()

    def train_agent(self):

        with main_lock:
            if self.parallelEnvs == True:
                envs = parallelEnv(self.gym_name, n=8, seed=1234)
                agent = self.get_agent(self.get_agent_train_config(envs)) # Create an agent that solves the environment:
            else:
                env = gym.make(self.gym_name) # Make the gym
                agent = self.get_agent(self.get_agent_train_config(env)) # Create an agent that solves the environment:

        runner = Runner(agent, save_best_score='Checkpoints/{}_train.ch'.format(self.gym_name.lower())) # Create a runner that runs the agent in the environment
        score, checkpoint = self.run(runner) # Run the agent
        score.visualize(save_to='Checkpoints/{}_train_plot.png'.format(self.gym_name.lower())) # Visualize the score

        if self.parallelEnvs == True:
            envs.close()
        else:
            env.close()

    def score_genome(self, genome, episodes, device, pbar):

        with main_lock:
            if self.parallelEnvs == True:
                envs = parallelEnv(self.gym_name, n=8, seed=1234)
                agent = self.get_agent(self.get_agent_evolve_config(envs, genome, device, episodes)) # Create an agent that solves the environment:
            else:
                env = gym.make(self.gym_name) # Make the gym
                agent = self.get_agent(self.get_agent_evolve_config(env, genome, device, episodes)) # Create an agent that solves the environment:

        runner = Runner(agent, verbose=1, pbar=pbar) # Create a runner that runs the agent in the environment
        score, checkpoint = self.run(runner)  # Run the agent

        if self.parallelEnvs == True:
            envs.close()
        else:
            env.close()

        return score, checkpoint

    def save_checkpoint(self, checkpoint, filepath):

        torch.save(checkpoint, filepath)

    def evolve_agent(self):

        evolverConfig = self.get_evolver_config()

        evolverConfig['score_genome'] = self.score_genome
        evolverConfig['save_checkpoint'] = self.save_checkpoint
        evolverConfig['save_filepath'] = 'Checkpoints/{}_evo.ch'.format(self.gym_name.lower())
        evolverConfig['save_plot'] = 'Checkpoints/{}_evo_plot.png'.format(self.gym_name.lower())

        evolver_test = Evolver(evolverConfig)
        evolver_test.start()

    def command(self, arg):
        if arg[1] == "-random":
            self.random_agent()
        if arg[1] == "-train":
            self.train_agent()
        if arg[1] == "-evolve":
            self.evolve_agent()
        if arg[1] == "-enjoy":
            self.enjoy_agent(arg[2])