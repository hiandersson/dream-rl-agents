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
from Evolve import Evolver

class RunExperiment():
    """Interacts with and learns from the environment."""

    def __init__(self, gym_name):
        self.gym_name = gym_name

    def random_agent(self):

        env = gym.make(self.gym_name) # Make the gym
        agent = self.get_agent(AgentConfig(env=env)) # Create the agent
        runner = Runner(agent) # Create a runner that runs the agent in the environment
        runner.random() # Run the agent
        
        env.close()

    def enjoy_agent(self):

        env = gym.make(self.gym_name) # Make the gym
        agent = self.get_agent(AgentConfig(env=env)) # Create the agent
        runner = Runner(agent) # Create a runner that runs the agent in the environment    
        runner.enjoy_checkpoint('Checkpoints/{}_evo.ch'.format(self.gym_name.lower())) # Run the agent

        env.close()

    def train_agent(self):

        env = gym.make(self.gym_name) # Make the gym

        agent = self.get_agent(self.get_agent_train_config(env)) # Create an agent that solves the environment:

        runner = Runner(agent, save_best_score='Checkpoints/{}_train.ch'.format(self.gym_name.lower())) # Create a runner that runs the agent in the environment
        score, checkpoint = self.run(runner) # Run the agent
        score.visualize() # Visualize the score

        env.close()

    def score_genome(self, genome, episodes, pbar):

        with main_lock:
            env = gym.make(self.gym_name) # make gym
            agent = self.get_agent(self.get_agent_evolve_config(env, genome, episodes)) 
            runner = Runner(agent, verbose=1, pbar=pbar) # Create a runner that runs the agent in the environment
        
        score, checkpoint = self.run(runner)  # Run the agent

        with main_lock:
            env.close() # Close

        return score.best_score, checkpoint

    def save_checkpoint(self, checkpoint, filepath):

        torch.save(checkpoint, filepath)

    def evolve_agent(self):

        evolverConfig = self.get_evolver_config()

        evolverConfig['score_genome'] = self.score_genome
        evolverConfig['save_checkpoint'] = self.save_checkpoint
        evolverConfig['save_filepath'] = 'Checkpoints/{}_evo.ch'.format(self.gym_name.lower())
        evolverConfig['GPUDevices'] = None

        evolver_test = Evolver(evolverConfig)
        evolver_test.start()

    def command(self, arg):
        if arg == "-random":
            self.random_agent()
        if arg == "-train":
            self.train_agent()
        if arg == "-evolve":
            self.evolve_agent()
        if arg == "-enjoy":
            self.enjoy_agent()