# External
import gym
import sys

# Internal
from Agents.Common import Runner
from Agents.Common import AgentConfig
from Agents.Common import Runner

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
        runner.enjoy_checkpoint('Checkpoints/{}_evo.ch'.format(self.gym_name)) # Run the agent

        env.close()

    def train_agent(self):

        env = gym.make(self.gym_name) # Make the gym

        agent = self.get_agent(self.get_agent_train_config(env)) # Create an agent that solves the environment:

        runner = Runner(agent, save_best_score='Checkpoints/{}_train.ch'.format(self.gym_name)) # Create a runner that runs the agent in the environment
        score, checkpoint = runner.run_agent() # Run the agent
        score.visualize() # Visualize the score

        env.close()

    def score_genome(self, genome, episodes, pbar):

        env = gym.make(self.gym_name) # make gym

        config = AgentConfig(
            env=env,
            n_episodes=episodes, 
            target_average=475,
            update_every=genome['updateEvery'],
            batch_size=genome['batchSize'],
            gamma=genome['gamma'],
            learning_rate=genome['learningRate'],
            fc1_units=genome['fc1_units'],
            fc2_units=genome['fc2_units'],
            tau=genome['tau'],
            memory_size=genome['memory_size'],
            deepq_double_learning=True,
            deepq_dueling_networks=True,
            per_active=True)
        agent = DQNAgent(config) # Create an agent that solves the environment:
        
        runner = Runner(agent, verbose=1, pbar=pbar) # Create a runner that runs the agent in the environment
        score, checkpoint = runner.run_agent()  # Run the agent
        env.close() # Close

        return score.best_score, checkpoint

    def save_checkpoint(self, checkpoint, filepath):
        torch.save(checkpoint, filepath)

    def evolve_agent(self):

        evolverConfig = {
            'score_genome': score_genome,
            'save_checkpoint': save_checkpoint,
            'save_filepath':'Checkpoints/{}_evo.ch'.format(self.gym_name),
            'GPUDevices': None,
            'episodes': 1000,
            'populationSize': 25,
            'retainSize': 5,
            'mutateOneGeneRandom': 5,
            'mutateTwoGenesRandom': 5,
            'crossoverOneGene': 5,
            'crossOverTwoGenes': 5,
            'mutateOneGeneClose': 5,
            'generations': 2,
            'randomSeed': 1,
            'allPossibleGenesSimple': {
                'learningRate': [0.01, 0.001, 0.002, 0.003, 0.004, 0.0001, 0.0005],
                'batchSize':    [32,64,128,256,512],
                'gamma':        [0.90, 0.92, 0.94, 0.96, 0.98, 0.9, 0.995],
                'updateEvery':  [1,2,3,4],
                'fc1_units':    [64,128,256,512],
                'fc2_units':    [64,128,256,512],
                'memory_size':  [int(1e4), int(1e5), int(1e6)],
                'tau':          [1e-2, 1e-3, 1e-4],
            },
        }

        evolver_test = evolver(evolverConfig)
        evolver_test.start()

    def command(self, arg):
        if arg == "-random":
            self.random_agent()
        if arg == "-train":
            self.train_agent()
        if arg == "-evolve":
            self.evolve_agent()