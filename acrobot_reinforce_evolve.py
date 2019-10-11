# External
import gym
import torch

# Internal
from Agents.Common import Runner
from Agents.REINFORCE import REINFORCEAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner
from Evolve import *

def score_genome(genome, episodes, pbar):

    # Make the gym
    env = gym.make("Acrobot-v1")

    # Create an agent that solves the environment:
    # "Acrobot-v1 is an unsolved environment, which means it does not have a specified reward threshold at which it's considered solved.""
    config = AgentConfig(
        env=env,
        n_episodes=episodes, 
        target_average=-10,
        max_t=200,
        gamma=genome['gamma'],
        learning_rate=genome['learningRate'],
        fc1_units=genome['fc1_units'])
    agent = REINFORCEAgent(config)

    # Create a runner that runs the agent in the environment
    runner = Runner(agent, verbose=1, pbar=pbar)

    # Run the agent
    score, checkpoint = runner.run_single_probability_trajectory()

    # Return
    return score.best_score, checkpoint

def save_checkpoint(checkpoint, filepath):
    torch.save(checkpoint, filepath)

evolverConfig = {
    'score_genome': score_genome,
    'save_checkpoint': save_checkpoint,
    'save_filepath':'Checkpoints/acrobot_v1_reinforce_evo.ch',
    'GPUDevices': None,
    'episodes': 200,
    'populationSize': 1,
    'retainSize': 5,
    'mutateOneGeneRandom': 5,
    'mutateTwoGenesRandom': 5,
    'crossoverOneGene': 5,
    'crossOverTwoGenes': 5,
    'mutateOneGeneClose': 5,
    'generations': 2,
    'randomSeed': 1,
    'allPossibleGenesSimple': {
        'learningRate': [0.01, 0.001, 0.002, 0.003, 0.004, 0.0005],
        'gamma':        [0.90, 0.92, 0.94, 0.96, 0.98, 0.9, 0.995, 1.0],
        'fc1_units':    [16,32,64,128,256,512],
    },
    'allPossibleGenesSimpleShort': {
        'learningRate': "Rate",
        'batchSize': "Batch",
    }
}

evolver_test = evolver(evolverConfig)
evolver_test.start()