# External
import gym
import torch

# Internal
from Agents.Common import Runner
from Agents.DQN import DQNAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner
from Evolve import *

def score_genome(genome, episodes, pbar):

    # make gym
    env = gym.make("CartPole-v1")

    # Create an agent that solves the environment:
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
    agent = DQNAgent(config)

    # Create a runner that runs the agent in the environment
    runner = Runner(agent, verbose=1, pbar=pbar)

    # Run the agent
    score, checkpoint = runner.run_agent()

    # Close
    env.close()

    # Return
    return score.best_score, checkpoint

def save_checkpoint(checkpoint, filepath):
    torch.save(checkpoint, filepath)

evolverConfig = {
    'score_genome': score_genome,
    'save_checkpoint': save_checkpoint,
    'save_filepath':'Checkpoints/cartpole_v1_dqn_evo.ch',
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