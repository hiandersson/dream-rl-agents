# External
import gym
import torch

# Internal
from Agents.Common import Runner
from Agents.DDPG import DDPGAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner
from Evolve import *

def score_genome(genome, episodes, pbar):

    # make gym
    env = gym.make("LunarLanderContinuous-v2")

    # Create an agent that solves the environment:
    config = AgentConfig(
        env=env,
        seed=42,
        target_average=200, 
        n_episodes=episodes, 
        batch_size=genome['batchSize'],
        gamma=genome['gamma'], 
        actor_learning_rate=genome['actor_learningRate'],
        critic_learning_rate=genome['critic_learningRate'],
        weight_decay=genome['weight_decay'],
        update_every=genome['updateEvery'],
        ou_noise_sigma=0.01,
        fc1_units=genome['fc1_units'],
        fc2_units=genome['fc2_units'],
        memory_size=genome['memory_size'],
        per_active=False,
        batch_norm=False)

    agent = DDPGAgent(config)

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
    'save_filepath':'Checkpoints/lunarlander_v2_ddpg_evo.ch',
    'GPUDevices': None,
    'episodes': 500,
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
        'actor_learningRate': [0.01, 0.001, 0.002, 0.003, 0.004, 0.0001, 0.0005],
        'critic_learningRate': [0.01, 0.001, 0.002, 0.003, 0.004, 0.0001, 0.0005],
        'batchSize':    [32,64,128,256,512],
        'gamma':        [0.90, 0.92, 0.94, 0.96, 0.98, 0.9, 0.995, 1.0],
        'updateEvery':  [1,2,3,4],
        'weight_decay':  [0, 1e-1, 1e-2, 1e-3],
        'fc1_units':    [64,128,256,512],
        'fc2_units':    [64,128,256,512],
        'memory_size':  [int(1e4), int(1e5), int(1e6)],
        'tau':          [1e-2, 1e-3, 1e-4],
    },
}

evolver_test = evolver(evolverConfig)
evolver_test.start()