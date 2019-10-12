from run_experiment_helper import *
from Agents.DDPG import DDPGAgent

class CardpoleDQN(RunExperiment):
    def __init__(self):
        RunExperiment.__init__(self, "LunarLanderContinuous-v2")

    def get_agent(self, config):
        return DDPGAgent(config)

    def get_agent_train_config(self, env):
        return AgentConfig(
            env=env,
            seed=42,
            target_average=200, 
            n_episodes=5, 
            batch_norm=False, 
            batch_size=64, 
            gamma=0.98, 
            actor_learning_rate=1e-4,
            critic_learning_rate=1e-3,
            weight_decay=0,
            update_every=1,
            ou_noise_sigma=0.01,
            fc1_units=400,
            fc2_units=300,
            per_active=False)

    def get_agent_evolve_config(self, env, genome, episodes):
        return AgentConfig(
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

    def get_evolver_config(self):
        return {
            'episodes': 5,
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

    def run(self, runner):
        return runner.run_agent()

if __name__ == "__main__":
    run = CardpoleDQN()
    run.command(sys.argv[1])
