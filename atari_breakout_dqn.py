from run_experiment_helper import *
from Agents.DQN import DQNAgent

class AtariBreakoutDQN(RunExperiment):
    def __init__(self):
        RunExperiment.__init__(self, "Breakout-v0")

    def get_agent(self, config):
        return DQNAgent(config)

    def get_agent_train_config(self, env):
        return AgentConfig(
            env=env,
            n_episodes=1000, 
            target_average=475,
            convolutional_input=True,
            update_every=1,
            batch_size=2,
            gamma=0.95,
            learning_rate=1e-4,
            deepq_double_learning=False,
            deepq_dueling_networks=False,
            per_active=False)

    def get_agent_evolve_config(self, env, genome, episodes):
        return AgentConfig(
            env=env,
            device="cpu",
            n_episodes=episodes, 
            target_average=500,
            convolutional_input=True,
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

    def get_evolver_config(self):
        return {
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
                'learningRate': [0.001, 0.002, 0.003, 0.004, 0.0001, 0.0005],
                'batchSize':    [32,64,128,256,512],
                'gamma':        [0.90, 0.92, 0.94, 0.96, 0.98, 0.9, 0.995, 0.999],
                'updateEvery':  [1,2,3,4,5,6],
                'fc1_units':    [64,128,256,512],
                'fc2_units':    [64,128,256,512],
                'memory_size':  [int(1e5), int(1e6)],
                'tau':          [1e-2, 1e-3, 1e-4],
            },
        }

    def run(self, runner):
        return runner.run_agent()

if __name__ == "__main__":
    run = AtariBreakoutDQN()
    run.command(sys.argv[1])
