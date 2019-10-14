from run_experiment_helper import *
from Agents.DQN import DQNAgent

class AtariBreakoutDQN(RunExperiment):
    def __init__(self):
        RunExperiment.__init__(self, "BreakoutDeterministic-v4")

    def get_agent(self, config):
        return DQNAgent(config)

    def get_agent_train_config(self, env):
        return AgentConfig(
            env=env,
            n_episodes=1000, 
            target_average=500,
            convolutional_input=True,
            memory_size=1000000,
            update_every=4,
            batch_size=32,
            gamma=0.99,
            learning_rate=0.00001,
            fc1_units=1024,
            deepq_double_learning=True,
            deepq_dueling_networks=True,
            per_active=True)

    def get_agent_evolve_config(self, env, genome, device, episodes):
        return AgentConfig(
            env=env,
            device=device,
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
            'GPUDevices': ['cuda:1'],
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
                'learningRate': [0.001, 0.0001, 0.0005, 0.00001, 0.00005],
                'batchSize':    [16,32,64,128],
                'gamma':        [0.97, 0.98, 0.9, 0.995, 0.999],
                'updateEvery':  [2,3,4,5,6],
                'fc1_units':    [64,128,256,512,1024],
                'fc2_units':    [64,128,256,512,1024],
                'memory_size':  [int(1e5), int(1e6)],
                'tau':          [1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            },
        }

    def run(self, runner):
        return runner.run_agent()

if __name__ == "__main__":
    run = AtariBreakoutDQN()
    run.command(sys.argv[1])
