from run_experiment_helper import *
from Agents.REINFORCE import REINFORCEAgent

class AcrobotREINFORCE(RunExperiment):
    def __init__(self):
        RunExperiment.__init__(self, "Acrobot-v1")

    def get_agent(self, config):
        return REINFORCEAgent(config)

    def get_agent_train_config(self, env):
        return AgentConfig(
            env=env,
            n_episodes=300, 
            target_average=None,
            max_t=200,
            gamma=0.92,
            fc1_units=512,
            learning_rate=0.002)

    def get_agent_evolve_config(self, env, genome, device, episodes):
        return AgentConfig(
            env=env,
            device=device,
            n_episodes=episodes, 
            target_average=None,
            max_t=200,
            gamma=genome['gamma'],
            learning_rate=genome['learningRate'],
            fc1_units=genome['fc1_units'])

    def get_evolver_config(self):
        return {
            'episodes': 200,
            'GPUDevices': None,
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
                'learningRate': [0.01, 0.001, 0.002, 0.003, 0.004, 0.0005],
                'gamma':        [0.90, 0.92, 0.94, 0.96, 0.98, 0.9, 0.995, 1.0],
                'fc1_units':    [16,32,64,128,256,512],
            },
        }

    def run(self, runner):
        return runner.run_single_probability_trajectory()

if __name__ == "__main__":
    run = AcrobotREINFORCE()
    run.command(sys.argv)
