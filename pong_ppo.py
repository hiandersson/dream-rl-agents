from run_experiment_helper import *
from Agents.PPO import PPOAgent

class PongPPO(RunExperiment):
    def __init__(self):
        RunExperiment.__init__(self, "PongDeterministic-v4", parallelEnvs=True)

    def get_agent(self, config):
        return PPOAgent(config)

    def get_agent_train_config(self, envs):
        return AgentConfig(
            parallelEnv=envs, 
            n_episodes=2000, 
            RIGHT=4,
            LEFT=5,
            max_t=500)

    def get_agent_evolve_config(self, envs, genome, device, episodes):
        return AgentConfig(
            parallelEnv=envs, 
            n_episodes=episodes, 
            device=device,
            RIGHT=4,
            LEFT=5,
            max_t=500)

    def get_evolver_config(self):
        return {
            'episodes': 2,
            'GPUDevices': None,
            'populationSize': 5,
            'retainSize': 5,
            'mutateOneGeneRandom': 5,
            'mutateTwoGenesRandom': 5,
            'crossoverOneGene': 5,
            'crossOverTwoGenes': 5,
            'mutateOneGeneClose': 5,
            'generations': 1,
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

    def enjoy_checkpoint(self, runner, filename):

         runner.enjoy_checkpoint_frame_batches_probabilities(filename) 

         pass

    def run(self, runner):
        
        return runner.run_parallel_trajectories()

if __name__ == "__main__":
    run = PongPPO()
    run.command(sys.argv)
