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
            n_episodes=1500, 
            learning_rate=3e-4,
            fc1_units=256,
            ppo_epsilon=0.1,
            ppo_sgd_steps_per_episode=4,
            ACTION_0=4,
            ACTION_1=5,
            max_t=500)

    def get_agent_evolve_config(self, envs, genome, device, episodes):
        return AgentConfig(
            parallelEnv=envs, 
            n_episodes=episodes, 
            device=device,
            learning_rate=genome['learningRate'],
            fc1_units=genome['fc1_units'],
            ACTION_0=4,
            ACTION_1=5,
            max_t=500)

    def get_evolver_config(self):
        return {
            'episodes': 2,
            'GPUDevices': None,
            'populationSize': 10,
            'retainSize': 5,
            'mutateOneGeneRandom': 5,
            'mutateTwoGenesRandom': 5,
            'crossoverOneGene': 5,
            'crossOverTwoGenes': 5,
            'mutateOneGeneClose': 5,
            'generations': 1,
            'randomSeed': 1,
            'allPossibleGenesSimple': {
                'learningRate': [5e-3,1e-3,5e-4,1e-4,5e-5,1e-5],
                'fc1_units':    [32,64,128,256,512.1024],
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
