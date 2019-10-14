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

    def run(self, runner):
        return runner.run_agent()

if __name__ == "__main__":
    run = AtariBreakoutDQN()
    run.command(sys.argv[1])
