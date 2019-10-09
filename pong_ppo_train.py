# Internals
from Agents.PPO import PPOAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Create environment
from Agents.Utils.parallelEnv import parallelEnv
envs = parallelEnv('PongDeterministic-v4', n=8, seed=1234)

# Create the agent
config = AgentConfig(parallelEnv=envs, max_t=320)
config.RIGHT = 4
config.LEFT = 5
agent = PPOAgent(envs, config)

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
score = runner.run_parallel_trajectories()

# Visualize the score
score.visualize()

# Close
env.close()
