# Externals
import gym

# Internals
from Agents.PPO import PPOAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner
from Agents.Utils.parallelEnv import parallelEnv

# Create an agent that solves the environment:
envs = parallelEnv("PongDeterministic-v4", n=8, seed=1234)

# Create the agent
config = AgentConfig(
    parallelEnv=envs, 
    n_episodes=1500, 
    learning_rate=1e-4,
    fc1_units=256,
    ACTION_0=4,
    ACTION_1=5,
    max_t=500)

agent = PPOAgent(config)

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
score = runner.run_parallel_trajectories()

# Visualize the score
score.visualize()

# Close
env.close()