import gym
from Agents.Common import Runner
from Agents.DDPG import DDPGAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("LunarLanderContinuous-v2")

# Create the agent
agent = DDPGAgent(env, AgentConfig(env=env))

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
runner.random()

env.close()