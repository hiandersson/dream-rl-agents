import gym
from Agents.Common import Runner
from Agents.DQN import DQNAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("CartPole-v1")

# Create the agent
agent = DQNAgent(env, AgentConfig(env=env))

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
runner.random()

env.close()