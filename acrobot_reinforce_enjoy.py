import gym
from Agents.Common import Runner
from Agents.REINFORCE import REINFORCEAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("Acrobot-v1")

# Create the agent
agent = REINFORCEAgent(AgentConfig(env=env))

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
runner.enjoy_checkpoint('Checkpoints/acrobot_v1_reinforce_evo.ch')

env.close()