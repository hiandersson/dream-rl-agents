import gym
from Agents.Common import Runner
from Agents.DDPG import DDPGAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("LunarLanderContinuous-v2")

# Create the agent
agent = DDPGAgent(env, AgentConfig(
    fc1_units=400,
    fc2_units=300,   
    batch_norm=False,  
    env=env))

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
runner.enjoy_checkpoint('Checkpoints/lunarlander_v2_ddpg.ch')

env.close()