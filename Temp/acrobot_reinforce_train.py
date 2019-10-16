import gym
from Agents.Common import Runner
from Agents.REINFORCE import REINFORCEAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("Acrobot-v1")

# Create an agent that solves the environment:
# "Acrobot-v1 is an unsolved environment, which means it does not have a specified reward threshold at which it's considered solved"
config = AgentConfig(
    env=env,
    n_episodes=300, 
    target_average=-100,
    max_t=200,
    gamma=0.92,
    fc1_units=512,
    learning_rate=0.002)
agent = REINFORCEAgent(config)

# Create a runner that runs the agent in the environment
runner = Runner(agent, save_best_score='Checkpoints/acrobot_v1_reinforce_train.ch')

# Run the agent
score, checkpoint = runner.run_single_probability_trajectory()

# Visualize the score
score.visualize()

env.close()