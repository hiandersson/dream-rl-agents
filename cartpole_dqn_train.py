import gym
from Agents.Common import Runner
from Agents.DQN import DQNAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Make the gym
env = gym.make("CartPole-v1")

# Create an agent that solves the environment:
# - CartPole-v1 defines "solving" as getting average reward of 475.0 over 100 consecutive trials."
config = AgentConfig(
    env=env,
    n_episodes=1000, 
    target_average=475,
    update_every=1,
    batch_size=64,
    gamma=0.95,
    learning_rate=1e-4,
    deepq_double_learning=True,
    deepq_dueling_networks=True,
    per_active=True)
agent = DQNAgent(env, config)

# Create a runner that runs the agent in the environment
runner = Runner(agent, save_best_score='Checkpoints/cardpole_v1_dqn.ch')

# Run the agent
score = runner.run_agent()

# Visualize the score
score.visualize()

env.close()