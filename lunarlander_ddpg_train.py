# Externals
import gym

# Internals
from Agents.DDPG import DDPGAgent
from Agents.Common import AgentConfig
from Agents.Common import Runner

# Create environment
env = gym.make("LunarLanderContinuous-v2")

# Create the agent
config = AgentConfig(
    env=env,
    seed=42,
    target_average=200, 
    n_episodes=500, 
    batch_norm=False, 
    batch_size=64, 
    gamma=0.98, 
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-3,
    weight_decay=0,
    update_every=1,
    ou_noise_sigma=0.01,
    fc1_units=400,
    fc2_units=300,
    per_active=False)

agent = DDPGAgent(env, config)

# Create a runner that runs the agent in the environment
runner = Runner(agent)

# Run the agent
score = runner.run_agent()

# Visualize the score
score.visualize()

# Close
env.close()
