# Dream RL Agents
## What is Dream RL Agents?
- An open source library for deep reinforcement learning agents
- Code and articles written to be easy to understand and learn from
- Implements Deep Q Learning, Deep Deterministic Policy Gradients and Proximal Policy Optimization
- Adds several additional features like Prioritized Experience Replay and Dueling Networks etc.
- "Solves" OpenAI's gym environments in example files
- Not meant for production. Can contain subtle bugs. Use OpenAI's baselines instead.

## Agents

The following agents have been implemented and documented:
- Deep Q Learning including Double Q Learning, Dueling Networks and Prioritized Experience Replay
- Deep Deterministic Policy Gradients
- Proximal Policy Gradients
- REINFORCE

### Deep Q Learning agent

#### A trained agent

![Trained agent](/Checkpoints/cartpole_v1_enjoy.gif)

#### Documentation

[Go to article about Deep Q Learning](/Documentation/DQN.md)

#### How to use

````
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
````

### Deep Deterministic Policy Gradients agent

#### A trained agent

![Trained agent](/Checkpoints/lunarlander-v2-enjoy.gif)

#### Documentation

[Go to article about Deep Deterministic Policy Gradients](/Documentation/DDPG.md)

#### How to use

````
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
````

### Proximal Policy Optimization agent
