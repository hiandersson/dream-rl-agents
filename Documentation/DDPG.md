# Deep Deterministic Policy Gradients (DDPG)

## Overview and motivation

#### Actor-critic

DDPG is a solution to using Deep Q Learning in continous action space. In reinforcement learning, policy based methods learns to estimate the optimal policy directly from a given state and outputs a probability distribution (an on-policy method that is not deterministic). Action-value based methods learns the optimal action-value function from sampled state transitions **s,a,r,s'** in memory and outputs a discrete action-value function (an off-policy method that is derministic, it always selects the best possible action).
- Policy based methods tends to have high variance and low bias because they are using a Monte Carlo estimate to calculate the total discounted reward (rolled out from the whole sequence).
- An action-value based methods tend to have low variance but high bias because they are using a Temporal Difference estimate to learn the optimal action value function. The next estimate is based on the previous and so we are compounding estimates upon estimates.

![VarianceBias](/Documentation/VarianceBias.png)

An actor-critic model tries to combine the best of two worlds by letting the actor and critic work together and use an actor-critic architecture, conceptualized here:

![ActorCritic](/Documentation/ActorCritic.png)

#### How DDPG works

An actor is using a policy-based approach and learning how to act by directly estimating the optimal policy and maximizing reward through gradient ascent. A critic, utilizes the value-based approach and learns how to estimate the value of different state — action pairs. The critic learns to evaluate the optimal action value function by using the actors best believed action. The actor is then learning the argmax **a** Q(**s, a**), which is the best action.

![ActorCriticModel](/Documentation/ActorCriticModelB.png)

## DDPG algorithm

#### Algorithm
* The actor estimates the next action  **a'** based on next state **s'** from a sample transition **s,a,r,s'** from a replay buffer (described below) and using the actors target network **s'**
* The critic does gradient descent using the loss function L = mean squared error (Q_target - Q_expected). Q_target is calculated by reward + gamma * critics target network(**s', a'**). Q_expected is calculated using the critics local network(**s,a**).
* The actor does gradient ascent (hence the - sign) using the loss function L = -critics local network(**s**, actor_local_network(**s**)).mean() where the actors local network(**s**) is the best belived action by the actor, then evaluated by the critics local network.

#### Implementation details
* A replay buffer is used to store state transitions s,a,r,s' from the agent acting in the environment. With a certain update frequency the agent learns by uniform random sampling of state transitions from the buffer. 
* A soft update function is used to stabilize learning by using using local and target networks for the actor and critic. The target networks are used in calculations using a' and s'. The local networks are used in calculations using s and a. A parameter Tau is used as a parameter for fast the two networks blend together.
* Noise is induced in the action space through an Ornstein–Uhlenbeck process in order to encourage exploration

## Enhancements to DDPG

* **Prioritized experience replay**. The idea is that when we store state transitions in replay memory we also store the error between the Q value target and the expected Q, the temporal difference error, or TD error. When we sample from replay memory to learn we do so with a probability that is the TD error. Larger error have a higher probability to be sampled.

* **Batch normalization**. The idea is to limit covariate shift by normalizing the activations of each layer (transforming the inputs to be mean 0 and unit variance). This allows each layer to learn on a more stable distribution of inputs, and would thus accelerate the training of the network.

## Experiment and result - LunarLander 

#### Environement

LunarLanderContinuous-v2 is an OpenAI gym environment, defined as:

"Landing pad is always at coordinates (0,0). Coordinates are the first two numbers in state vector. Reward for moving from the top of the screen to landing pad and zero speed is about 100..140 points. If lander moves away from landing pad it loses reward back. Episode finishes if the lander crashes or comes to rest, receiving additional -100 or +100 points. Each leg ground contact is +10. Firing main engine is -0.3 points each frame. Solved is 200 points. Landing outside landing pad is possible. Fuel is infinite, so an agent can learn to fly and then land on its first attempt. Four discrete actions available: do nothing, fire left orientation engine, fire main engine, fire right orientation engine."

"LunarLanderV2 defines "solving" as getting average reward of 200.0 over 100 consecutive trials."

#### A random agent

![Random agent](/Checkpoints/lunarlander-v2-random.gif)

#### A trained agent - 50% slow motion

![Trained agent](/Checkpoints/lunarlander-v2-enjoy.gif)

#### Hyper parameters for training

Some of the core hyper parameters used:

````
update_every=1,               # Learn at every step
memory_size=10000,            # Size of replay memory
batch_size=64,                # Batch size 
gamma=0.98,                   # Gamma / discount factor
actor_learning_rate=1e-4,     # Actor learning rate
critic_learning_rate=1e-3,    # Critic learning rate
fc1_units=400,                # Number of neurons in the first hidden layer
fc2_units=300,                # Number of neurons in the second hidden layer
per_active=False,             # Do not use Prioritized Experience Replay 
batch_norm=False,             # Do not use Batch Normalization 
````

#### Plot of rewards over time - environment solved in episode 322

![Plot](/Checkpoints/lunarlander_v2_plot.png)
