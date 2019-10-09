# Deep Q Learning

## Reinforcement learning

In reinforcement learning an **agent** learns to interact with the environment using raw sensory input, its own actions and rewards (more or less sparse in time) are handed out for the behaviour we want to reinforce. Unlike supervised learning where there is a correct label Y for any input X there is no labeling in reinforcment learning.

The agent interacts with the environment via a Markov chain: state, action, reward and then next state:

![Markov](/Documentation/MarkovChain.png)

## The Deep Q Learning algorithm (DQN)

#### Q Learning

In Q learning the goal of the agent is to maximize the future discounted reward in any given state of the agent. It does this by adding the maximum reward attainable from future states to the reward for achieving its current state, effectively influencing the current action by the potential future reward. This potential reward is a weighted sum of the expected values of the rewards of all future steps starting from the current state.

The formal notation for Q-learning is this (source: https://en.wikipedia.org/wiki/Q-learning#Deep_Q-learning):

![Q Learning](/Documentation/DeepQLearning.png)

Explanations
- Old value, Q(s,a) - the old policy in this state for the agent
- Learning rate, alpha - to what extent does new information override old. A learning rate of 0 means nothing is learnt in this update. A learning rate of 1 means that the agent fully overrides all prior learning in this update.
- Reward, r - handed out by the environment in state S. 
- Discount factor, gamma - A gamma of zero makes the agent short sighted and only consider immidiate future rewards. A gamma of close to one makes the agent consider long term future rewards (which is often desired).
- MaxQ - estimate of optimal future value reward in the **next state** in the environment. The optimal value is the max Q value of all possible actions in the next state.

#### Deep Q Learning 

In Deep Q Learning we build a state-value function, Q(s,a), adapting a neural network as a function approximator. Doing this enables us to handle larger continous state spaces and solving bigger problems:

![Deep Q Learning](/Documentation/DeepQ.png)

## Enhancements to DQN

The vanilla DQN algorithm is not robust in terms of learning, therefore many additions have been added over time and these are the ones implemented in this library:

* **Prioritized experience replay**. The idea is that when we store state transitions in replay memory we also store the error between the Q value target and the expected Q, the temporal difference error, or TD error. When we sample from replay memory to learn we do so with a probability that is the TD error. Larger error have a higher probability to be sampled.
* **Double Q Learning**. The idea is to reduce reward overestimations by decomposing the max operation in the target into action selection and action evaluation.
* **Dueling networks**. The idea is that instead of just estimating the state value function we also estimate the advantage A of this action compared to other actions. Intuitively our dueling network can learn which states are (or are not) valuable without having to learn the effect of each action at each state.
* **Batch normalization**. The idea is to limit covariate shift by normalizing the activations of each layer (transforming the inputs to be mean 0 and unit variance). This allows each layer to learn on a more stable distribution of inputs, and would thus accelerate the training of the network.

## Experiment and result - CartPole

#### Environement

CartPole is an OpenAI gym environment, defined as:

"A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright. The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center."

The environment is "solved" as:

"CartPole-v1 defines "solving" as getting average reward of 475.0 over 100 consecutive trials."

#### A random agent

![Random agent](/Checkpoints/cartpole_v1_random.gif)

#### A trained agent

![Trained agent](/Checkpoints/cartpole_v1_enjoy.gif)

#### Hyper parameters for training

Some of the core hyper parameters used:

````
update_every=1,               # Learn at every step
memory_size=10000,            # Size of replay memory
batch_size=64,                # Batch size 
gamma=0.95,                   # Gamma / discount factor
learning_rate=1e-4,           # Learning rate
fc1_units=128,                # Number of neurons in the first hidden layer
fc2_units=128,                # Number of neurons in the second hidden layer
deepq_double_learning=True,   # Use Double Q Learning 
deepq_dueling_networks=True,  # Use Dueling networks 
per_active=True               # Use Prioritized Experience Replay 
batch_norm=True               # Use Batch Normalization 
````

#### Plot of rewards over time

![Plot](/Checkpoints/cartpole_v1_plot.png)
