# Proximal Policy Optimization

#### Policy gradient methods

Policy gradient methods directly takes in the state of the environment and outputs a probability distribution for an action. Compared to off policy methods like DQN, policy gradient methods does not use a replay buffer to sample from.

A policy gradient method can only sample episodic tasks because we need to know the future discounted reward in each step when we update our policy. We call sequence of state-action pairs in an episode a trajectory. When we update our neural network weights we look at each state-action pair in the trajectory an up the probability for actions leading to a higher future reward and down the probability for actions leading to a lower future reward. Since we want to maximize the rewards we use gradient ascent when we update our neural network.

Policy gradient methods are similar to supervised learning in that when we have collected a trajectory and calculated the future discounted reward for each step, we have created a setup that is similar to supervised learning of X, Y target pairs (say training a convolutional neural network to regonize images).

#### REINFORCE algorithm

The goal of a policy gradient method is to maximize the future expected return. We can begin with set this up as the follow equation, calculating the expectation of rewards in a trajectory:

![Problem setup](/Documentation/REINFORCE_Goal.png)

- Tau is a trajectory, a sequence of state action pairs collected during an episode
- U(theta) is the expected return
- P(tau;theta) is the probability of the trajectory, depending on theta
- R(tau) is the return from the trajectory

The goal is to find the weights of theta that maximizes the future expected return.

Consider that we have collected **one** trajectory, we can **estimate** the gradient **g hat** that we use to perform gradient ascent this way:

![Gradient](/Documentation/REINFORCE_Gradient.png)

- R(tau) is the cumulative reward from the trajectory
- Policy pi (at | st) is the probability of selecting an action **a** in state **t**. The policy is parameterized by theta which is the neural network.
- The full gradient statement takes the log probability of selecting an action **a** in state **t**.

If we take a step in the direction of the gradient when we update the neural network, we will increase the log probability of taking action **a** in that step.
