# Proximal Policy Optimization

#### Policy gradient methods

Policy gradient methods directly takes in the state of the environment and outputs a probability distribution for an action. Compared to off policy methods like DQN, policy gradient methods does not use a replay buffer to sample from.

A policy gradient method can only sample episodic tasks because we need to know the future discounted reward in each step when we update our policy. When we update our neural network weights we look at each state-action pair in the episode an up the probability for actions leading to a higher future reward and down the probability for actions leading to a lower future reward. Since we want to maximize the rewards we use gradient ascent when we update our neural network.

Policy gradient methods har similar to supervised learning in that we collect the whole episode, calculated the future discounted reward and then updated our networ, we have created a setup that is similar to supervised learning of X, Y target pairs.
