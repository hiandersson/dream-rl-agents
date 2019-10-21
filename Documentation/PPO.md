# Proximal Policy Optimization

### Motivation

Building on vanilla policy gradients like REINFORCE, three major challenges arises:

- Updating the policy is very ineffecient. We sample a trajectory of states, actions of rewards only once and then throw it away.

- The gradient g is very noisy. The single trajectory we a sampled by chance may not be reprasentive of our policy.

- No clear credit assignment for each action. A trajectory may contain good and bad actions and reinforcing these depend on the final outcome.

### Improvements to REINFORCE

#### Sample parallel trajectories to reduce noise

The easiest option to reduce noise is to sample more trajectories in parallel and average them out. This also gives the advantage of distributed computing since agent enviroment can be run on parallel threads or computers.

![Parallel trajectories](/Documentation/PPOParallelTrajectories.png)

#### Use credit assigment to reinforce 'good' actions

In REINFORCE, rewards in each step are added to a total reward which we then use to calculate the gradient. This gives the credit assignment problem since good and bad actions are bunched up together.

Since we have a Markov process (where each step in the process is only depending on the previous) the action in time t can only affect the future reward so we can ignore the past reward. So to correctly assign credit to each action a at time step t we only consider the future rewards and t.

![RFuture](/Documentation/RFuture.png)

#### Use importance sampling to reuse old policies

In REINFORCE we use trajectories once and throw them away. What if we can reuse them again and again to update our policy? This would make updaing the policy much more effecient.

In statistics, weighting factors is used to make a sample matching a population. For example, when sampling from a population we can use census data to understand that the balance between females and males should be 51% and 49%. If a smaller sample from the population had a balance of 60% and 40% we need to add factor weights to the female and males so it becomes a reprasensitive set, matching the census data. 

In a similar way we will use reweighting factors that takes into account how under or over–represented each trajectory is under the new policy compared to the old one. In his way we can use old trajectories for computing averages for our new policy. 

So we begin with that each single trajectory has a probability P(τ;θ) to be sampled. The same trajectory can be sampled under the new policy with probability P(τ;θ′). If we were about to calculate the average of some quantity f(τ) we could generate trajectories from the new policy, compute f(τ) and average them out. This is mathematically the same as adding up all f(τ) weighted by a probability of sampling each trajectory under the new policy:

![ReweightingFactor](/Documentation/PPOFactor.png)

By multiplying and dividing by the same number, P(τ;θ) and rearrange the terms:

![ReweightingFactor](/Documentation/PPOReweightingFactor.png)

And now we have the reweighting factors that takes into account how under or over–represented each trajectory is.

But, if we take a look a the full reweighting factor ..

![ReweightingFactor](/Documentation/PPOReweightingFull.png)

.. the formula above is a chain of products. If some policy gets close to 0 the reweighting factor gets close to zero. If some policy gets close to 1 over 0 it deverges to infitity. 

#### Proximal policy and how to update the gradient

To update the gradient we can use the formula from REINFORCE times the reweighting factor calculated when doing importance sampling:

![ReweightingFactor](/Documentation/PPOReweightingPolicyGradient.png)

We can cancel out some terms but are still left with ...

![ReweightingFactor](/Documentation/PPOCancelTerms.png)

When the factors between the old and the new policy gets close to 1 we can simply the expression even more. This is why it is called "proximal policy optimization":

![ReweightingFactor](/Documentation/PPOGradient.png)

#### The approximate gradient function, also called the "surrogate function"

Now we have an approximate form of the gradient we can think of it as the gradient of a new object and call it the surrogate function:

![Surrogate function](/Documentation/PPOSurrogate.png)

We can use this function to peform gradient ascent (maximize the expected future reward in each step of the trajectory by increasing the probability for actions that leads to a higher reward).

One problem with this is our new policy might start to differ too much from the old one so our approximation that does not hold anymore (remember that we simplified our expression above with the assumption that the factors are close to 1).

#### Keeping things in check with the clipped surrogate function

If the approximation does not hold anymore (the ratio being close to 1) this can lead to a really bad policy. When the policy changes by a large amount it is easy that we run over a cliff (from being at the peak of expected future reward, from the policy perspective). After that we might be stuck in a flat bottom and not be able to recover.

The way to solve this problem is to 

![Clipped surrogate](/Documentation/PPOClippedSurrogate.png)

