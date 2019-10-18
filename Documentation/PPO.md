# Proximal Policy Optimization

### Motivation

Building on vanilla policy gradients like REINFORCE, three major challenges arises:

- Updating the policy is very ineffecient. We sample a trajectory of states, actions of rewards only once and then throw it away.

- The gradient g is very noisy. The single trajectory we a sampled by chance may not be reprasentive of our policy.

- No clear credit assignment for each action. A trajectory may contain good and bad actions and reinforcing these depend on the final outcome.

### Improvements to REINFORCE

#### Noise reduction

The easiest option to reduce noise is to sample more trajectories in parallel and average them out. This also gives the advantage of distributed computing since agent enviroment can be run on parallel threads or computers.

![Parallel trajectories](/Documentation/PPOParallelTrajectories.png)

#### Credit assigment

In REINFORCE the rewards in each step are summed to a total reward which we then use to calculate the gradient. This gives the credit assignment problem since good and bad actions are bunched up together.

Since we have a Markov process (where each step in the process is only depending on the previous) the action in time t can only affect the future reward so we can igonre the past reward. So to correctly assign credit to each action a at time step t we only consider the future rewards and t.

![RFuture](/Documentation/RFuture.png)

#### Importance sampling

In REINFORCE we use trajectories once and throw them away. What if we can reuse them again and again to update our policy? This would make updaing the policy much more effecient.

In REINFORCE each trajectory had the probability P(τ;θ) to be samples. The same trajectory can be sampled under the new policy P(τ;θ′)
