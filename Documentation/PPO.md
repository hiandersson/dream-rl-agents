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
