# Proximal Policy Optimization

#### Motivation

Building on vanilla policy gradients like REINFORCE, three major challenges arises:

- Updating the polic is very ineffecient. We sample a trajectory of states, actions of rewards only once and then throw it away.

- The gradient g is very noisy. The single trajectory we a sampled by chance may not be reprasentive of our policy.

- No clear credit assignment for each action. A trajectory may contain good and bad actions and reinforcing these depend on the final outcome.
