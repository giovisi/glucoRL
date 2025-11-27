

# Reinforcement Learning Project

This project provides a clean and modular implementation of fundamental Reinforcement Learning (RL) algorithms. It covers the full progression from basic bandits to deep RL and policy gradient methods.

## Contents

- **Bandits** – Greedy, ε-greedy, UCB  
- **MDPs & Dynamic Programming** – Policy and value iteration  
- **Monte Carlo & TD** – MC prediction/control, TD(0), Sarsa, Q-learning  
- **Function Approximation** – Linear value/Q approximation  
- **Policy Gradients** – REINFORCE, baseline variants  
- **Actor–Critic** – Combined value + policy learning  
- **Deep RL** – DQN with target networks and replay buffer

## Run Examples

```
python bandits/experiment_bandits.py
python control/q_learning.py
python policy_gradient/reinforce.py
python deep_rl/cartpole_dqn.py
```

## Goal

Provide simple, readable implementations for studying and experimenting with RL methods.
