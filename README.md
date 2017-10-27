# Introduction

I collect some DQN implementation here, some of them are little modified, eg: The backup process may not step by step.
In these implementation, I take [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) as my testbed to estimate performance.

# File Information
- morvan_dqn_pytorch: This version is collected from [MorvanZhou/PyTorch-Tutorial](https://github.com/MorvanZhou/PyTorch-Tutorial/blob/master/tutorial-contents/405_DQN_Reinforcement_learning.py)
- modified_from_morvan_dqn.py: This file is modified from the original version: `morvan_dqn_pytorch`.
- n1try_dqn.py: collected from gym-CartPole-v0 Ladder: [link](https://gym.openai.com/evaluations/eval_EIcM1ZBnQW2LBaFN6FY65g/)
- modified_from_n1try_dqn.py: combine transformed reward from morvan's dqn and n1try_dqn.py together.
- dqn_cartpole.py: implemented by myself.

# Performance
- condition: env is unwrapped.

- dqn_cartpole.py: it periodly performs badly.

![dqb_cartpole](/result_diagram/dqn_cartpole.png)

- modified_from_n1try_dqn.py

![modified_n1try](/result_diagram/modified_n1try.png)