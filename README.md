# ModelBasedRLTensorDecomp
Code for my position paper "Model based Multi-agent Reinforcement Learning with Tensor Decompositions" https://arxiv.org/abs/2110.14524#

experiment.py contains the main logic to execute an experiment

agent.py contains an Agent class that create estimates for the transition and reward functions, as well as a Policy class that performs policy improvement (Algorithm 3)

torch_CPDecomp.py contains classes for CP-decompositions (this could probably be swapped out for other decomposition packages e.g. tensorly)

alternating_rank_1_updates.py contains an implementation of Algorithms 1, 3, 4

transition_tensor_prep.py contains Algorithm 7 and is used to create transition tensors.

The results to generate the figures as well as python files to plot the results are included in experiment_1_results and experiment_2_results.
