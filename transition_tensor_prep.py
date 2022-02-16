import torch_CPDecomp as cpd
import numpy as np
import torch
from agent import MemoryBuffer, Agent, Policy
import matplotlib.pyplot as plt
import math
import pickle
from tqdm import tqdm
import sys
from alternating_rank_1_updates import alternating_rank_1_updates

for j in tqdm(range(20)):
    T = cpd.CPTensor.from_components([torch.tensor(generate_factor_matrix((d, transition_rank))) for d in transition_shape], torch.tensor(transition_weights))
    prev_T = cpd.CPTensor([torch.tensor(generate_factor_matrix((d, transition_rank))) for d in transition_shape], torch.tensor(transition_weights))
    i = 0
    normalization_loss = 1
    while normalization_loss > 1e-6 and i < 20:
        T = np.array(T)
        T = T * (T > 0)
        invalid_rows = np.where(T.sum(axis=-1) == 0)
        T[(*invalid_rows, (np.arange(len(invalid_rows[0])) + 1) % S)] = 1
        T = T / T.sum(axis=-1)[:, :, :, :, None]
        T = torch.tensor(T)

        T_decomp = alternating_rank_1_updates(T, 20, N_power_iter=5, N_altmin=5, start=prev_T)
        T_decomp.sort()
        prev_T = T_decomp.copy()
        normalization_loss = torch.mean( (T - T_decomp.drop_components(list(range(5, 20))).to_tensor())**2 )
        print(T_decomp.weights, torch.mean( (T - T_decomp.to_tensor())**2 ), normalization_loss)
        T_decomp = T_decomp.drop_components(list(range(5, 20)))
        T = np.array(T_decomp.to_tensor())
        i += 1
    

    T = T * (T > 0)
    invalid_rows = np.where(T.sum(axis=-1) == 0)
    T[(*invalid_rows, (np.arange(len(invalid_rows[0])) + 1) % S)] = 1
    T = T / T.sum(axis=-1)[:, :, :, :, None]
    T = torch.tensor(T)
    np.save('20_10_10_10_rank5_run' + str(j), T)