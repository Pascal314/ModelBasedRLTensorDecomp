import torch_CPDecomp as cpd
import torch
import numpy as np
from alternating_rank_1_updates import alternating_rank_1_updates
from tqdm import tqdm
import math
import itertools

class MemoryBuffer:
    def __init__(self, state_shape, actions_shape):
        if isinstance(state_shape, tuple):
            self.state_shape = state_shape
        else:
            self.state_shape = (state_shape,)

        if isinstance(state_shape, tuple):
            self.actions_shape = actions_shape
        else:
            self.actions_shape = (actions_shape,)

        self.transition_buffer = torch.zeros((*state_shape, *actions_shape, *state_shape), dtype=torch.int32)
        self.reward_buffer = torch.zeros( (*state_shape, *actions_shape), dtype=torch.float64)
        self.reward_mask = torch.zeros(self.reward_buffer.shape, dtype=bool)

    def store(self, state, action, next_state, reward):
        if not isinstance(state, tuple):
            state = (state,)
            next_state = (next_state,)
        if not isinstance(action, tuple):
            action = (action, )

        self.transition_buffer[(*state, *action, *next_state)] += 1
        self.reward_buffer[(*state, *action)] = reward
        self.reward_mask[(*state, *action)] = 1

class Agent:
    def __init__(self, 
        state_shape, actions_shape, use_decompositions=True,
        transition_decomp_rank=5, reward_decomp_rank=5, gamma=0.9,
    ):
        self.memory = MemoryBuffer(state_shape, actions_shape)
        self.policy = Policy(state_shape, actions_shape, gamma=gamma)
        self.transition_decomp_rank = transition_decomp_rank
        self.reward_decomp_rank = reward_decomp_rank
        self.state_shape = state_shape
        self.actions_shape = actions_shape
        self.n_agents = len(actions_shape)
        self.use_decompositions = use_decompositions

    def move(self, state):
        move = self.policy.pi[state]
        return tuple((x.item() for x in move))

    def approximate_T(self):
        transition_buffer = self.memory.transition_buffer.double()
        T = torch.nan_to_num(
            transition_buffer / torch.sum(transition_buffer, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)], 1 / math.prod(self.state_shape)
        )
        if self.use_decompositions:
            T = alternating_rank_1_updates(T, self.transition_decomp_rank, N_power_iter=20, N_altmin=20).to_tensor()
            T = T * (T > 0)
            T = torch.nan_to_num(
                T / torch.sum(T, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)],
                1 / math.prod(self.state_shape)
            )
        return T

    def approximate_R(self):
        if self.use_decompositions:
            R = alternating_rank_1_updates(self.memory.reward_buffer, self.reward_decomp_rank, mask=self.memory.reward_mask, N_power_iter=20, N_altmin=20)
            R = R.to_tensor()

        else:
            R = self.memory.reward_buffer + torch.mean(self.memory.reward_buffer[self.memory.reward_mask]) * (~self.memory.reward_mask).double()
        
        return R

class TesseractAgent(Agent):
    def approximate_T(self):
        transition_buffer = self.memory.transition_buffer.double()
        T = torch.nan_to_num(
            transition_buffer / torch.sum(transition_buffer, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)], 1 / math.prod(self.state_shape)
        )
        
        T_placeholder = torch.zeros(T.shape)
        for s_1 in itertools.product(*[range(n) for n in self.state_shape]):
            for s_2 in itertools.product(*[range(n) for n in self.state_shape]):
                
                approx = alternating_rank_1_updates(T[ s_1 + (slice(None), ) * self.n_agents + s_2], self.transition_decomp_rank, N_power_iter=20, N_altmin=20).to_tensor()
                approx = approx * (approx > 0)
                T_placeholder[ s_1 + (slice(None), ) * self.n_agents + s_2] = approx

        T_placeholder = torch.nan_to_num(
                T_placeholder / torch.sum(T_placeholder, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)],
                1 / math.prod(self.state_shape)
        )
        return T_placeholder

    def approximate_R(self):
        R_placeholder = torch.zeros(self.memory.reward_buffer.shape)
        for s_1 in itertools.product(*[range(n) for n in self.state_shape]):
            R_placeholder[s_1] = alternating_rank_1_updates(
                self.memory.reward_buffer[s_1], self.reward_decomp_rank, mask=self.memory.reward_mask[s_1], N_power_iter=20, N_altmin=20
            ).to_tensor()
        return R_placeholder

class TransitionCompletionAgent(Agent):
    def approximate_T(self):
        transition_buffer = self.memory.transition_buffer.double()
        T = torch.nan_to_num(
            transition_buffer / torch.sum(transition_buffer, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)], 0
        )
        T = alternating_rank_1_updates(T, self.transition_decomp_rank, mask= torch.sqrt(self.memory.transition_buffer) / torch.sqrt(torch.max(self.memory.transition_buffer)), N_power_iter=20, N_altmin=20).to_tensor()
        T = T * (T > 0)
        T = torch.nan_to_num(
            T / torch.sum(T, axis=tuple(range(-len(self.state_shape), 0)))[ [...] + [None] * len(self.state_shape)],
            1 / math.prod(self.state_shape)
        )
        return T

class Policy:
    def __init__(self, state_shape, actions_shape, gamma=0.9):
        if isinstance(state_shape, int):
            state_shape = (state_shape,)
        n_agents = len(actions_shape)
        pi = np.random.randint( [0]*n_agents, actions_shape, (*state_shape, len(actions_shape)))
        self.pi = torch.tensor(data=pi, dtype=torch.int64)
        self.gamma = gamma
        self.state_shape = state_shape
        self.actions_shape = actions_shape

    def improve(self, T, R, n_iter=10, verbose=False):
        Q = self.compute_Q_tensor(T, R)
        V = self.compute_V_tensor(Q)
        index_array = torch.tensor(np.stack(np.meshgrid(*[np.arange(d) for d in self.state_shape]), axis=len(self.state_shape)).swapaxes(0,1)).reshape(-1, len(self.state_shape))

        rng = tqdm(range(n_iter)) if verbose else range(n_iter)


        for _ in rng:
            for index in index_array:
                best = torch.argmax(Q[index])
                if V[index] < Q[index].flatten()[best]:
                    self.pi[index] = torch.tensor(np.unravel_index(best, self.actions_shape))
            
            Q = self.compute_Q_tensor(T, R)
            V = self.compute_V_tensor(Q)

    def compute_Q_tensor(self, T, R, n_iter=300):
        # This can be done with inverses too
        
        Q = torch.ones(R.shape)
        for _ in range(n_iter):
            Q = R + self.gamma * torch.sum(T * self.compute_V_tensor(Q)[ [None] * (len(self.state_shape) + len(self.actions_shape))], axis=-1)
        
        return Q

    def compute_V_tensor(self, Q):
        index_array = torch.tensor(np.stack(np.meshgrid(*[np.arange(d) for d in self.state_shape]), axis=len(self.state_shape)).swapaxes(0,1))
        assert index_array.shape[0] == len(self.state_shape)
        index_array = torch.cat([index_array.T, self.pi], axis=-1)
        index_array = index_array.reshape(-1, len(self.state_shape) + len(self.actions_shape))
        V = Q[tuple(index_array[:, i] for i in range(index_array.shape[1]))]
        return V
