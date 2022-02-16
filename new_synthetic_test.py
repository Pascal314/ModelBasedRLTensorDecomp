import torch_CPDecomp as cpd
import numpy as np
import torch
from agent import MemoryBuffer, Agent, Policy, TransitionCompletionAgent, TesseractAgent
import matplotlib.pyplot as plt
import math
import pickle
from tqdm import tqdm
import sys
from alternating_rank_1_updates import alternating_rank_1_updates

def generate_factor_matrix(shape):
    X = np.random.uniform(-1, 1, size=shape)
    X = X / np.sqrt(np.sum(X**2, axis=0))
    return X

class SyntheticEnv:
    def __init__(self, T, R):
        self.T = T
        self.R = R
        self.n_states = T.shape[0]
        self.n_actions = T.shape[1:-1]

    def reset(self):
        self.state = 0
        self.time = 0
        return self.state

    def step(self, action):
        reward = self.R[(self.state, *action)]
        self.state = np.random.choice(self.n_states, p=self.T[(self.state, *action)])

        self.time += 1
        if self.time > 100:
            done = True
        else:
            done = False
        info = {}
        return self.state, reward, done, info

    def sample_action(self):
        action = tuple([np.random.randint(0, d) for d in self.n_actions])
        return action


def generate_env_variables(j):
    S = 20
    A = 10
    transition_rank = 5
    reward_rank = 5

    state_shape = (S,)
    actions_shape = (A, A, A)

    transition_shape = (S, A, A, A, S)
    reward_shape = (S, A, A, A)

    transition_weights = np.linspace(0.1, 1, transition_rank)
    reward_weights = np.linspace(0.1, 1, reward_rank)

    T = np.load(f'prepared_transition_tensors/20_10_10_10_rank5_run{j}.npy')
    T = torch.tensor(T)
    R = cpd.CPTensor.from_components([torch.tensor(generate_factor_matrix((d, reward_rank))) for d in reward_shape], torch.tensor(reward_weights))
    return T, R, state_shape, actions_shape

def train(env, agent, epochs=10, improve_every=5, test_every=5, T=None, R=None, epsilon=0, epsilon_decay=0, min_epsilon=0):
    true_visit_counts = torch.zeros(T.shape, dtype=torch.int32)
    total_train_rewards = []
    test_rewards = []
    model_checks = []

    model_checks.append(check_models(agent, T, R, true_visit_counts))
    test_reward, test_std = test(env, agent, episodes=100)
    test_rewards.append((test_reward, test_std))

    for epoch in tqdm(range(epochs)):
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            if np.random.uniform() < epsilon:
                action = env.sample_action()
            else:
                action = agent.move(state)
            prev_state = state
            state, reward, done, info = env.step(action)

            true_visit_counts[ (prev_state, *action, state)] += 1

            agent.memory.store(prev_state, action, state, reward)
            total_reward += reward
        total_train_rewards.append(total_reward.item())
        if epsilon > min_epsilon:
            epsilon = epsilon * (1 - epsilon_decay)
        if (epoch % improve_every) == (improve_every -1):
        # if (epoch % improve_every) == 0:
            # print("Improving...", 'epsilon:', epsilon)
            agent.policy.improve(agent.approximate_T(), agent.approximate_R())
            model_checks.append(check_models(agent, T, R, true_visit_counts))
            test_reward, test_std = test(env, agent, episodes=100)
            test_rewards.append((test_reward, test_std))


    return np.array(total_train_rewards), np.array(test_rewards), np.array(model_checks)

def test(env, agent, episodes=1000):
    total_rewards = []
    for episode in range(episodes):
        total_reward = []
        done = False
        state = env.reset()
        total_reward = 0
        while not done:
            action = agent.move(state)
            state, reward, done, _ = env.step(action)
            total_reward += reward
        total_rewards.append(total_reward.item())
    return np.mean(total_rewards), np.std(total_rewards)


def check_models(agent, T, R, true_visit_counts):
    agent_T = agent.approximate_T()
    naive_T = torch.nan_to_num(
        true_visit_counts / torch.sum(true_visit_counts, axis=tuple(range(-len(agent.state_shape), 0)))[ [...] + [None] * len(agent.policy.state_shape)], 1 / math.prod(agent.state_shape)
    )

    agent_R = agent.approximate_R()
    visited = torch.sum(true_visit_counts, axis=tuple(range(-len(agent.policy.state_shape), 0))) > 0

    # print(torch.mean((T - naive_T)**2).item())

    return torch.mean((agent_T - T )**2).item(), torch.mean((T - naive_T)**2).item(), torch.sum( (agent_R - R)**2).item(), torch.sum( (R * visited - R)**2 ).item(), torch.sum(visited).item()
    

if __name__ == "__main__":    
    if len(sys.argv) > 1:
        j = sys.argv[1]
        print("j = ", j)
    else:
        j = "_"

    run_experiment = True
    plot = False

    if run_experiment:
        T, R, state_shape, actions_shape = generate_env_variables(j)

        env = SyntheticEnv(T, R)
        optimal_policy = Policy(state_shape, actions_shape, gamma=0.99)
        optimal_policy.improve(T, R, n_iter=50)

        optimal_Q = optimal_policy.compute_Q_tensor(T, R)
        optimal_action_value = [torch.max(values).item() for values in optimal_Q]

        optimal_V= optimal_policy.compute_V_tensor(optimal_Q)
        optimal_agent = Agent(state_shape, actions_shape)
        optimal_agent.policy = optimal_policy

        print("Optimal_agent:")
        optimal_mean, optimal_std = test(env, optimal_agent)
        print(optimal_mean)

        agent_settings = {
            'rank_5_decomp': (Agent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99)),
            'rank_3_decomp': (Agent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99, reward_decomp_rank=3, transition_decomp_rank=3)),
            'rank_10_decomp': (Agent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99, reward_decomp_rank=10, transition_decomp_rank=10)),
            'no_decomp': (Agent, dict(state_shape=state_shape, actions_shape=actions_shape, 
                use_decompositions=False, gamma=0.99)),
            'transition_completion': (TransitionCompletionAgent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99)),
            'tesseract_rank_5': (TesseractAgent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99)),
            'tesseract_rank_1': (TesseractAgent, dict(state_shape=state_shape, actions_shape=actions_shape, gamma=0.99, reward_decomp_rank=1, transition_decomp_rank=1))
        }

        agents = {
            agent: agent_settings[agent][0](**agent_settings[agent][1]) for agent in agent_settings.keys()
        }

        train_settings = dict(
            epochs=200, improve_every=5, test_every=5, T=T, R=R, 
            epsilon=0.9, epsilon_decay=1e-3, min_epsilon=0.1
        )

        results = {}
        for name in agents.keys():
            results[name] = train(env, agents[name], **train_settings)

        data = {
            "results": results,
            "agent_settings": agent_settings,
            "train_settings": train_settings,
            "T": T,
            "R": R,
            "optimal": [optimal_mean, optimal_std]
        }


        try:
            with open(f'new_synthetic_experiment_results_{j}.pkl', 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print('pickling failed:', e)
    else:
        with open(f'new_synthetic_experiment_results_{j}.pkl', 'rb') as f:
            data = pickle.load(f)
    if plot:
        results = data["results"]
        optimal_mean, optimal_std = data["optimal"]

        # plot 1: rewards
        plt.figure()
        for agent in results.keys():
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            test_rewards = results[agent][1]
            mean = test_rewards[:, 0]
            std = test_rewards[:, 1]
            plt.plot(epochs, mean, label=agent)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.5)
        # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
        plt.axhline(optimal_mean, color='black', linestyle=':', label="optimal")
        plt.xlabel("episodes")
        plt.ylabel("total reward")
        plt.legend()

        #plot 2: transition tensors
        plt.figure()
        for agent in results.keys():
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            transition_mse = results[agent][2][:, 0]
            plt.semilogy(epochs, transition_mse, label=agent)
        # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
        plt.xlabel("episodes")
        plt.ylabel("transition MSE")
        plt.legend()

        #plot 2: transition tensors
        plt.figure()
        for agent in results.keys():
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            reward_sse = results[agent][2][:, 2]
            plt.semilogy(epochs, reward_sse, label=agent)
        # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
        plt.xlabel("episodes")
        plt.ylabel("reward SSE")
        plt.legend()

        #plot 2: visited_states
        plt.figure()
        for agent in results.keys():
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            transition_mse = results[agent][2][:, -1]
            plt.semilogy(epochs, transition_mse, label=agent)
        # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
        plt.xlabel("episodes")
        plt.ylabel("reward SSE")
        plt.legend()
        
        plt.show()

