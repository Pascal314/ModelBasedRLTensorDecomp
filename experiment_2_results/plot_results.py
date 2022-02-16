import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import sys
sys.path.append('..')
import agent


if __name__ == "__main__":

    exclude_agents = ['transition_completion']
    
    files = []
    for j in range(20):
        if f'degenerate_states_experiment_results_{j}.pkl' in os.listdir():
            with open(f'degenerate_states_experiment_results_{j}.pkl', 'rb') as f:
                files.append(pickle.load(f))

    data = files[0]

    # plot 1: rewards
    plt.figure()
    for agent in data["results"].keys():
        if agent not in exclude_agents:
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            test_rewards = np.array([data["results"][agent][1][:, 0] for data in files])

            print(test_rewards.shape)
            mean = np.mean(test_rewards.T - np.array([data["optimal"][0] for data in files]), axis=1)
            std = np.std(test_rewards, axis=0)
            plt.plot(epochs, mean, label=agent)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.5)
    #
    # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
    plt.axhline(0, color='black', linestyle=':', label="optimal")
    plt.xlabel("episodes")
    plt.ylabel("total reward")
    plt.grid()
    plt.legend()
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig('rewards.pdf')

    #plot 2: transition tensors
    plt.figure()
    for agent in data["results"].keys():
        if agent not in exclude_agents:
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            transition_mse = np.array([data["results"][agent][2][:, 0] for data in files])
            mean = np.mean(transition_mse, axis=0)
            std = np.std(transition_mse, axis=0) 
            plt.plot(epochs, mean, label=agent)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.5)
    # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
    plt.xlabel("episodes")
    plt.ylabel("transition MSE")
    plt.grid()
    plt.legend()
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig('transition.pdf')

    plt.axis(ymax=0.002, ymin=0)
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig('transition_zoomed.pdf')

    #plot 3: reward tensors
    plt.figure()
    for agent in data["results"].keys():
        if agent not in exclude_agents:
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            reward_sse = np.log(np.array([data["results"][agent][2][:, 2] for data in files]))
            mean = np.mean(reward_sse, axis=0)
            std = np.std(reward_sse, axis=0) 
            plt.plot(epochs, mean, label=agent)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.5)
    # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
    plt.xlabel("episodes")
    plt.ylabel("log reward SSE")
    # plt.axis(ymax=10, ymin=-15)
    plt.grid()
    plt.legend()
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig("reward.pdf")
    #plot 4: visited_states
    plt.figure()
    for agent in data["results"].keys():
        if agent not in exclude_agents:
            epochs = range(0, data["train_settings"]["epochs"] + 1, data["train_settings"]["improve_every"])
            reward_sse = np.array([data["results"][agent][2][:, -1] for data in files])
            mean = np.mean(reward_sse, axis=0)
            std = np.std(reward_sse, axis=0) 
            plt.plot(epochs, mean, label=agent)
            plt.fill_between(epochs, mean-std, mean+std, alpha=0.5)
    # plt.fill_between(epochs, optimal_mean - optimal_std, optimal_mean + optimal_std, color='black', alpha=0.5)
    plt.xlabel("episodes")
    plt.ylabel("unique visited state-action pairs")
    plt.grid()
    plt.legend()
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig("visited.pdf")
    
    plt.figure()
    agent = "no_decomp"
    plt.plot(epochs, data["results"][agent][2][:, 0], color='C0')
    plt.ylabel('transition error', color='C0')
    plt.grid()
    plt.twinx()
    plt.ylabel('unique visited state-action pairs', color='C1')
    plt.plot(epochs, data["results"][agent][2][:, -1], color='C1')
    plt.gcf().set_size_inches(4, 3)
    plt.tight_layout()
    plt.savefig("visitedvsmse.pdf")
    
    plt.show()