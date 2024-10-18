import numpy as np
import matplotlib.pyplot as plt

# Function that calculates moving average of a series
def movingaverage(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

# Function that plots scores and epsilon against number of episodes
def plot_scores_epsilon(rewards_history, step_history, found_target_hit, epsilon_history, moving_avg_window=100):
    
    # team_rewards = [sum(x) for x in zip(*rewards_history)]
    
    f, axarr = plt.subplots(1,4, figsize=(10,3))
    # for i in range(len(rewards_history)):
        # axarr[0].plot(movingaverage(np.array(rewards_history), moving_avg_window), label=f'Agent {i+1} (MA)')
    # axarr[0].plot(movingaverage(np.array(team_rewards), moving_avg_window), label=f'Team rewards (MA)')

    axarr[0].plot(rewards_history, label='rewards')
    axarr[0].set_xlabel('Episodes')
    axarr[0].set_ylabel('Rewards')
    axarr[0].legend()

    axarr[1].plot(step_history, label='coverage_rate')
    axarr[1].set_xlabel('Episodes')
    axarr[1].set_ylabel('coverage_rate')
    axarr[1].legend()

    axarr[2].plot(found_target_hit, label='found_target')
    axarr[2].set_xlabel('Episodes')
    axarr[2].set_ylabel('found_target')
    axarr[2].legend()

    axarr[3].plot(epsilon_history, label='loss')
    axarr[3].set_xlabel('Episodes')
    axarr[3].set_ylabel('current_uncer')
    axarr[3].legend()
    plt.show()