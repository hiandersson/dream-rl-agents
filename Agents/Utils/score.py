from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class Score():
    """Interacts with and learns from the environment."""

    def __init__(self, target_average, window_size, total_episodes, verbose=1, pbar=None):
        self.target_average = target_average
        self.window_size = window_size
        self.total_episodes = total_episodes
        self.verbose = verbose
        self.pbar = pbar
        self.reset()

    def reset(self):
        self.scores = []                         
        self.scores_window = deque(maxlen=self.window_size) 
        self.target_average_reached = False
        if self.verbose == 1 and self.pbar is None:
            self.pbar = tqdm(total=self.total_episodes, desc = "Learning:{:03}".format(self.bar_position), ncols = 128)
        self.target_reached_in = "---"
        self.best_score = -np.inf

    def reset_episode(self):
        self.score = 0

    def add_reward(self, reward):
        self.score += reward

    def set_total_reward(self, total_reward):
        self.score = total_reward

    def post_episode(self, i_episode):
        self.scores_window.append(self.score)       
        self.scores.append(self.score)   
        average_score = np.mean(self.scores_window)

        average_score_str = "{:.4}".format(average_score).ljust(5,"0")
        latest_str = "{:.4}".format(self.score).ljust(5,"0")
        best_score_str = "{:.4}".format(self.best_score).ljust(5,"0")
        
        if self.verbose == 1:
            self.pbar.set_postfix(best=best_score_str,average=average_score_str,latest=latest_str, solved_episode=self.target_reached_in)
            self.pbar.update(1)

        if average_score > self.target_average and self.target_average_reached == False:
            self.target_average_reached = True
            self.target_reached_in = i_episode

        found_best_score = None

        if average_score > self.best_score:
            self.best_score = average_score
            found_best_score = average_score

        return found_best_score

    def visualize(self):

        def moving_average(a, n) :
            ret = np.cumsum(a, dtype=float)
            ret[n:] = ret[n:] - ret[:-n]
            return ret[n - 1:] / n

        fig = plt.figure()
        x = np.arange(len(self.scores))
        fig,ax = plt.subplots()

        data_line = ax.plot(x,self.scores, label='Rewards', color="steelblue")
        mean_line = ax.plot(x[self.window_size - 1:], moving_average(self.scores, self.window_size), label='Mean', linestyle='--', color="orange")

        legend = ax.legend(loc='upper right')

        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()