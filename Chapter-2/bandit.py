import numpy as np
import random

class Bandit():
    def __init__(self, k=10, std_of_all_action_reward_distribution=1, std_random_walk=0):
        """k-armed bandit environment, with expected reward for each action is sampled from normal distribution (0, 1), and Reward during each action is sampled from stationary or non-stationary distribution with inital mean 0, and std deviation given.
        For non-stationary distribution randomwalks are taken, ie, mean <- mean + N(mean, std_random_walk)

        Args:
            k (int, optional): The number of arms for the bandit. Defaults to 10.
            std_of_all_action_reward_distribution (int, optional): standard deviation of all actions stationary reward distribution whose mean is sampled from normal distribution . Defaults to 1.
            std_random_walk (int, optional): For non-statinary distribution it is non-zero. Defaults to 0.
        """
        self.std_of_all_action_reward_distributions = std_of_all_action_reward_distribution
        self.expected_rewards = {i: np.random.normal(0, 1) for i in range(1, k+1)}
        self.std_random_walk = std_random_walk
        self.k = k
    
    def makeMove(self, action):
        reward = np.random.normal(self.expected_rewards[action], self.std_of_all_action_reward_distributions)
        # mean changes for non-stationary distribution alone
        self.expected_rewards[action] += np.random.normal(0, self.std_random_walk)
        return reward
    
    def reset(self):
        self.expected_rewards = {i: np.random.normal(0, 1) for i in range(1, self.k+1)}
        return
    
if __name__=="__main__":
    # code for testing
    bandit = Bandit(10, 1, 0.01)
    print(bandit.expected_rewards)
    bandit.makeMove(1)
    print(bandit.expected_rewards)
        
