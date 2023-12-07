import numpy as np

class kBandit():
    def __init__(self, k=10, std=1):
        """
        k is the number of arms of the bandit,
        std of each action's associated reward's stationary prob distribution
        """
        self.k = k
        self.std = std
        self.actions = [i for i in range(1, k+1)]
        self.mean_rewards = [np.random.normal(0, 1) for i in range(1, k+1)]
        print(f"mean_rewards is {self.mean_rewards}")
    def makeAction(self, action):
        if action == 0:
            return 0
        # print(action)
        assert(action <= self.k and action >= 0)
        return np.random.normal(self.mean_rewards[action-1], self.std)


    