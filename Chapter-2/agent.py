import numpy as np
import random
from Bandit import Bandit

class Agent():
    """It helps to create an agent for stationary, non-stationary environments, can handle exploration and exploitation tradeoff for a k-armed bandit problem.
    """
    def __init__(self, numActions=10, epsilon=0.01, alpha=0):
        """_summary_

        Args:
            numActions (int, optional): The number of actions available to take in the environment. Defaults to 10.
            epsilon (float, optional): The exploration probability. Defaults to 0.01.
            alpha (float, optional): Whether the distribution of the reward signals in the environment is stationary or non-stationary. Defaults to 0, ie, stationary distribution uses exponent, else constant step size.
        """
        # Q_n+1 = R_n+1 + stepsize (R_n-Qn)
        self.numActions = numActions
        self.epsilon = epsilon
        self.estimates = {i: 0 for i in range(1, numActions+1)}
        self.actionCount = {i: 0 for i in range(1, numActions+1)}
        self.previousAction = 0
        self.alpha = alpha
        return

    def chooseAction(self):
        """chooseAction according to the agent settings.

        Returns:
            int: action to take to maximize the expected reward.
        """
        if np.random.rand() < self.epsilon:
            # Exploration
            self.previousAction = np.random.randint(1, self.numActions+1)
            self.actionCount[self.previousAction]+=1
            return self.previousAction
        else:
            # Exploitation
            self.previousAction = max(self.estimates, key=lambda x: self.estimates[x])
            self.actionCount[self.previousAction]+=1
            return self.previousAction
    
    def update(self, reward):
        """Used to update the estimates

        Args:
            reward (float): The reward recieved for the action taken to update the estimates.
        """
        if self.alpha:
            self.estimates[self.previousAction] = self.estimates[self.previousAction] + (reward - self.estimates[self.previousAction])/self.alpha
        else:
            self.estimates[self.previousAction] = self.estimates[self.previousAction] + 1/self.actionCount[self.previousAction] (reward - self.estimates[self.previousAction])
        return
            

if __name__ == "__main__":
    agent = Agent(10, 0, 1)
    bandit = Bandit()
    for i in range(10):
        print(agent.chooseAction(), ":", agent.estimates)
        agent.update(1)

