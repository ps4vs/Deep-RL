# Agent needs to interact with the environment
# keep note of it's q estimate, which is the estimate of mean of the rewards
# take actions using this, and explore if necessary, and exploit
import random
import numpy as np

class Agent():
    def __init__(self, epsilon = 0.01, numActions=10):
        self.epsilon = epsilon
        self.numActions = numActions
        self.q_estimate = [0 for i in range(1, numActions+1)]
        self.previousAction = 0
        self.actionTakenCount = [0 for i in range(1, numActions+1)]

    def chooseAction(self, reward):
        if np.random.rand() < self.epsilon or reward==0:
            # exploration
            # print("Explore")
            return np.random.randint(1, self.numActions+1)
        else:
            # exploration
            if self.previousAction!=0:
                # updating the previousAction's q estimate using the reward received.
                self.q_estimate[self.previousAction-1] = (self.q_estimate[self.previousAction-1]*self.actionTakenCount[self.previousAction-1]+reward)/(self.actionTakenCount[self.previousAction-1]+1.0)
                self.actionTakenCount[self.previousAction-1] = self.actionTakenCount[self.previousAction-1]+1
            self.previousAction = self.q_estimate.index(max(self.q_estimate)) + 1
            # print("previous action is ", self.previousAction)
            return self.previousAction
        
    def learn(self, bandit, nplays=1000):
        reward = 0
        average_reward = 0
        for i in range(nplays):
            reward = bandit.makeAction((self.chooseAction(reward)))
            average_reward += reward
        return average_reward/nplays