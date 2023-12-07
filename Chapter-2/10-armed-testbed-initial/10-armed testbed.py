from agent import Agent
from bandit import kBandit
import matplotlib.pyplot as plt

def compareAgents(bandit, nEpisodes=2000, **Agents):
    rewards = {}
    plt.xscale("log")
    for agentName, agent in Agents.items():
        rewards[agentName] = [0]
        for i in range(nEpisodes):
            rewards[agentName].append(agent.learn(bandit))
        plt.plot(rewards[agentName], scalex=True, scaley=True, label=f"{agentName}")
    plt.ylabel("Average Reward Received")
    plt.xlabel("nEpisode")
    plt.title("Average Reward Received vs Episode")
    plt.legend()
    plt.show()
    return

if __name__ == '__main__':
    k = int(input("Enter K: number of arms "))
    nEpisodes = int(input("Enter nEpisodes: number of episodes "))
    agent_greedy = Agent(k, epsilon=0)
    agent_01 = Agent(k, epsilon=0.1)
    agent_001 = Agent(k, epsilon=0.01)

    # environment is not influenced by actions, so we are using shared environment.
    bandit = kBandit(k)
    compareAgents(bandit, nEpisodes=nEpisodes, agent_001=agent_001, agent_01=agent_01, agent_greedy=agent_greedy)
    
