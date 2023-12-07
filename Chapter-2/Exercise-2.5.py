from Agent import Agent
from Bandit import Bandit
import matplotlib.pyplot as plt

def episode(agent, environment, steps=2000):
    """Runs the agent for a single episode which is n steps, and return the average reward during the episode

    Args:
        agent (Agent): The agent to learn and interact.
        environment (Bandit): The environment to interact with.
        steps (int): number of steps to run in this episode.
    """
    running_reward = 0
    for i in range(steps):
        running_reward += environment.makeMove(agent.chooseAction())
    return running_reward/steps

def run(environment, nEpisodes, **kwargs):
    """Run the agent for nRuns of episodes, and plot comparision against agent non-stationary vs agent stationary.

    Args:
        nEpisodes (int, optional): number of episodes. Defaults to 1e4.
    """ 
    nEpisodes = int(nEpisodes)
    for agentName, agent in kwargs.items():
        rewards = []
        environment.reset()
        for i in range(nEpisodes):
            rewards.append(episode(agent, environment))
        plt.plot(rewards, label=agentName)
    plt.xlabel("Episode Number")
    plt.ylabel("Average Return")
    plt.title("Average Return vs Episode")
    print("plot created")
    plt.legend()
    plt.show()
    return 

if __name__ == "__main__":
    agent_non_stationary = Agent(10, 0.01, 0.1)
    agent_stationary = Agent(10, 0.01, 0)
    environment = Bandit(10, 1, 0.01)
    run(environment=environment, nEpisodes = 1e4, agent_non_stationary=agent_non_stationary, agent_stationary=agent_stationary)

