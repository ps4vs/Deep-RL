from Agent import Agent
from Bandit import Bandit
import matplotlib.pyplot as plt
from tqdm import trange
import numpy as np

def run(exercise, environment, runs=2000, steps=1000, **kwargs):
    """Run the agent for runs, each run for steps number of times, and plot comparision against average optimal action over runs for each time-step vs time-step

    Args:
        nEpisodes (int, optional): number of episodes. Defaults to 1e4.
    """ 
    agent_rewards = {}
    agent_optimal_actions = {}
    for agentName, agent in kwargs.items():
        agent_rewards[agentName] = np.zeros(steps)
        agent_optimal_actions[agentName] = np.zeros(steps)
        for run in trange(runs):
            agent.reset()
            environment.reset()
            for step in range(steps):
                action = agent.chooseAction()
                reward = environment.makeMove(action)
                if action==environment.optimal_action:
                    agent_optimal_actions[agentName][step] += 1
                # print(agent_rewards[agentName])
                agent_rewards[agentName][step] += reward
                agent.update(reward)
    
    fig, ax = plt.subplots()
    for agentName, agent in kwargs.items():
        ax.plot(agent_rewards[agentName]/steps, label=agentName)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Average Return")
    ax.set_title("Average Return vs Episode")
    print("plot created")
    ax.legend()
    fig.savefig(f'./images/{exercise}-ARvsE.png')
    fig.show()
    
    fig, ax = plt.subplots()
    for agentName, agent in kwargs.items():
        ax.plot(agent_optimal_actions[agentName]/steps, label=agentName)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Optimal Action %')
    ax.set_title('Optimal action % vs Episode')
    ax.legend()
    fig.savefig(f'./images/{exercise}-OAvsE.png')
    fig.show()

    return 