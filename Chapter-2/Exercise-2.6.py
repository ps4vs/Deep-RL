from Agent import Agent
from Bandit import Bandit
import matplotlib.pyplot as plt
from tqdm import trange
def episode(agent, environment, steps=2000):
    """Runs the agent for a single episode which is n steps, and return the average reward during the episode

    Args:
        agent (Agent): The agent to learn and interact.
        environment (Bandit): The environment to interact with.
        steps (int): number of steps to run in this episode.
    """
    running_reward = 0
    optimal_action_count = 0
    for i in range(steps):
        action = agent.chooseAction()
        reward = environment.makeMove(action)
        if action==environment.optimal_action:
            optimal_action_count += 1
        agent.update(reward)
        running_reward += reward
        # if i < 10:
            # print(agent.estimates)
    return running_reward/steps, (optimal_action_count/steps)

def run(environment, nEpisodes, steps=2000, **kwargs):
    """Run the agent for nRuns of episodes, and plot comparision against agent non-stationary vs agent stationary.

    Args:
        nEpisodes (int, optional): number of episodes. Defaults to 1e4.
    """ 
    nEpisodes = int(nEpisodes)
    agent_rewards = {}
    agent_optimal_actions = {}
    for agentName, agent in kwargs.items():
        rewards = []
        optimal_actions = []
        environment.reset()
        for i in trange(nEpisodes):
            expected_return, optimal_action_percentage = episode(agent, environment, steps)
            rewards.append(expected_return)
            optimal_actions.append(optimal_action_percentage)
        agent_rewards[agentName] = rewards
        agent_optimal_actions[agentName] = optimal_actions
        # print(agent.estimates)

    fig, ax = plt.subplots()
    for agentName, agent in kwargs.items():
        ax.plot(agent_rewards[agentName], label=agentName)
    ax.set_xlabel("Episode Number")
    ax.set_ylabel("Average Return")
    ax.set_title("Average Return vs Episode")
    print("plot created")
    ax.legend()
    fig.savefig('Optimistic Initial Values ARvsE.png')
    fig.show()
    
    fig, ax = plt.subplots()
    for agentName, agent in kwargs.items():
        ax.plot(agent_optimal_actions[agentName], label=agentName)
    ax.set_xlabel('Episode Number')
    ax.set_ylabel('Optimal Action %')
    ax.set_title('Optimal action % vs Episode')
    ax.legend()
    fig.savefig('Optimistic Initial Values OAvsE')
    fig.show()

    return 

if __name__ == "__main__":
    agent_optimistic_greedy = Agent(10, 0, optimistic=5, alpha=0.1)
    agent_non_optimistic_01= Agent(10, 0.1, alpha=0.1)
    environment = Bandit(10, 1)
    print(environment.optimal_action)
    run(environment=environment, nEpisodes = 1000, steps=2000, agent_optimistic_greedy=agent_optimistic_greedy, agent_non_optimistic_01=agent_non_optimistic_01)