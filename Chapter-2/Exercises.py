from Agent import Agent
from Bandit import Bandit
import Run

if __name__ == "__main__":
    # # Exercise-2.5
    # agent_non_stationary = Agent(10, 0.01, 0.1)
    # agent_stationary = Agent(10, 0.01, 0)
    # environment = Bandit(10, 1, 0.01)
    # print(environment.expected_rewards)
    # Run.run(exercise="Exercise-2.5", environment=environment, runs = 2000, steps=10000, agent_non_stationary=agent_non_stationary, agent_stationary=agent_stationary)

    # #Exercise-2.6
    # agent_optimistic_greedy = Agent(10, 0, optimistic=5, alpha=0.1)
    # agent_non_optimistic_01= Agent(10, 0.1, alpha=0.1)
    # environment = Bandit(10, 1)
    # print(environment.optimal_action)
    # Run.run(exercise="Exercise-2.6", environment=environment, runs = 2000, steps=1000, agent_optimistic_greedy=agent_optimistic_greedy, agent_non_optimistic_01=agent_non_optimistic_01)

    # #Exercise-2.8
    # agent_ucb2 = Agent(10, ucb=2)
    # agent_01 = Agent(10, 0.1)
    # environment = Bandit(10, 1)
    # print(environment.optimal_action)
    # Run.run(exercise="Exercise-2.8", environment=environment, runs = 2000, steps=1000, agent_ucb2=agent_ucb2, agent_01=agent_01)

    #Exercise-2.8
    agent_gradientBandit_Baseline_01 = Agent(10, )
    agent_gradientBandit_Baseline_04 = Agent(10, )
    agent_gradientBandit_nBaseline_01 = Agent(10)
    agent_gradientBandit_nBaseline_04 = Agent(10)
    environment = Bandit(10, 1)
    print(environment.optimal_action)
    Run.run(exercise="Exercise-2.8", environment=environment, runs = 2000, steps=1000, agent_ucb2=agent_ucb2, agent_01=agent_01)