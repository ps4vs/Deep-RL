from Agent import Agent
from Bandit import Bandit
from Run import run

if __name__ == "__main__":
    agent_optimistic_greedy = Agent(10, 0, optimistic=5, alpha=0.1)
    agent_non_optimistic_01= Agent(10, 0.1, alpha=0.1)
    environment = Bandit(10, 1)
    print(environment.optimal_action)
    run(environment=environment, runs = 2000, steps=1000, agent_optimistic_greedy=agent_optimistic_greedy, agent_non_optimistic_01=agent_non_optimistic_01)