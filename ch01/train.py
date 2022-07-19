from typing import List

import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from bandit import Bandit

runs = 200
steps = 1000
epsilon = 0.1
all_rates = np.zeros((runs, steps))

for run in range(runs):
    bandit = Bandit(arms=10)
    agent = Agent(epsilon=epsilon, action_size=10)
    total_reward = 0
    rates: List[float] = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.update(action=action, reward=reward)
        total_reward += reward
        rates.append(total_reward / (step + 1))
    all_rates[run] = rates

avg_rates = np.average(all_rates, axis=0)

plt.figure()
plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates)
plt.savefig("rates.png")
