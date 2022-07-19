from typing import List

import matplotlib.pyplot as plt
import numpy as np
from agent import Agent
from alpha_agent import AlphaAgent
from non_stationary_bandit import NonStatBandit

runs = 200
steps = 1000
epsilon = 0.1
alpha = 0.8
all_rates = np.zeros((runs, steps))
all_rates_alpha = np.zeros((runs, steps))

for run in range(runs):
    bandit = NonStatBandit(arms=10)
    agent = Agent(epsilon=epsilon, action_size=10)
    alpha_agent = AlphaAgent(epsilon=epsilon, alpha=alpha, actions=10)

    total_reward = 0
    total_reward_alpha = 0
    rates: List[float] = []
    rates_alpha: List[float] = []

    for step in range(steps):
        action = agent.get_action()
        action_alpha = alpha_agent.get_action()

        reward = bandit.play(action)
        reward_alpha = bandit.play(action_alpha)

        agent.update(action=action, reward=reward)
        alpha_agent.update(action=action_alpha, reward=reward_alpha)

        total_reward += reward
        total_reward_alpha += reward_alpha

        rates.append(total_reward / (step + 1))
        rates_alpha.append(total_reward_alpha / (step + 1))

    all_rates[run] = rates
    all_rates_alpha[run] = rates_alpha

avg_rates = np.average(all_rates, axis=0)
avg_rates_alpha = np.average(all_rates_alpha, axis=0)

plt.figure()
plt.ylabel("Rates")
plt.xlabel("Steps")
plt.plot(avg_rates, label="sample average")
plt.plot(avg_rates_alpha, label="alpha")
plt.legend()
plt.savefig("rates_non_stationary.png")
