from collections import defaultdict
from typing import Optional, Tuple

import numpy as np

from common.gridworld import GridWorld


class TdAgent:
    def __init__(self):
        self.gamma = 0.9
        self.alpha = 0.01
        self.action_size = 4

        random_actions = {0: 0.25, 1: 0.25, 2: 0.25, 3: 0.25}
        self.pi = defaultdict(lambda: random_actions)
        self.V = defaultdict(lambda: 0)

    def get_action(self, state: Tuple[int, int]):
        action_probs = self.pi[state]
        actions = list(action_probs.keys())
        probs = list(action_probs.values())
        return np.random.choice(actions, p=probs)

    def eval(
        self,
        state: Tuple[int, int],
        reward: Optional[int],
        next_state: Tuple[int, int],
        done: bool,
    ):
        next_V = 0 if done else self.V[next_state]
        target = reward + self.gamma * next_V

        self.V[state] += (target - self.V[state]) * self.alpha


episodes = 1000

env = GridWorld()
agent = TdAgent()

for episode in range(episodes):
    state = env.reset()

    while True:
        action = agent.get_action(state)
        next_state, reward, done = env.step(action=action)

        agent.eval(state=state, reward=reward, next_state=next_state, done=done)
        if done:
            break

        state = next_state

env.render_v(agent.V)
