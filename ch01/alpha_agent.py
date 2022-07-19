import numpy as np


class AlphaAgent:
    def __init__(self, epsilon: float, alpha: float, actions: int = 10):
        self.epsilon = epsilon
        self.Qs = np.zeros(actions)
        self.alpha = alpha

    def update(self, action: int, reward: int) -> None:
        self.Qs[action] += (reward - self.Qs[action]) * self.alpha

    def get_action(self) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return int(np.argmax(self.Qs))
