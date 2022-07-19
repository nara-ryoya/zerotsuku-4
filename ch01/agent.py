import numpy as np

class Agent:
    def __init__(self, epsilon: float, action_size: int = 10) -> None:
        self.epsilon = epsilon
        self.Qs = np.zeros(action_size)
        self.ns = np.zeros(action_size)
    
    def update(self, action: int, reward: int) -> None:
        self.ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.ns[action]
    
    def get_action(self) -> int:
        """epsilon-greedy法に従い、Agentの行動を決定する

        Returns:
            int: 行動
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, len(self.Qs))
        return np.argmax(self.Qs)