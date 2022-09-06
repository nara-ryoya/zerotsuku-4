from typing import Tuple

import numpy as np


def greedy_probs(Q, state: Tuple[int, int], action_size: int = 4, epsilon: float = 0):
    qs = [Q[(state, action)] for action in range(action_size)]
    max_action = np.argmax(qs)

    base_probs = epsilon / action_size

    action_probs = {action: base_probs for action in range(action_size)}
    action_probs[max_action] += 1 - epsilon

    return action_probs
