from typing import Generator, Optional, Tuple
from .gridworld_render import Renderer

import numpy as np


class GridWorld:
    def __init__(self):
        self.action_space = [0, 1, 2, 3]
        self.action_meaning: dict[int, str] = {
            0: "UP",
            1: "DOWN",
            2: "LEFT",
            3: "RIGHT",
        }
        self.reward_map: np.ndarray = np.array(
            [0, 0, 0, 1], [0, None, 0, -1], [0, 0, 0, 0]
        )
        self.goal_state = (0, 3)
        self.wall_state = (1, 1)
        self.start_state = (2, 0)
        self.agent_state = self.start_state

    @property
    def height(self) -> int:
        return len(self.reward_map)

    @property
    def width(self) -> int:
        return len(self.reward_map[0])

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    def action(self) -> list[int]:
        return self.action_space

    def state(self) -> Generator[tuple[int, int], None, None]:
        for h in range(self.height):
            for w in range(self.width):
                yield (h, w)

    def next_state(self, state: tuple[int, int], action: int) -> tuple[int, int]:
        action_move_map = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        move = action_move_map[action]
        new_h, new_w = state[0] + move[0], state[1] + move[1]
        next_state = (new_h, new_w)

        if new_w < 0 or new_w >= self.width or new_h < 0 or new_h >= self.height:
            next_state = state
        elif next_state == self.wall_state:
            next_state = state

        return next_state

    def reward(
        self, state: tuple[int, int], action: int, next_state: tuple[int, int]
    ) -> Optional[int]:
        return self.reward_map[next_state]

    def step(self, action: int) -> Tuple[Tuple[int, int], Optional[int], bool]:
        state = self.agent_state
        next_state = self.next_state(state, action)
        reward = self.reward(state, action, next_state)
        done = next_state == self.goal_state

        self.agent_state = next_state
        return next_state, reward, done

    def reset(self):
        self.agent_state = self.start_state
        return self.agent_state

    def render_v(self, v=None, policy=None, print_value=True):
        renderer = Renderer(self.reward_map, self.goal_state,
                                          self.wall_state)
        renderer.render_v(v, policy, print_value)
