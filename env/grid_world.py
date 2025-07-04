import random
from enum import IntEnum

import numpy as np


class Action(IntEnum):
    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    KEEP = 4


class GridWorld:
    def __init__(
        self,
        rows: int = 3,
        cols: int = 3,
        desc=None,
        forbidden_score=-1,
        terminal_score=1,
    ) -> None:
        self.action_effects = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.KEEP: (0, 0),
        }
        self.terminal_score = terminal_score
        self.forbidden_score = forbidden_score
        if desc is not None:
            self.rows = len(desc)
            self.cols = len(desc[0])
            grid = []
            for i in range(self.rows):
                tmp = []
                for j in range(self.cols):
                    tmp.append(
                        self.forbidden_score
                        if desc[i][j] == "#"
                        else self.terminal_score
                        if desc[i][j] == "T"
                        else 0
                    )
                grid.append(tmp)
            self.reward = np.array(grid)
            self.state_index = [
                [i * self.cols + j for j in range(self.cols)] for i in range(self.rows)
            ]
        else:
            state_index = [i for i in range(self.rows * self.cols)]
            random.shuffle(state_index)

            reward = [0 for i in range(self.rows * self.cols)]
            forbidden_area_nums = np.random.randint(1, 4)
            for i in range(forbidden_area_nums):
                reward[state_index[i]] = self.forbidden_score

            terminal_area_nums = 1
            for i in range(terminal_area_nums):
                reward[state_index[forbidden_area_nums + i]] = self.terminal_score

            self.reward = np.array(reward).reshape(rows, cols)
            self.state_index = [
                [i * self.cols + j for j in range(self.cols)] for i in range(self.rows)
            ]

    def render_grid(self):
        for i in range(self.rows):
            s = ""
            for j in range(self.cols):
                tmp = {0: "⬜️", self.forbidden_score: "🚫", self.terminal_score: "✅"}
                s = s + tmp[self.reward[i][j]]
            print(s)

    def get_reward(self, nowState, action):
        x = nowState // self.cols
        y = nowState % self.cols

        if x < 0 or y < 0 or x >= self.rows or y >= self.cols:
            print(f"coordinate error: ({x},{y})")
        if action not in Action:
            print(f"action error: ({action})")

        next_x = x + self.action_effects[action][0]
        next_y = y + self.action_effects[action][1]

        if next_x < 0 or next_y < 0 or next_x >= self.rows or next_y >= self.cols:
            return -1, nowState
        return self.reward[next_x][next_y], self.state_index[next_x][next_y]

    def render_policy(self, policy):
        # 用emoji表情，可视化策略，在平常的可通过区域就用普通箭头⬆️➡️⬇️⬅️
        # 但若是forbiddenArea，那就十万火急急急,于是变成了双箭头⏫︎⏩️⏬⏪
        s = ""
        for i in range(self.rows * self.cols):
            nowx = i // self.cols
            nowy = i % self.cols
            if self.reward[nowx][nowy] == self.terminal_score:
                s = s + "✅"
            if self.reward[nowx][nowy] == 0:
                tmp = {0: "⬆️", 1: "➡️", 2: "⬇️", 3: "⬅️", 4: "🔄"}
                if policy.ndim == 0:
                    s = s + tmp[policy[i]]
                else:
                    s = s + tmp[np.argmax(policy[i])]
            if self.reward[nowx][nowy] == self.forbidden_score:
                tmp = {0: "⏫️", 1: "⏩️", 2: "⏬", 3: "⏪", 4: "🔄"}
                if policy.ndim == 0:
                    s = s + tmp[policy[i]]
                else:
                    s = s + tmp[np.argmax(policy[i])]
            if nowy == self.cols - 1:
                print(s)
                s = ""

    def get_traj(self, state, action, policy, steps, stop_when_reach_target=False):
        res = []
        if stop_when_reach_target:
            steps = 20000

        for i in range(steps + 1):
            state = state
            action = action

            reward, next_state = self.get_reward(state, action)
            next_action = np.random.choice(
                range(5), size=1, replace=False, p=policy[next_state]
            )[0]
            res.append((state, action, reward, next_state, next_action))

            if stop_when_reach_target:
                x = state // self.cols
                y = state % self.cols
                if self.reward[x][y] == self.terminal_score:
                    return res

        return res

    def get_state_space_size(self) -> int:
        """Return the number of possible states"""
        return self.rows * self.cols

    def get_action_space_size(self) -> int:
        """Return the number of possible actions"""
        return len(Action)

    def convert_pos_to_state(self, pos) -> int:
        """Convert 2D position to state number"""
        return pos[0] * self.cols + pos[1]

    def convert_state_to_pos(self, state: int):
        """Convert state number to 2D position"""
        return state // self.cols, state % self.cols
