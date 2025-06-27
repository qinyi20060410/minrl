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
        init_state=10,
        move_score=0,
        hit_wall_score=-1,
        action_space=5,
    ) -> None:
        self.move_score = move_score
        self.hit_wall_score = hit_wall_score
        self.terminal_score = terminal_score
        self.forbidden_score = forbidden_score
        self.action_space = action_space
        self.map_description = None
        self.terminal = 0

        self.action_effects = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            Action.KEEP: (0, 0),
        }

        if desc is not None:
            self.map_description = desc
            self.rows = len(desc)
            self.cols = len(desc[0])
            self.init_state = [init_state // self.cols, init_state % self.cols]
            self.now_state = self.init_state
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
            self.init_state = [init_state // self.cols, init_state % self.cols]
            self.now_state = self.init_state
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

            desc = []
            for i in range(self.rows):
                s = ""
                for j in range(self.cols):
                    tmp = {
                        self.move_score: ".",
                        self.forbidden_score: "#",
                        self.terminal_score: "T",
                    }
                    s = s + tmp[self.reward[i][j]]
                desc.append(s)
            self.map_description = desc
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

    def get_observation_space(self):
        return 2

    def get_action_space(self):
        return self.action_space

    def get_map_description(self):
        return self.map_description

    def get_reward(self, nowState, action):
        x = nowState // self.cols
        y = nowState % self.cols

        if x < 0 or y < 0 or x >= self.rows or y >= self.cols:
            print(f"coordinate error: ({x},{y})")
        if (action < 0 or action >= self.action_space):
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
                s = s + tmp[np.argmax(policy[i])]
            if self.reward[nowx][nowy] == self.forbidden_score:
                tmp = {0: "⏫️", 1: "⏩️", 2: "⏬", 3: "⏪", 4: "🔄"}
                s = s + tmp[np.argmax(policy[i])]
            if nowy == self.cols - 1:
                print(s)
                s = ""

    def get_traj(self, state, action, policy):
        res = []
        nextState = state
        nextAction = action

        for i in range(1001):
            state = nextState
            action = nextAction

            score, nextState = self.get_reward(state, action)
            nextAction = np.random.choice(
                range(self.action_space),
                size=1,
                replace=False,
                p=policy[nextState],
            )[0]

            terminal = 0
            nxtx, nxty = nextState // self.cols, nextState % self.cols
            if self.reward[nxtx][nxty] == self.terminal_score:
                terminal = 1

            res.append((state, action, score, nextState, nextAction, terminal))

            if terminal:
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

    def reset(self):
        self.now_state = self.init_state
        self.terminal = 0
        return self.now_state

    def step(self, action):
        reward, next_state = self.get_reward(
            self.now_state[0] * self.cols + self.now_state[1], action
        )

        nxtx, nxty = next_state // self.cols, next_state % self.cols
        next_state = [nxtx, nxty]
        self.now_state = [nxtx, nxty]
        if self.reward[nxtx][nxty] == self.terminal_score:
            self.terminal = 1

        return next_state, reward, self.terminal, 0
