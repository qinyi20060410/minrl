import copy
from enum import IntEnum
from typing import List, Optional, Tuple

import numpy as np


class Action(IntEnum):
    """Enum class for possible actions in the grid world"""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3
    # KEEP = 4


class GridWorld:
    """
    A customizable Grid World environment that can be initialized with different sizes.
    The environment follows the standard gym-like interface.
    """

    def __init__(self, size: int = 3, random_seed: Optional[int] = None):
        """
        Initialize the Grid World environment.

        Args:
            size (int): Size of the grid (size × size)
            random_seed (int, optional): Seed for reproducibility
        """
        if size < 3:
            raise ValueError("Grid size must be at least 3x3")

        self.size = size
        self.n_states = size * size
        self.n_actions = len(Action)

        # Set random seed if provided
        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize grid
        self.current_pos = (0, 0)

        # Define terminal states (corners in this implementation)
        self.terminal_states = {
            0: 1.0,  # Top-left (positive reward)
            size - 1: -1.0,  # Top-right (negative reward)
            size * (size - 1): -1.0,  # Bottom-left (negative reward)
            size * size - 1: 1.0,  # Bottom-right (positive reward)
        }

        # Define action effects (row, col changes)
        self.action_effects = {
            Action.UP: (-1, 0),
            Action.RIGHT: (0, 1),
            Action.DOWN: (1, 0),
            Action.LEFT: (0, -1),
            # Action.KEEP: (0, 0),
        }

    def clone(self) -> "GridWorld":
        """
        Create a deep copy of the environment.

        Returns:
            GridWorld: A new instance with the same state
        """
        new_env = GridWorld(size=self.size)
        new_env.terminal_states = copy.deepcopy(self.terminal_states)
        new_env.current_pos = copy.deepcopy(self.current_pos)
        return new_env

    def convert_pos_to_state(self, pos: Tuple[int, int]) -> int:
        """Convert 2D position to state number"""
        return pos[0] * self.size + pos[1]

    def convert_state_to_pos(self, state: int) -> Tuple[int, int]:
        """Convert state number to 2D position"""
        return state // self.size, state % self.size

    def step(self, action: Action) -> Tuple[int, float, bool, dict]:
        """
        Take an action in the environment.

        Args:
            action (Action): The action to take

        Returns:
            Tuple containing:
                - next_state: The resulting state after the action
                - reward: The reward received
                - done: Whether the episode is finished
                - info: Additional information (empty dict)
        """
        if not isinstance(action, Action):
            action = Action(action)

        current_state = self.convert_pos_to_state(self.current_pos)

        # Check if current state is terminal
        if current_state in self.terminal_states:
            return current_state, 0.0, True, {}

        # Calculate new position
        row, col = self.current_pos
        d_row, d_col = self.action_effects[action]
        new_row = max(0, min(row + d_row, self.size - 1))
        new_col = max(0, min(col + d_col, self.size - 1))

        # Update position
        self.current_pos = (new_row, new_col)
        new_state = self.convert_pos_to_state(self.current_pos)

        # Determine reward and done flag
        reward = self.terminal_states.get(
            new_state, -0.1
        )  # Small negative reward for non-terminal states
        done = new_state in self.terminal_states

        return new_state, reward, done, {}

    def get_valid_actions(self, state: Optional[int] = None) -> List[Action]:
        """
        Get list of valid actions for the given state.

        Args:
            state (int, optional): State to check. If None, uses current state.

        Returns:
            List[Action]: List of valid actions
        """
        if state is None:
            pos = self.current_pos
        else:
            pos = self.convert_state_to_pos(state)

        valid_actions = []

        for action in Action:
            d_row, d_col = self.action_effects[action]
            new_row = pos[0] + d_row
            new_col = pos[1] + d_col

            if 0 <= new_row < self.size and 0 <= new_col < self.size:
                valid_actions.append(action)

        return valid_actions

    def render_grid(self):
        """
        Render the grid world as a string.

        Returns:
            str: String representation of the grid
        """
        grid = []
        for i in range(self.size):
            row = []
            for j in range(self.size):
                state = self.convert_pos_to_state((i, j))
                if (i, j) == self.current_pos:
                    row.append("A")  # Agent
                elif state in self.terminal_states:
                    row.append("T")  # Terminal
                else:
                    row.append(".")  # Empty
            grid.append(" ".join(row))
        print("\n".join(grid))

    def get_policy(self, gamma, state_values):
        policy = {}
        for state in range(self.get_state_space_size()):
            if state in self.terminal_states:
                policy[state] = 1.0 / pow(self.get_action_space_size(), 2)
                continue

            pos = self.convert_state_to_pos(state)
            valid_actions = self.get_valid_actions()
            action_values = []
            for action in valid_actions:
                self.current_pos = pos
                next_state, reward, done, _ = self.step(action)
                action_values.append(reward + (gamma * state_values[next_state]))

            best_action = valid_actions[np.argmax(action_values)]
            probs = [0.0] * self.get_action_space_size()
            probs[best_action] = 1.0
            policy[state] = probs
        return policy

    def render_policy(self, policy):
        action_symbols = ["↑", "→", "↓", "←"]

        for i in range(self.size):
            row = []
            for j in range(self.size):
                state = self.convert_pos_to_state((i, j))
                if state in self.terminal_states:
                    row.append("T")
                else:
                    action_idx = np.argmax(policy[state])
                    row.append(action_symbols[action_idx])
            print(" ".join(row))

    def get_state_space_size(self) -> int:
        """Return the number of possible states"""
        return self.n_states

    def get_action_space_size(self) -> int:
        """Return the number of possible actions"""
        return self.n_actions
