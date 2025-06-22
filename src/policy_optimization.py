from typing import Dict, List

import numpy as np

from env import GridWorld

from .policy_evaluation import PolicyEvaluator


class PolicyOptimizer(object):
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(self.env.get_state_space_size())
        self.policy = self.create_random_policy()
        self.policy_evaluator = PolicyEvaluator(self.env, self.gamma)

    def create_random_policy(self) -> Dict[int, List[float]]:
        """Create an initial random policy"""
        policy = {}
        n_actions = self.env.get_action_space_size()

        for state in range(self.env.get_state_space_size()):
            valid_actions = self.env.get_valid_actions(state)
            probs = [
                1.0 / len(valid_actions) if action in valid_actions else 0.0
                for action in range(n_actions)
            ]
            policy[state] = probs

        return policy

    def value_iter(self, theta:float=1e-6, max_iters:int=1000):
        self.state_values = np.zeros(self.env.get_state_space_size())

        for iter in range(max_iters):
            delta = 0

            for state in range(self.env.get_action_space_size()):
                if state in self.env.terminal_states:
                    self.state_values[state] = self.env.terminal_states[state]
                    continue

                old_value = self.state_values[state]
                pos = self.env.convert_state_to_pos(state)

                action_values = []
                for action in self.env.get_valid_actions(state):
                    self.env.current_pos = pos
                    next_state, reward, done, _ = self.env.step(action)
                    action_values.append(reward + self.gamma * self.state_values[next_state])
                self.state_values[state] = max(action_values) if action_values else 0

                delta = max(delta, abs(old_value - self.state_values[state]))

            if delta < theta:
                print(f"Value iteration converged after {iter + 1} iterations")
                break

            policy = self.env.get_policy(self.gamma, self.state_values)
            return policy, self.state_values

    def policy_iter(self, theta:float=1e-6, max_iters:int=1000):
        policy_stabel = False
        iter = 0

        while not policy_stabel and iter < max_iters:
            self.state_values = np.zeros(self.env.get_state_space_size())

            policy_stabel = True
            for state in range(self.env.get_state_space_size()):
                if state in self.env.terminal_states:
                    continue

                old_action = np.argmax(self.policy[state])


