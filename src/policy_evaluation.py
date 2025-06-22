from typing import Dict, List

import numpy as np

from env import Action, GridWorld


class PolicyEvaluator(object):
    def __init__(self, env: GridWorld, gamma: float = 0.99):
        """
        Initialize the policy evaluator.

        Args:
            env (GridWorld): The GridWorld environment
            gamma (float): Discount factor for future rewards
        """
        self.env = env
        self.gamma = gamma
        self.state_values = np.zeros(env.get_state_space_size())

    def evaluate_policy(
        self,
        policy: Dict[int, List[float]],
        theta: float = 1e-6,
        max_iterations: int = 1000,
    ) -> np.ndarray:
        """
        Evaluate a given policy using the Bellman expectation equation.

        Args:
            policy (Dict[int, List[float]]): Dictionary mapping states to action probabilities
            theta (float): Convergence threshold
            max_iterations (int): Maximum number of iterations

        Returns:
            np.ndarray: The computed state values
        """
        # Initialize state values
        self.state_values = np.zeros(self.env.get_state_space_size())

        for iteration in range(max_iterations):
            delta = 0  # Track maximum change in value

            # Iterate through all states
            for state in range(self.env.get_state_space_size()):
                if state in self.env.terminal_states:
                    self.state_values[state] = self.env.terminal_states[state]
                    continue

                old_value = self.state_values[state]
                new_value = 0

                # Calculate expected value for the state based on policy
                state_pos = self.env.convert_state_to_pos(state)

                for action in self.env.get_valid_actions(state):
                    action_prob = policy[state][action]
                    if action_prob > 0:
                        # Simulate the action
                        self.env.current_pos = state_pos
                        next_state, reward, done, _ = self.env.step(action)

                        # Calculate value using Bellman expectation equation
                        new_value += action_prob * (
                            reward + self.gamma * self.state_values[next_state]
                        )

                # Update state value
                self.state_values[state] = new_value
                delta = max(delta, abs(old_value - new_value))

            print(f"iter {iteration} delate {delta} state value:")
            self.print_values()
            print("policy:")
            self.env.render_policy(self.env.get_policy(self.gamma, self.state_values))

            # Check for convergence
            if delta < theta:
                print(f"Policy evaluation converged after {iteration + 1} iterations")
                break

            if iteration == max_iterations - 1:
                print(
                    "Warning: Policy evaluation did not converge within the maximum iterations"
                )

        return self.state_values

    def print_values(self):
        """Print the state values in a grid format"""
        for i in range(self.env.size):
            row_values = []
            for j in range(self.env.size):
                state = self.env.convert_pos_to_state((i, j))
                row_values.append(f"{self.state_values[state]:6.6f}")
            print(" ".join(row_values))


if __name__ == "__main__":
    env = GridWorld(size=3)
    # random policy
    random_policy = {}
    n_actions = env.get_action_space_size()

    for state in range(env.get_state_space_size()):
        # For terminal states, action probabilities don't matter
        if state in env.terminal_states:
            random_policy[state] = [0.25] * n_actions
            continue

        valid_actions = env.get_valid_actions(state)
        probs = []

        # Create probability distribution over actions
        for action in range(n_actions):
            if action in valid_actions:
                probs.append(1.0 / len(valid_actions))
            else:
                probs.append(0.0)

        random_policy[state] = probs
    print(f"random policy:{random_policy}")

    evaluator = PolicyEvaluator(env, gamma=0.99)
    print("Evaluating random policy...")
    state_values = evaluator.evaluate_policy(random_policy, theta=1e-6)

    print("\nFinal state values:")
    evaluator.print_values()

    print("\nFinal policy:")
    evaluator.env.render_policy(evaluator.env.get_policy(evaluator.gamma, state_values))
    print("\nFinal grid:")
    evaluator.env.render_grid()

    deterministic_policy = {}
    for state in range(env.get_state_space_size()):
        probs = [0.0] * env.get_action_space_size()
        valid_actions = env.get_valid_actions(state)

        if Action.RIGHT in valid_actions:
            probs[Action.RIGHT] = 1.0
        elif Action.DOWN in valid_actions:
            probs[Action.DOWN] = 1.0
        elif valid_actions:
            probs[valid_actions[0]] = 1.0

        deterministic_policy[state] = probs

    evaluator = PolicyEvaluator(env, gamma=0.99)

    # Evaluate the policy
    print("\nEvaluating deterministic policy...")
    state_values = evaluator.evaluate_policy(deterministic_policy, theta=1e-6)

    print("\nFinal state values for deterministic policy:")
    evaluator.print_values()

    print("\nFinal policy:")
    evaluator.env.render_policy(evaluator.env.get_policy(evaluator.gamma, state_values))

    print("\nFinal grid:")
    evaluator.env.render_grid()
