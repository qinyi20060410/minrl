import numpy as np
from grid_world import GridWorld


def value_iteration(gamma=0.99, theta=1e-10):
    env = GridWorld(
        env_size=(5, 5),
        start_state=(2, 2),
        target_state=(4, 4),
        forbidden_states=[(2, 1), (3, 3), (1, 3)],
        reward_target=5,
        reward_forbidden=-10,
        reward_step=-1,
        animation_interval=0.2,
        debug=False,
    )

    state_values = np.random.uniform(
        low=env.reward_forbidden, high=env.reward_target, size=env.num_states
    )
    policy = np.zeros((env.num_states, len(env.action_space)))

    iter_cnt = 0
    while True:
        iter_cnt += 1
        delta = 0
        for state_index in range(env.num_states):
            state = state_index % env.env_size[0], state_index // env.env_size[1]
            old_state_value = state_values[state_index]
            action_values = []
            for _, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(state, action)
                action_values.append(reward + gamma * state_values[next_state])

            max_idx = np.argmax(action_values)

            policy[state_index, max_idx] = 1
            policy[state_index, np.arange(len(env.action_space)) != max_idx] = 0

            state_values[state_index] = np.max(action_values)

            delta = max(delta, abs(old_state_value - state_values[state_index]))

        if delta < theta:
            break

    return policy, state_values