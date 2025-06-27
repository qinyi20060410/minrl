import numpy as np
from grid_world import GridWorld


def policy_iteration(gamma=0.99, theta=1e-10, epochs=10):
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
    policy = np.random.uniform(low=0, high=1, size=(env.num_states, len(env.action_space)))
    policy /= policy.sum(axis=1, keepdims=True)

    iter_cnt = 0
    while True:
        delta = 0
        iter_cnt += 1

        now_state_value = state_values.copy()

        for i in range(epochs):
            for state_index in range(env.num_states):
                state = state_index % env.env_size[0], state_index // env.env_size[0]
                action_values= []
                for _, action in enumerate(env.action_space):
                    next_state, reward = env.get_next_state_reward(state, action)
                    if state == env.target_state:
                        value_next_state = 0
                    else:
                        value_next_state = state_values[next_state]
                    action_values.append(reward + gamma * value_next_state)

                state_values[state_index] = np.dot(policy[state_index], action_values)


        for state_index in range(env.num_states):
            state = state_index % env.env_size[0], state_index // env.env_size[0]
            action_values = []

            for _, action in enumerate(env.action_space):
                next_state, reward = env.get_next_state_reward(state, action)
                if state == env.target_state:
                    value_next_state = 0
                else:
                    value_next_state = state_values[next_state]
                action_values.append(reward + gamma * value_next_state)

            max_idx = np.argmax(action_values)
            policy[state_index, max_idx] = 1
            policy[state_index, np.arange(len(env.action_space)) != max_idx] = 0

            if state == env.target_state:
                policy[state_index, -1] = 1
                policy[state_index, :-1] = 0

        delta = max(np.abs(state_values - now_state_value))

        if delta < theta:
            break

    return policy, state_values