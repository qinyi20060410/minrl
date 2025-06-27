import numpy as np
from grid_world import GridWorld


def sarsa(start_state, gamma = 0.9, alpha=0.1, epsilon=0.1, episodes = 1000, iterations = 100):
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

    state_values = np.zeros(env.num_states)
    action_values = np.zeros((env.num_states, len(env.action_space)))
    policy = np.random.rand(env.num_states, len(env.action_space))
    policy /= policy.sum(axis=1, keepdims=True)

    lengths = []
    total_rewards = []

    for iter in range(iterations):
        state = start_state
        state_index = state[1] * env.env_size[0] + state[0]
        action_index= np.random.choice(len(env.action_space), p=policy[state_index])
        action = env.action_space[action_index]

        length = 0
        total_reward = 0
        while state != env.target_state:
            next_state, reward = env.get_next_state_reward(state, action)
            next_action = np.random.choice(len(env.action_space), p=policy[next_state])
