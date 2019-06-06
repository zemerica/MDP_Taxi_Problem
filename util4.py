# -*- coding: utf-8 -*-
import gym.envs.toy_text.frozen_lake as frozen_lake  # I guess `toy_text` is a separate package?
from gym import make
import gym
from gym.envs.registration import register, registry
import numpy as np
#import pandas as pd
#import my_env
#from pprint import pprint
#import warnings
#import xlrd

# See plug_and_chug() for iteration code

# These are the non-random maps
MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
    "16x16": [
        'SFFFFFHHFFFFFFFF',
        'FFFFHFFHFFFFFFHF',
        'FHFFFFFHFHFHFFFF',
        'HFFHFFFFFFFFFHFH',
        'FHFFFFHHFFFHHFHF',
        'FHFFHFHFFFFHFFFF',
        'FFHFFFFHFFFHHHHF',
        'HFHFFFFFFFFFFFFH',
        'FFFFFFFFFHFFFFHF',
        'FFFFFFHHFHFFFFFH',
        'FFHFFFFFFHFFFHFF',
        'FFFHHFHFFFFFHFFF',
        'FFFFFHFHFFFFHFFF',
        'HHHHHFFFHFFHFFFF',
        'FHFFHFFFFFFFFFFF',
        'FFHFHHFFHFFFFFFG'
    ]
}


###
# Generate a random 16x16 map and return the description
##

def sixteen_by_sixteen_map():
    return frozen_lake.generate_random_map(size=16)


# ##
# Utility Functions
# ##

def getStateReward(env):
    n_states = env.observation_space.n
    Rs = np.empty(n_states)
    Rs.fill(np.nan)
    p = env.P
    for state in p:
        for action_commanded in p[state]:
            for action_possible in p[state][action_commanded]:
                Rs[action_possible[1]] = action_possible[2]

    return Rs


def getReward(env):
    n_states, n_actions = env.observation_space.n, env.action_space.n

    R = np.zeros((n_states, n_actions))
    for s in range(n_states):
        for a, moves in env.P[s].items():
            for possible_move in moves:
                prob, _, r, _ = possible_move
                R[s, a] += r * prob

    return R


def getProb(env):
    n_states, n_actions = env.observation_space.n, env.action_space.n

    P = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            for moves in env.P[s][a]:
                prob, next_s, _, _ = moves
                P[s, a, next_s] += prob

    return P


def print_value(V, width=4, height=4):
    return np.around(np.resize(V, (width, height)), 4)


# let's plot the policy matrix (as in Part 1). according to
# https://github.com/openai/gym/blob/master/gym/envs/toy_text/frozen_lake.py
# LEFT = 0   DOWN = 1   RIGHT = 2  UP = 3
def print_policy(V, width=4, height=4):
    table = {0: "←", 1: "↓", 2: "→", 3: "↑"}
    policy = np.resize(V, (width, height))

    # transform using the dictionary
    return np.vectorize(table.get)(policy)


# https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
# matprint.py Pretty print a matrix in Python 3 with numpy
def matprint(mat, fmt="g"):
    if mat[0][0] in ["←", "↓", "→", "↑"]:
        fmt = 's'  # fmt='s' for arrows
    if mat[0][0] in [b'S', b'F', b'H', b'G']:
        fmt = 's'
        mat = mat.astype('<U8')
    col_maxes = [max([len(("{:" + fmt + "}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:" + str(col_maxes[i]) + fmt + "}").format(y), end="  ")
        print("")


# ##
# Policy Iteration Functions
# ##

# to evaluate the policy, as there is no max in the equation we can just solve
# the linear system
def policy_evaluation(pi, P, R, gamma, n_states):
    p = np.zeros((n_states, n_states))
    r = np.zeros((n_states, 1))

    for s in range(n_states):
        r[s] = R[s, pi[s]]
        p[s, :] = P[s, pi[s], :]

    # we take [:, 0] to return a vector because otherwise we have
    # a matrix of size (# states, 1)
    return np.linalg.inv((np.eye(n_states) - gamma * p)).dot(r)[:, 0]


def policy_iteration(env, epsilon=1e-8, gamma=0.8, max_iter=10000, report=False):
    n_states = env.observation_space.n

    # initialize arbitrary value function
    V = np.zeros(n_states)

    # initialize arbitrary policy
    pi = np.ones(n_states, dtype=int)

    R = getReward(env)
    P = getProb(env)

    i = 0

    while True and i < max_iter:
        V_prev = V.copy()

        # evaluate the policy
        V = policy_evaluation(pi, P, R, gamma, n_states)

        # policy improvement
        for s in range(n_states):
            pi[s] = np.argmax(R[s, :] + gamma * P[s, :, :].dot(V))

        print(np.linalg.norm(V_prev-V))
        if np.linalg.norm(V_prev - V) < epsilon:
            if report:
                print("Policy iteration converged after ", i + 1, "epochs")
            break

        i += 1

    return V, pi, i + 1


###
# Value Iteration Functions
###
def valueIteration(env, epsilon=1e-8, gamma=0.8, max_iter=10000, report=False):
    n_states = env.observation_space.n

    # initialize utilities to 0
    V = np.zeros(n_states)

    R = getReward(env)
    P = getProb(env)

    i = 0
    while True and i < max_iter:
        i += 1
        prev_V = V.copy()
        for s in range(n_states):
            V[s] = max(R[s, :] + gamma * P[s, :, :].dot(V))

        if np.linalg.norm(prev_V - V) <= epsilon:
            if report:
                print("Value iteration converged after ", i + 1, "epochs")
            break

    return V, i + 1


# transform value function into a policy
def value_to_policy(env, V, gamma=0.8):
    n_states, n_actions = env.observation_space.n, env.action_space.n

    policy = np.zeros(n_states, dtype=int)
    for state in range(n_states):
        best_action = 0
        best_reward = -float("inf")
        for action in range(n_actions):
            moves = env.P[state][action]  # [(prob, next_state, reward, terminate), ...]
            avg_reward = sum([prob * reward + gamma * V[next_state] for (prob, next_state, reward, _) in moves])

            if avg_reward > best_reward:
                best_reward = avg_reward
                best_action = action

        policy[state] = best_action

    return policy


# ##
# Q-Learning Functions
# ##

def epsilon_greedy(Q, s, qepsilon, rand_action):
    rand = np.random.uniform()
    if rand < qepsilon:
        # the sample() method from the environment allows
        # to randomly sample an action from the set of actions
        return rand_action
    else:
        # act greedily by selecting the best action possible in the current state
        return np.argmax(Q[s, :])


def Qlearning(env, qepsilon=0.1, lr=0.8, qgamma=0.95, episodes=10000, initial=0, decay=False, report=False):
    # initialize our Q-table: matrix of size [n_states, n_actions] with zeros
    n_states, n_actions = env.observation_space.n, env.action_space.n
    Q = np.ones((n_states, n_actions)) * initial
    Q_old = Q.copy()

    # get a single list of state descriptions: 'S', 'H', 'F', 'G'
    desc_states = [s.astype('<U8')[0] for row in env.desc for s in row]

    for episode in range(episodes):
        state = env.reset()
        terminate = False  # did the game end ?

        if decay:
            decay_e = 1
        while True:
            # choose an action using the epsilon greedy strategy
            if decay:
                qepsilon = 1.0 / decay_e
                decay_e += 1
            action = epsilon_greedy(Q, state, qepsilon, env.action_space.sample())

            # execute the action. The environment provides us
            # 4 values:
            # - the next_state we ended in after executing our action
            # - the reward we get from executing that action
            # - whether or not the game ended
            # - the probability of executing our action
            # (we don't use this information here)
            next_state, reward, terminate, _ = env.step(action)

            if desc_states[next_state] == 'H':  # if the agent falls in an hole
                r = reward  # then apply the state reward
                # the Q-value of the terminal state equals the reward
                Q[next_state] = np.ones(n_actions) * r

            elif desc_states[next_state] in ['S', 'F']:  # the agent is in a frozen tile
                r = reward  # give the agent the state reward

            elif desc_states[next_state] == 'G':  # the agent reach the goal state
                r = reward  # give him a big reward
                # the Q-value of the terminal state equals the reward
                Q[next_state] = np.ones(n_actions) * r

            # Q-learning update
            Q[state, action] = Q[state, action] + lr * (r + qgamma * np.max(Q[next_state, :]) - Q[state, action])

            # move the agent to the new state before executing the next iteration
            state = next_state

            # if we reach the goal state or fall in an hole
            # end the current episode
            if terminate:
                break
                # print(Q_old, '\n', Q, '\n', np.allclose(Q_old, Q))

        if np.allclose(Q_old, Q):
            if report:
                print("Q-Learning converged after ", episode + 1, "epochs")
            break
        Q_old = Q.copy()

    return Q, episode + 1


def Q_to_policy(Q):
    return np.argmax(Q, axis=1)


def Qlearning_trajectory(env, Q, max_steps=100, render=True, report=True):
    state = env.reset()  # reinitialize the environment
    i = 0
    while i < max_steps:
        # once the agent has been trained, it
        # will take the best action in each state
        action = np.argmax(Q[state, :])

        # execute the action and recover a tuple of values
        next_state, _, terminate, _ = env.step(action)
        if render:
            print("####################")
            env.render()  # display the new state of the game

        # move the agent to the new state before executing the next iteration
        state = next_state

        i += 1

        # if the agent falls in an hole or ends in the goal state
        if terminate:
            break  # break out of the loop

    if report:
        print('\n--Final position--')
        env.render()
        print(f"Agent ended up at state {state} after {i} steps")

    return state, i


# ##
# Generate a Customized Frozen Lake
# ##

def getEnv(env_id='default', rH=0, rG=1, rF=0, desc=None, map_name='4x4', is_slippery=True, render_initial=True):
    if env_id in gym.envs.registry.env_specs:
        del gym.envs.registry.env_specs[env_id]

    register(
        id=env_id,  # name given to this new environment
        entry_point='my_env:CustomizedFrozenLake',  # env entry point
        kwargs={'rH': rH, 'rG': rG, 'rF': rF,
                'desc': desc,
                'map_name': map_name,
                'is_slippery': is_slippery}  # argument passed to the env
    )

    this_env = make(env_id)

    if render_initial:
        print('--Board--')
        this_env.render()
        print('\n--Actions for Position to the Left of the Goal--')
        print(this_env.P[this_env.nS - 2])

    return this_env

# Random nuggets
# env.unwrapped.specstions/52774793/get-name-id-of-a-openai-gym-environment.id # https://stackoverflow.com/que
# https://github.com/openai/gym/issues/1172
