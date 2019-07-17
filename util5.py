## From: https://www.kaggle.com/angps95/intro-to-reinforcement-learning-with-openai-gym

import numpy as np
import warnings
import random
from IPython.display import clear_output
from collections import defaultdict
from matplotlib import animation
from IPython.display import display
import time
import gym
import util4 as u4



def count(policy, env, counter_max = 2000, verbose = False):
    curr_state = env.reset()
    counter = 0
    reward = None
    last_state = None
    print_counter = 0
    while reward != 20 and reward != 50 and counter < counter_max:
        state, reward, done, info = env.step(np.argmax(policy[curr_state]))
        last_state = curr_state
        curr_state = state
        counter += 1

        if last_state == state and print_counter < 5 and verbose:
            print("we have a repeat will robinson")
        if counter > 100 and print_counter < 5 and verbose:
            print(state, reward)
            taxi_row, taxi_col, pass_idx, dest_idx = env.decode(env.env.s)
            print(taxi_row, taxi_col, pass_idx, dest_idx)
            print('------------')
            print_counter += 1
    return counter

testing=True
if testing:
    env2 = gym.make("TaxiLarge-v0")

    q_pol_large = np.load('q_pol_large.npy') # = Q_learning_train(env3, alpha=0.8, gamma=0.9, epsilon=0.1, episodes=20001)

    #results_complex = np.load('results_complex.npy').item()
    policy_large, _, _ = q_pol_large #results_complex['output'][10]
    print(count(policy_large, env2, counter_max=2000))
    print("all done")




def policy_eval(policy, env, gamma=0.99, epsilon=0.001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.

    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        gamma: Gamma discount factor.

    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    V = np.zeros(env.env.nS)
    while True:
        delta = 0  # delta = change in value of state from one iteration to next

        for state in range(env.env.nS):  # for all states
            val = 0  # initiate value as 0
            for action, act_prob in enumerate(policy[state]):  # for all actions/action probabilities
                for prob, next_state, reward, done in env.env.P[state][action]:  # transition probabilities,state,rewards of each action
                    val += act_prob * prob * (reward + gamma * V[next_state])  # eqn to calculate
            delta = max(delta, np.abs(val - V[state]))
            V[state] = val

        if delta < epsilon:  # break if the change in value is less than the threshold (epsilon)
            break
    return np.array(V)


def policy_iteration(myenv, policy_eval_fn=policy_eval, gamma=0.99, epsilon=0.001):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.

    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        gamma: gamma discount factor.

    Returns:
        A tuple (policy, V).
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.

    """

    def one_step_lookahead(state, V, myenv = myenv):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(myenv.env.nA)
        for a in range(myenv.env.nA):
            for prob, next_state, reward, done in myenv.env.P[state][a]:
                A[a] += prob * (reward + gamma * V[next_state])
        return A

    # Start with a random policy
    policy = np.ones([myenv.env.nS, myenv.env.nA]) / myenv.env.nA

    counter = 0
    while True:
        curr_pol_val = policy_eval_fn(policy, myenv, gamma, epsilon)  # eval current policy
        policy_stable = True  # Check if policy did improve (Set it as True first)
        counter = 0
        for state in range(myenv.env.nS):  # for each states
            counter += 1
            chosen_act = np.argmax(policy[state])  # best action (Highest prob) under current policy
            act_values = one_step_lookahead(state, curr_pol_val, myenv)  # use one step lookahead to find action values
            best_act = np.argmax(act_values)  # find best action
            print(best_act)
            print(chosen_act)
            if chosen_act != best_act:
                #print('poke1 - count = ', counter)
                policy_stable = False  # Greedily find best action
            if chosen_act == 6:
                print('stop was chosen')
            if best_act == 6:
                print('best act is stop')
            policy[state] = np.eye(myenv.env.nA)[best_act]  # update
            if counter > 10000:
                print('whoa mama whats up')
                break
        if policy_stable:
            print('poke2 - count = ', counter)
            return policy, curr_pol_val



    return policy, np.zeros(env.env.nS)


def value_iteration(myenv, epsilon=0.001, gamma=0.99):
    """
    Value Iteration Algorithm.

    Args:
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a list of transition tuples (prob, next_state, reward, done).
            env.nS is a number of states in the environment.
            env.nA is a number of actions in the environment.
        epsilon: We stop evaluation once our value function change is less than theta for all states.
        gamma: Gamma discount factor.

    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.
    """

    def one_step_lookahead(state, V, myenv=myenv):
        """
        Helper function to calculate the value for all action in a given state.

        Args:
            state: The state to consider (int)
            V: The value to use as an estimator, Vector of length env.nS

        Returns:
            A vector of length env.nA containing the expected value of each action.
        """
        A = np.zeros(myenv.env.nA)
        for act in range(myenv.env.nA):
            for prob, next_state, reward, done in myenv.env.P[state][act]:
                A[act] += prob * (reward + gamma * V[next_state])
        return A

    V = np.zeros(myenv.env.nS)
    while True:
        delta = 0  # checker for improvements across states
        for state in range(myenv.env.nS):
            act_values = one_step_lookahead(state, V, myenv)  # lookahead one step
            best_act_value = np.max(act_values)  # get best action value
            delta = max(delta, np.abs(best_act_value - V[state]))  # find max delta across all states
            V[state] = best_act_value  # update value to best action value
        if delta < epsilon:  # if max improvement less than threshold
            break
    policy = np.zeros([myenv.env.nS, myenv.env.nA])
    for state in range(myenv.env.nS):  # for all states, create deterministic policy
        act_val = one_step_lookahead(state, V, myenv)
        best_action = np.argmax(act_val)
        policy[state][best_action] = 1

    return policy, V



testing2 = False
if testing2:
    env3 = gym.make("TaxiLargeComplex-v0")
    #value_iteration(env3, gamma=0.025, epsilon=0.001)
    #pitest = u4.policy_iteration(env3, policy_eval_fn=policy_eval, gamma=0.5, epsilon=0.1)
    pitest = u4.policy_iteration(env3, epsilon=1e-8, gamma=0.8, max_iter=10000, report=True)
    #pitest[1]
    np.save('pitest', pitest)
    print('can i make changes last minute? if so - yay!')
    print('done')


def view_policy(policy, env):
    curr_state = env.reset()
    counter = 0
    reward = None
    while reward != 20 and counter < 2000:
        state, reward, done, info = env.step(np.argmax(policy[0][curr_state]))
        curr_state = state
        counter += 1
        env.env.s = curr_state
        env.render()


def Q_learning_train(env, alpha, gamma, epsilon, episodes, verbose = True):
    """Q Learning Algorithm with epsilon greedy

    Args:
        env: Environment
        alpha: Learning Rate --> Extent to which our Q-values are being updated in every iteration.
        gamma: Discount Rate --> How much importance we want to give to future rewards
        epsilon: Probability of selecting random action instead of the 'optimal' action
        episodes: No. of episodes to train on

    Returns:
        Q-learning Trained policy

    """
    """Training the agent"""

    # For plotting metrics
    results = defaultdict(list)

    # Initialize Q table of 500 x 6 size (500 states and 6 actions) with all zeroes
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for i in range(1, episodes + 1):
        state = env.reset()

        epochs, penalties, reward, = 0, 0, 0
        t0 = time.time()
        done = False

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space randomly
            else:
                action = np.argmax(q_table[state])  # Exploit learned values by choosing optimal values

            next_state, reward, done, info = env.step(action)

            old_value = q_table[state, action]
            next_max = np.max(q_table[next_state])

            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state, action] = new_value

            if reward == -10:
                penalties += 1

            state = next_state
            epochs += 1

        results['epochs'].append(epochs)
        results['penalties'].append(penalties)
        results['comp_times'].append(time.time()-t0)
        results['episode_list'].append(i)

        if i % 10000 == 0 and verbose:
            clear_output(wait=True)
            print(f"Episode: {i}")
    # Start with a random policy
    policy = np.ones([env.env.nS, env.env.nA]) / env.env.nA

    for state in range(env.env.nS):  # for each states
        best_act = np.argmax(q_table[state])  # find best action
        policy[state] = np.eye(env.env.nA)[best_act]  # update

    if verbose:
        print("Training finished.\n")
    return policy, q_table, results


def view_policy_anim(policy, env):
    penalties, reward = 0, 0

    frames = [] # for animation

    done = False
    curr_state = env.reset()
    while not done:
        action = np.argmax(policy[0][curr_state])
        state, reward, done, info = env.step(action)
        curr_state = state
        if reward == -10:
            penalties += 1

        # Put each rendered frame into dict for animation
        frames.append({
            'frame': env.render(mode='ansi'),
            'state': state,
            'action': action,
            'reward': reward
            }
        )
    def print_frames(frames):
        for i, frame in enumerate(frames):
            clear_output(wait=True)
            print(frame['frame'].getvalue())
            print(f"Timestep: {i + 1}")
            print(f"State: {frame['state']}")
            print(f"Action: {frame['action']}")
            print(f"Reward: {frame['reward']}")
            time.sleep(.2)

    print_frames(frames)

def get_set_of_actions(just_policy, env, max_steps=2000):
    curr_state = env.reset()
    counter = 0
    reward = None

    path_list_row = []
    path_list_col = []
    reward_list = []

    while reward != 20 and reward != 50 and counter < max_steps:
        state, reward, done, info = env.step(np.argmax(just_policy[curr_state]))
        curr_state = state
        counter += 1
        env.env.s = curr_state
        taxi_row, taxi_col, pass_idx, dest_idx = env.decode(env.env.s)
        path_list_row.append(taxi_row)
        path_list_col.append(taxi_col)
        reward_list.append(reward)

    return path_list_row, path_list_col, reward_list
