from collections import deque
import gym
import random
import numpy as np
import time
import pickle

from collections import defaultdict


EPISODES =   20000
LEARNING_RATE = .1
DISCOUNT_FACTOR = .99
EPSILON = 1
EPSILON_DECAY = .999



def default_Q_value():
    return 0


if __name__ == "__main__":
    random.seed(1)
    np.random.seed(1)
    env = gym.envs.make("FrozenLake-v0")
    env.seed(1)
    env.action_space.np_random.seed(1)


    Q_table = defaultdict(default_Q_value) # starts with a pessimistic estimate of zero reward for each state.

    episode_reward_record = deque(maxlen=100)


    for i in range(EPISODES):
        episode_reward = 0

        # PERFORM SARSA LEARNING

        done = False
        state = env.reset()
        reward = 0
        previous_reward = 0

        while not done:
            if random.uniform(0, 1) < EPSILON:
                action = env.action_space.sample()
            else:
                prediction = np.array([Q_table[(state, i)] for i in range(env.action_space.n)])
                action = np.argmax(prediction)

            next_state, reward, done, info = env.step(action)
            episode_reward += reward

            sarsa_action = np.argmax(np.array([Q_table[(next_state, i)] for i in range(env.action_space.n)]))
            q_value = Q_table[state, action]
            next_value = Q_table[(next_state,
                                  np.argmax(np.array([Q_table[(next_state, i)] for i in range(env.action_space.n)])))]

            Q_table[state, action] = q_value + LEARNING_RATE * (previous_reward + DISCOUNT_FACTOR * next_value - q_value)
            state = next_state
            action = sarsa_action

            if done:
                q_value = Q_table[(state, action)]
                new_q_value1 = q_value + LEARNING_RATE * (reward - q_value)
                Q_table[state, action] = new_q_value1

            previous_reward = reward
        episode_reward_record.append(episode_reward)
        EPSILON = EPSILON * EPSILON_DECAY

        if i % 100 == 0 and i > 0:
            print("LAST 100 EPISODE AVERAGE REWARD: " + str(sum(list(episode_reward_record))/100))
            print("EPSILON: " + str(EPSILON) )


    ####DO NOT MODIFY######
    model_file = open('SARSA_Q_TABLE.pkl' ,'wb')
    pickle.dump([Q_table,EPSILON],model_file)
    #######################
