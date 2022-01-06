import gym
import numpy as np
import random
from config import *


"""
This file contains simulation of random approach to given task of landing a spacecraft between two flags.
Actions are taken randomly.
"""


if __name__ == '__main__':
    # Set seeds
    np.random.seed(SEED)
    random.seed(SEED)

    env = gym.make('LunarLander-v2')
    env.seed(SEED)

    scores = []

    for game in range(10):
        observation = env.reset()
        score = 0
        for t in range(1000):
            env.render()

            action = env.action_space.sample()

            observation, reward, done, info = env.step(action)

            score += reward

            if done:
                break

        scores.append(score)
        print(f'Game[{game}] - Score: {score}')

    print(f'Average score: {np.round(np.mean(scores), 2)}')

    env.close()
