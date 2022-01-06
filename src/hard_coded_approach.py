import gym
import numpy as np
import random
from config import *


"""
This file contains simulation of approach based on hard coded "optimal" behaviour in given state.
Example from official OpenAi GitHub repository.

Created by: Klimov
GitHub repo: https://github.com/openai/gym/blob/master/gym/envs/box2d/lunar_lander.py
"""


def heuristic(env, observation):

    angle_targ = observation[0] * 0.5 + observation[2] * 1.0  # angle should point towards center
    if angle_targ > 0.4:
        angle_targ = 0.4  # more than 0.4 radians (22 degrees) is bad
    if angle_targ < -0.4:
        angle_targ = -0.4
    hover_targ = 0.55 * np.abs(
        observation[0]
    )  # target y should be proportional to horizontal offset

    angle_todo = (angle_targ - observation[4]) * 0.5 - (observation[5]) * 1.0
    hover_todo = (hover_targ - observation[1]) * 0.5 - (observation[3]) * 0.5

    if observation[6] or observation[7]:  # legs have contact
        angle_todo = 0
        hover_todo = (
            -(observation[3]) * 0.5
        )  # override to reduce fall speed, that's all we need after contact

    if env.continuous:
        a = np.array([hover_todo * 20 - 1, -angle_todo * 20])
        a = np.clip(a, -1, +1)
    else:
        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > 0.05:
            a = 2
        elif angle_todo < -0.05:
            a = 3
        elif angle_todo > +0.05:
            a = 1
    return a


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

            action = heuristic(env, observation)

            observation, reward, done, info = env.step(action)

            score += reward

            if done:
                break

        scores.append(score)
        print(f'Game[{game}] - Score: {score}')

    print(f'Average score: {np.round(np.mean(scores), 2)}')

    env.close()
