import gym
import torch
import numpy as np
import random
from agent import Agent
from config import *


"""
This file contains simulation of our Agent behaviour in OpenAi environment (LunarLander-v2). 
Agent state is loaded from external files.

The main goal of our Agent is to land the spacecraft in the place between two flags and receive 
the highest possible score.
"""


if __name__ == '__main__':
    # Set seeds
    torch.manual_seed(SEED)  # Sets up seed for both devices
    np.random.seed(SEED)
    random.seed(SEED)

    env = gym.make('LunarLander-v2')
    env.seed(SEED)

    with torch.no_grad():

        agent = Agent(gamma=GAMMA, eps=0, eps_min=EPS_MIN, eps_dec=EPS_DEC, lr=LEARNING_RATE, batch_size=BATCH_SIZE,
                      n_actions=NUM_ENV_ACTIONS, input_dims=NUM_ENV_VARIABLES, mem_size=MEMORY_SIZE)
        agent.load_agent()
        agent.q_eval.eval()

        scores = []

        for game in range(10):
            observation = env.reset()
            score = 0
            for t in range(1000):
                env.render()

                action = agent.choose_action(observation)

                observation, reward, done, info = env.step(action)

                score += reward

                if done:
                    break

            scores.append(score)
            print(f'Game[{game}] - Score: {score}')

    print(f'Average score: {np.round(np.mean(scores), 2)}')

    env.close()
