import gym
from agent import Agent
from config import *


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')
    env.seed(42)
    agent = Agent(gamma=GAMMA, eps=0, eps_min=EPS_MIN, eps_dec=EPS_DEC, lr=LEARNING_RATE, batch_size=BATCH_SIZE,
                  n_actions=NUM_ENV_ACTIONS, input_dims=NUM_ENV_VARIABLES, mem_size=MEMORY_SIZE)
    agent.load_agent()

    for game in range(10):
        observation = env.reset()

        for t in range(1000):
            env.render()

            action = agent.choose_action(observation)

            observation, reward, done, info = env.step(action)

            if done:
                break

    env.close()
