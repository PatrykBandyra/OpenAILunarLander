import gym
import time
import numpy as np


if __name__ == '__main__':
    SEED = 3
    env = gym.make('LunarLander-v2')
    env.seed(SEED)  # Here we also need to set a seed for first iter to be the same as others

    min_angle = np.inf
    max_angle = -np.inf

    for i_episode in range(20):
        observation = env.reset()

        for t in range(1000):
            env.render()

            action = env.action_space.sample()
            # 0 - do nothing
            # 1 - fire right engine
            # 2 - fire main engine
            # 3 - fire left engine

            # observation, reward, done, info = env.step(action)
            observation, reward, done, info = env.step(1)
            env.seed(SEED)  # We need that in this loop to always have the same environment
            # Observation
            # 0 - X (-1 ... 1) checked
            # 1 - Y (-1 ... 1.41) checked (but never gets to -1 )
            # 2 - X speed (useful range is -1 .. +1, but spikes can be higher)
            # 3 - Y speed (useful range is -1 .. +1, but spikes can be higher)
            # 4 - angle (-4 ... 4) not sure (even more)
            # 5 - angular speed (very differs)
            # 6 - right leg (1 or 0) checked
            # 7 - left leg (1 or 0) checked

            if observation[5] < min_angle:
                min_angle = observation[5]
            elif observation[5] > max_angle:
                max_angle = observation[5]

            # print(observation[1])

            if done:
                print(f'Episode finished after {t + 1} timestamps ----------------------------------------------------')
                # print(f'Right leg: {observation[6]}, left leg: {observation[7]}')
                # print(f'Min angle: {min_angle}, Max angle: {max_angle}')
                print(f'Min ang speed: {min_angle}, Max ang seed: {max_angle}')
                # time.sleep(2)
                break

    env.close()
