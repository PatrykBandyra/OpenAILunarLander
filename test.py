import gym


if __name__ == '__main__':
    env = gym.make('LunarLander-v2')

    for i_episode in range(20):
        observation = env.reset()

        for t in range(100):
            env.render()
            # print(observation)

            action = env.action_space.sample()
            # 0 - do nothing
            # 1 - fire right engine
            # 2 - fire main engine
            # 3 - fire left engine

            observation, reward, done, info = env.step(action)
            print(reward)

            if done:
                print(f'Episode finished after {t + 1} timestamps ----------------------------------------------------')
                break

    env.close()
