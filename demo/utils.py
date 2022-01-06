# import numpy as np
# import torch
# import torch.nn as nn
# from torch.nn import functional as F
# import torch.optim as optim
# from config import *
#
#
# def init_actions():
#     """
#     Returns array of possible actions in 2 variants:
#     1. Normal -> [0 1 2 3]
#     2. One-hot-encoding -> [[1 0 0 0], ... ,[0 0 0 1]]
#     """
#     actions = np.arange(0, NUM_ENV_ACTIONS)
#     actions_one_hot = np.zeros((NUM_ENV_ACTIONS, NUM_ENV_ACTIONS))
#     actions_one_hot[np.arange(NUM_ENV_ACTIONS), actions] = 1
#     return actions, actions_one_hot
#
#
# class Model(nn.Module):
#     def __init__(self, num_inputs, num_outputs):
#         super().__init__()
#
#         self.linear1 = nn.Linear(num_inputs, 512)
#         self.linear2 = nn.Linear(512, 256)
#         self.linear3 = nn.Linear(256, 256)
#         self.linear4 = nn.Linear(256, num_outputs)
#
#     def forward(self, x):
#         x = F.relu(self.linear1(x))
#         x = F.relu(self.linear2(x))
#         x = F.relu(self.linear3(x))
#         x = self.linear4(x)
#         return x
#
#
# def get_optimizer(model):
#     return optim.Adam(model.parameters(), lr=LEARNING_RATE)
#
#
# def get_loss_fn():
#     return nn.MSELoss()
#
#
# def load_model_state(model, filename):
#     try:
#         model.load_state_dict(torch.load(filename))
#     except Exception as e:
#         print(f'Could not load model -> {e}')
#
#
# def predict_reward(model, actions_one_hot, q_state, action):
#     qs_a = np.concatenate((q_state, actions_one_hot[action]), axis=0)
#     pred_x = np.zeros(shape=(1, NUM_ENV_VARIABLES + NUM_ENV_ACTIONS))
#     pred_x[0] = qs_a
#
#     pred = model(pred_x[0].reshape(1, pred_x.shape[1]))
#     remembered_total_reward = pred[0][0]
#     return remembered_total_reward
#
#

import matplotlib.pyplot as plt
import numpy as np
import gym


def plotLearning(x, scores, epsilons, filename, lines=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, label="1")
    ax2 = fig.add_subplot(111, label="2", frame_on=False)

    ax.plot(x, epsilons, color="C0")
    ax.set_xlabel("Game", color="C0")
    ax.set_ylabel("Epsilon", color="C0")
    ax.tick_params(axis='x', colors="C0")
    ax.tick_params(axis='y', colors="C0")

    N = len(scores)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = np.mean(scores[max(0, t - 20):(t + 1)])

    ax2.scatter(x, running_avg, color="C1")
    # ax2.xaxis.tick_top()
    ax2.axes.get_xaxis().set_visible(False)
    ax2.yaxis.tick_right()
    # ax2.set_xlabel('x label 2', color="C1")
    ax2.set_ylabel('Score', color="C1")
    # ax2.xaxis.set_label_position('top')
    ax2.yaxis.set_label_position('right')
    # ax2.tick_params(axis='x', colors="C1")
    ax2.tick_params(axis='y', colors="C1")

    if lines is not None:
        for line in lines:
            plt.axvline(x=line)

    plt.savefig(filename)


class SkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        super(SkipEnv, self).__init__(env)
        self._skip = skip

    def step(self, action):
        t_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            t_reward += reward
            if done:
                break
        return obs, t_reward, done, info

    def reset(self):
        self._obs_buffer = []
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs


class PreProcessFrame(gym.ObservationWrapper):
    def __init__(self, env=None):
        super(PreProcessFrame, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(80, 80, 1), dtype=np.uint8)

    def observation(self, obs):
        return PreProcessFrame.process(obs)

    @staticmethod
    def process(frame):
        new_frame = np.reshape(frame, frame.shape).astype(np.float32)

        new_frame = 0.299 * new_frame[:, :, 0] + 0.587 * new_frame[:, :, 1] + \
                    0.114 * new_frame[:, :, 2]

        new_frame = new_frame[35:195:2, ::2].reshape(80, 80, 1)

        return new_frame.astype(np.uint8)


class MoveImgChannel(gym.ObservationWrapper):
    def __init__(self, env):
        super(MoveImgChannel, self).__init__(env)
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                                shape=(self.observation_space.shape[-1],
                                                       self.observation_space.shape[0],
                                                       self.observation_space.shape[1]),
                                                dtype=np.float32)

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)


class ScaleFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class BufferWrapper(gym.ObservationWrapper):
    def __init__(self, env, n_steps):
        super(BufferWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(n_steps, axis=0),
            env.observation_space.high.repeat(n_steps, axis=0),
            dtype=np.float32)

    def reset(self):
        self.buffer = np.zeros_like(self.observation_space.low, dtype=np.float32)
        return self.observation(self.env.reset())

    def observation(self, observation):
        self.buffer[:-1] = self.buffer[1:]
        self.buffer[-1] = observation
        return self.buffer


def make_env(env_name):
    env = gym.make(env_name)
    env = SkipEnv(env)
    env = PreProcessFrame(env)
    env = MoveImgChannel(env)
    env = BufferWrapper(env, 4)
    return ScaleFrame(env)
