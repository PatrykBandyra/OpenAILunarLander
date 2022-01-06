import numpy as np
import gym
from config import *
import utils


def main():
    actions, actions_one_hot = utils.init_actions()

    # Initialize environment
    env = gym.make('LunarLander-v2')
    env.reset()

    # Init training matrix with random states and actions
    data_x = np.random.random((5, NUM_ENV_VARIABLES + NUM_ENV_ACTIONS))

    # Output of total score
    data_y = np.random.random((5, 1))

    # Init Neural Network
    model = utils.Model(num_inputs=data_x.shape[1], num_outputs=data_y.shape[1])
    optimizer = utils.get_optimizer(model=model)
    loss_fn = utils.get_loss_fn()

    utils.load_model_state(model=model, filename=MODEL_STATE_FILENAME)

    # Initialize training data array
    total_steps = 0
    data_x = np.zeros(shape=(1, NUM_ENV_VARIABLES + NUM_ENV_ACTIONS))
    data_x = np.zeros(shape=(1, 1))

    # Initialize Memory Array data array
    memory_x = np.zeros(shape=(1, NUM_ENV_VARIABLES + NUM_ENV_ACTIONS))
    memory_x = np.zeros(shape=(1, 1))


if __name__ == '__main__':
    main()
