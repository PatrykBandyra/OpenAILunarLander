# Environment parameters ----------------------------------------------------------------------------------------------
NUM_ENV_VARIABLES = 8                 # Number of variables that environment provides us
"""
Observations (environment variables):
    0 - X position
    1 - Y position
    2 - X velocity
    3 - Y velocity
    4 - angle
    5 - angular velocity
    6 - right leg touching ground
    7 - left leg touching ground
"""
NUM_ENV_ACTIONS = 4                   # Number of actions of LunarLander
"""
Actions of LunarLander:
    0 - do nothing
    1 - fire right engine
    2 - fire main engine
    3 - fire left engine
"""
NUM_INITIAL_OBSERVATIONS = 30         # Number of observations before start of rendering

# Q-learning parameters -----------------------------------------------------------------------------------------------
LEARNING_RATE = 0.001                 # Learning rate of optimizer
DISCOUNT = 0.99
MAX_MEMORY_LEN = 100_000
EXPLORE_PROB = 0.05
EPOCHS = 5
NUM_GAMES_TO_PLAY = 1000

# Run parameters ------------------------------------------------------------------------------------------------------
MODEL_STATE_FILENAME = 'model.pt'
