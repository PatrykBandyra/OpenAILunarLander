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
SEED = 42

# Q-learning parameters -----------------------------------------------------------------------------------------------
LEARNING_RATE = 0.001
BATCH_SIZE = 128
GAMMA = 0.99
MEMORY_SIZE = 100_000
EPS = 1.0
EPS_MIN = 0.01
EPS_DEC = 5e-6
N_GAMES = 500
