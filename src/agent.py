import numpy as np
import torch
import json
from model import Model


class Agent:
    """
    Agent receives rewards for its actions taken based on observations received from environment and previous experience
    (represented in form of Q-table).
    """

    def __init__(self, gamma, eps, eps_min, eps_dec, lr, input_dims, batch_size, n_actions, mem_size):
        self.gamma = gamma  # Discount
        self.eps = eps  # Probability of not choosing the best action in current state
        self.eps_min = eps_min  # Minimal value of eps
        self.eps_dec = eps_dec  # Decrease step of eps
        self.lr = lr  # Learning rate
        self.action_space = [i for i in range(n_actions)]  # List of actions (0 - 3)
        self.mem_size = mem_size  # Size of memory that stores
        self.batch_size = batch_size  # Size of data batch that Model operates on
        self.mem_cntr = 0  # Keeps track of position of first available memory

        self.q_eval = Model(lr, input_dims=input_dims, n_actions=n_actions)  # Model representing Q-table
        # State memory stores observations received from environment
        self.state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        # New state memory stores new observations in order to calculate temporal difference
        self.new_state_memory = np.zeros((self.mem_size, input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)  # Memory to store actions taken by Agent
        # Reward memory stores rewards received from environment for specific action in given state
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool)  # Stores info if given state is a terminal one

    def store_transition(self, state, action, reward, state_, terminal):
        """
        Stores transition between learning steps in learning loop.
        """
        index = self.mem_cntr % self.mem_size  # If out of memory - start filling memory from the beginning
        self.state_memory[index] = state
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = terminal

        self.mem_cntr += 1

    def choose_action(self, observation):
        """
        Chooses next action of Agent given observation from environment.
        """
        # Choose action based on Q-table - exploitation
        if np.random.random() > self.eps:
            state = torch.tensor([observation]).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = torch.argmax(actions).item()

        # Choose action randomly - exploration
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        """
        Performs one learning step on a batch of data.
        """
        # If memory size filled with data is not larger or equal to the size of a batch - do not learn, get more data
        if self.mem_cntr < self.batch_size:
            return

        # If we have enough data - learn from batch of data
        self.q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)  # Get max index of filled memory

        # Batch of data is selected randomly
        batch = np.random.choice(max_mem, self.batch_size,
                                 replace=False)  # List of indices of memory to be part of data batch
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        # Retrieve data from different memories based on selected indices of a data batch
        state_batch = torch.tensor(self.state_memory[batch]).to(self.q_eval.device)
        new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.q_eval.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.q_eval.device)
        terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.q_eval.device)
        action_batch = self.action_memory[batch]

        # Fit forward
        q_eval = self.q_eval.forward(state_batch)[
            batch_index, action_batch]  # Get values of actions Agent has actually taken
        q_next = self.q_eval.forward(new_state_batch)  # Agent's estimate for next state
        q_next[terminal_batch] = 0.0

        # Updating Q-table with optimizer
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.q_eval.loss(q_target, q_eval).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()

        # Decrease exploration by given step with each learning loop - linear decrease
        self.eps = self.eps - self.eps_dec if self.eps > self.eps_min else self.eps_min

    def save_agent(self, filename_suffix=''):
        """
        Saves state of agent and model in files.
        """
        torch.save(self.q_eval.state_dict(), f'data/model{filename_suffix}.tar')
        np.save(f'data/state_memory{filename_suffix}.npy', self.state_memory)
        np.save(f'data/new_state_memory{filename_suffix}.npy', self.new_state_memory)
        np.save(f'data/action_memory{filename_suffix}.npy', self.action_memory)
        np.save(f'data/reward_memory{filename_suffix}.npy', self.reward_memory)
        np.save(f'data/terminal_memory{filename_suffix}.npy', self.terminal_memory)
        data = {'eps': self.eps, 'mem_cntr': self.mem_cntr}
        with open(f'data/agent_data{filename_suffix}.json', 'w') as f:
            json.dump(data, f)

    def load_agent(self, filename_suffix=''):
        """
        Loads state of agent and model from files.
        """
        try:
            self.q_eval.load_state_dict(torch.load(f'data/model{filename_suffix}.tar'))
            self.state_memory = np.load(f'data/state_memory{filename_suffix}.npy')
            self.new_state_memory = np.load(f'data/new_state_memory{filename_suffix}.npy')
            self.action_memory = np.load(f'data/action_memory{filename_suffix}.npy')
            self.reward_memory = np.load(f'data/reward_memory{filename_suffix}.npy')
            self.terminal_memory = np.load(f'data/terminal_memory{filename_suffix}.npy')
            with open(f'data/agent_data{filename_suffix}.json', 'r') as f:
                data = json.load(f)
                self.eps = data['eps']
                self.mem_cntr = data['mem_cntr']
        except Exception as e:
            print(e)
