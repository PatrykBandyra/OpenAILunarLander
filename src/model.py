import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class Model(nn.Module):
    """
    Neural network representing Q-table in Q-learning algorithm.
    """
    def __init__(self, lr, input_dims,  n_actions):
        super(Model, self).__init__()

        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions


class Model2(nn.Module):
    """
    Neural network representing Q-table in Q-learning algorithm.
    """
    def __init__(self, lr, input_dims,  n_actions):
        super(Model2, self).__init__()

        self.fc1 = nn.Linear(input_dims, 512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)

        self.fc3 = nn.Linear(1024, 1024)
        self.bn3 = nn.BatchNorm1d(1024)

        self.fc4 = nn.Linear(1024, 1024)
        self.bn4 = nn.BatchNorm1d(1024)

        self.fc5 = nn.Linear(1024, 512)
        self.bn5 = nn.BatchNorm1d(512)

        self.fc6 = nn.Linear(512, 256)
        self.bn6 = nn.BatchNorm1d(256)

        self.fc7 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.bn1(self.fc1(state)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))
        actions = self.fc7(x)

        return actions

class ModelDropout(nn.Module):
    """
    Neural network representing Q-table in Q-learning algorithm.
    """
    def __init__(self, lr, input_dims,  n_actions):
        super(ModelDropout, self).__init__()

        self.fc1 = nn.Linear(input_dims, 512)
        self.d1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(512, 512)
        self.d2 = nn.Dropout(0.2)

        self.fc3 = nn.Linear(512, 512)
        self.d3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(512, 256)
        self.d4 = nn.Dropout(0.2)

        self.fc5 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.d1(F.relu(self.fc1(state)))
        x = self.d2(F.relu(self.fc2(x)))
        x = self.d3(F.relu(self.fc3(x)))
        x = self.d4(F.relu(self.fc4(x)))
        actions = self.fc5(x)

        return actions

class ModelSmall(nn.Module):
    """
    Neural network representing Q-table in Q-learning algorithm.
    """
    def __init__(self, lr, input_dims,  n_actions):
        super(ModelSmall, self).__init__()

        self.fc1 = nn.Linear(input_dims, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 128)
        self.fc5 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return 
        
class ModelL1Loss(nn.Module):
    """
    Neural network representing Q-table in Q-learning algorithm.
    """
    def __init__(self, lr, input_dims,  n_actions):
        super(ModelL1Loss, self).__init__()

        self.fc1 = nn.Linear(input_dims, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 512)
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.L1Loss()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        actions = self.fc5(x)

        return actions