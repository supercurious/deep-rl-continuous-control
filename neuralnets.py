import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_range(layer):
    """Distribution range based on layer fan-in"""
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Neural net for policy (actor)"""

    def __init__(self, state_size, action_size, seed, 
                 fc1_units=400, fc2_units=300):
        """
        # Parameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fc1_units (int): units in 1st hidden layer
            fc2_units (int): units in 2nd hidden layer
        """
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (based on DDPG paper)"""
        self.fc1.weight.data.uniform_(*get_range(self.fc1))
        self.fc2.weight.data.uniform_(*get_range(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build policy network mapping states -> actions"""
        x = F.relu(self.bn1(self.fc1(self.bn0(state))))
        x = F.relu(self.bn2(self.fc2(x)))
        return torch.tanh(self.fc3(x))

class Critic(nn.Module):
    """
    Neural net for value function (critic)
    Action inputs introduced into 2nd hidden layer
    """

    def __init__(self, state_size, action_size, seed, 
                 fcs1_units=400, fc2_units=300):
        """
        # Parameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fcs1_units (int): units in 1st hidden layer
            fc2_units (int): units in 2nd hidden layer
        """
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (based on DDPG paper)"""
        self.fcs1.weight.data.uniform_(*get_range(self.fcs1))
        self.fc2.weight.data.uniform_(*get_range(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build value network mapping (state, action) -> Q-value"""
        xs = self.bn0(state)
        xs = F.relu(self.bn1(self.fcs1(xs)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class Critic_v2(nn.Module):
    """
    Neural net for value function (critic)
    Modified DDPG from TD3 paper
    Action inputs introduced into 1st hidden layer rather than 2nd
    """

    def __init__(self, state_size, action_size, seed, 
                 fcs1_units=400, fc2_units=300):
        """
        # Parameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fcs1_units (int): units in 1st hidden layer
            fc2_units (int): units in 2nd hidden layer
        """
        super(Critic_v2, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fc1 = nn.Linear(state_size+action_size, fcs1_units)
        self.fc2 = nn.Linear(fcs1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (based on DDPG paper)"""
        self.fc1.weight.data.uniform_(*get_range(self.fc1))
        self.fc2.weight.data.uniform_(*get_range(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build value network mapping (state, action) -> Q-value"""
        xs = self.bn0(state)
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class TwinCritics(nn.Module):
    """
    Neural net for value function (critic) 
    Twin critics for clipped double Q-learning (TD3)
    Action inputs introduced into 2nd hidden layer (like DDPG paper, not TD3)
    """

    def __init__(self, state_size, action_size, seed, 
                 fcs1_units=400, fc2_units=300):
        """
        # Parameters
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            fcs1_units (int): units in 1st hidden layer
            fc2_units (int): units in 2nd hidden layer
        """
        super(TwinCritics, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.bn0 = nn.BatchNorm1d(state_size)
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.bn1 = nn.BatchNorm1d(fcs1_units)
        self.fc2 = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3 = nn.Linear(fc2_units, 1)

        self.fcs1b = nn.Linear(state_size, fcs1_units)
        self.bn1b = nn.BatchNorm1d(fcs1_units)
        self.fc2b = nn.Linear(fcs1_units+action_size, fc2_units)
        self.fc3b = nn.Linear(fc2_units, 1)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters (based on DDPG paper)"""
        self.fcs1.weight.data.uniform_(*get_range(self.fcs1))
        self.fc2.weight.data.uniform_(*get_range(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

        self.fcs1b.weight.data.uniform_(*get_range(self.fcs1b))
        self.fc2b.weight.data.uniform_(*get_range(self.fc2b))
        self.fc3b.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Build value network mapping (state, action) -> Q-value"""
        inputs = self.bn0(state)
        # Critic 1
        xs = F.relu(self.bn1(self.fcs1(inputs)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        # Critic 2
        xsb = F.relu(self.bn1b(self.fcs1b(inputs)))
        xb = torch.cat((xsb, action), dim=1)
        xb = F.relu(self.fc2b(xb))
        return self.fc3(x), self.fc3b(xb)
        
    def single(self, state, action):
        """Build value network mapping (state, action) -> Q-value"""
        inputs = self.bn0(state)
        xs = F.relu(self.bn1(self.fcs1(inputs)))
        x = torch.cat((xs, action), dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)