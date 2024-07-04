import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import model_config as config


class PhiNet(nn.Module):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output):
        super().__init__()
        self.fc1 = nn.Linear(dim_of_input, config.phi_net['dim_of_layer0'])
        self.fc2 = nn.Linear(config.phi_net['dim_of_layer0'], config.phi_net['dim_of_layer1'])
        self.fc3 = nn.Linear(config.phi_net['dim_of_layer1'], config.phi_net['dim_of_layer2'])
        # One of the NN outputs is a constant bias term, which is append below
        self.fc4 = nn.Linear(config.phi_net['dim_of_layer2'], config.phi_net['dim_of_output'] - 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        if len(x.shape) == 1:
            # single input (batch size == 1)
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for training (assuming that x is at most 2D)
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)
        

class HNet(nn.Module):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output):
        super().__init__()
        self.fc1 = nn.Linear(dim_of_input, config.h_net['dim_of_layer0'])
        self.fc2 = nn.Linear(config.h_net['dim_of_layer0'], dim_of_output)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x