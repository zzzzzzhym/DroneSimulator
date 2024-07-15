import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import model
import model_config as config
import data_manager

class Trainer:
    def __init__(self) -> None:
        self.phi_net = model.PhiNet(config.phi_net['dim_of_input'], config.phi_net['dim_of_output'])
        self.h_net = model.HNet(config.h_net['dim_of_input'], config.h_net['dim_of_output'])
        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=config.training['learning_rate'])
        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=config.training['learning_rate'])
        # initialization
        self.optimizer_phi.zero_grad()
        self.h_net.zero_grad()
        self.loaderset_a = []
        self.loaderset_phi = []

    @staticmethod
    def get_optimal_a(phi: torch.Tensor, ground_truth):
        """
        phi*a = labels
        a = inv(phi_t*phi)*phi_t*labels
        
        Args:
            phi (_type_): network output
            labels (_type_): ground truth

        Returns:
            _type_: _description_
        """
        phi_t = phi.transpose(0,1)
        phi_t_phi_inv = torch.inverse(torch.mm(phi_t, phi))
        a = torch.mm(phi_t_phi_inv, torch.mm(phi_t, ground_truth))
        if torch.norm(a, 'fro') < 0.00001:
            raise ValueError("a is too small")
        else:
            a = a / torch.norm(a, 'fro') * config.training['gamma']
        return a

    @staticmethod
    def get_prediction(phi, a):
        return torch.mm(phi, a)
    
    @staticmethod
    def can_train_h_net() -> bool:
        """randomly insert h_net training"""
        can_do = np.random.rand() <= 1.0 / config.training['frequency_h']
        return can_do

    def step_training(self, epoch: int) -> tuple[float, float]:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(config.num_of_conditions)
        np.random.shuffle(randomized_cases)
        
        loss_phi_sum = 0.0
        loss_h_sum = 0.0
        for case_num in randomized_cases:
            with torch.no_grad():
                batch_phi = next(iter(self.loaderset_phi[case_num]))
                batch_a = next(iter(self.loaderset_a[case_num]))
            self.optimizer_phi.zero_grad()   # reset gradient before each training section starts
            phi_output = self.phi_net(batch_phi['input'])
            a = self.get_optimal_a(phi_output, batch_phi['output'])
            prediction = self.get_prediction(phi_output, a)
            loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
            loss_phi = self.criterion(prediction, batch_phi['output']) - config.training['alpha']*loss_h
            loss_phi.backward()
            self.optimizer_phi.step()

            if self.can_train_h_net():
                self.optimizer_h.zero_grad() # remove h_net gradient gained from loss_pi 
                phi_output = self.phi_net(batch_phi['input']) # get output again after optimizer step
                loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
                loss_h.backward()
                self.optimizer_h.step()

            '''
            Spectral normalization
            '''
            if config.training['SN'] > 0:
                for param in self.phi_net.parameters():
                    M = param.detach().numpy()
                    if M.ndim > 1:
                        s = np.linalg.norm(M, 2)
                        if s > config.training['SN']:
                            param.data = param / s * config.training['SN']
            '''
            record loss trace
            '''
            loss_phi_sum += loss_phi.item()
            loss_h_sum += loss_h.item()
        return loss_phi_sum, loss_h_sum


    def train_model(self, data_files: list):
        self.loaderset_phi, self.loaderset_a = data_manager.prepare_loader_sets(data_files)
        loss_phi_trace_phi = []
        loss_h_trace = []
        # for epoch in range(config.training['num_epochs']):
        for epoch in range(1000):
            loss_phi, loss_h = self.step_training(epoch)
            loss_phi_trace_phi.append(loss_phi/config.num_of_conditions)
            loss_h_trace.append(loss_h/config.num_of_conditions)
            if epoch % 10 == 0:
                print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch + 1, loss_phi_trace_phi[-1], loss_h_trace[-1]))


if __name__ == "__main__":
    trainer = Trainer()
    trainer.train_model(["test_sample.csv"])




    

