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
import model_config
import data_manager

class Trainer:
    def __init__(self, config: model_config.ModelConfig) -> None:
        self.config = config
        self.phi_net = model.PhiNet(config.phi_net)
        self.h_net = model.HNet(config.h_net)
        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=config.trainer.learning_rate)
        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=config.trainer.learning_rate)
        # initialization
        self.optimizer_phi.zero_grad()
        self.h_net.zero_grad()
        self.loaderset_a = []
        self.loaderset_phi = []
        self.dataset = []
        self.loss_phi_trace = []
        self.loss_h_trace = []

    def get_optimal_a(self, phi: torch.Tensor, ground_truth):
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
            a = a / torch.norm(a, 'fro') * self.config.trainer.gamma
        return a

    @staticmethod
    def get_prediction(phi, a):
        return torch.mm(phi, a)
    
    def can_train_h_net(self) -> bool:
        """randomly insert h_net training"""
        can_do = np.random.rand() <= 1.0 / self.config.trainer.frequency_h
        return can_do

    def step_training(self, epoch: int) -> tuple[float, float]:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(self.config.num_of_conditions)
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
            loss_phi = self.criterion(prediction, batch_phi['output']) - self.config.trainer.alpha*loss_h
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
            if self.config.trainer.SN > 0:
                for param in self.phi_net.parameters():
                    M = param.detach().numpy()
                    if M.ndim > 1:
                        s = np.linalg.norm(M, 2)
                        if s > self.config.trainer.SN:
                            param.data = param / s * self.config.trainer.SN
            '''
            record loss trace
            '''
            loss_phi_sum += loss_phi.item()
            loss_h_sum += loss_h.item()
        return loss_phi_sum, loss_h_sum


    def train_model(self, data_files: list, is_back2back=False):
        if is_back2back:
            self.dataset = data_manager.prepare_back2back_datasets(data_files)
        else:
            self.dataset = data_manager.prepare_datasets(data_files)
        self.loaderset_phi, self.loaderset_a = data_manager.prepare_loadersets(self.dataset, self.config.trainer)
        self.loss_phi_trace = []
        self.loss_h_trace = []
        for epoch in range(self.config.trainer.num_epochs):
            loss_phi, loss_h = self.step_training(epoch)
            self.loss_phi_trace.append(loss_phi/self.config.num_of_conditions)
            self.loss_h_trace.append(loss_h/self.config.num_of_conditions)
            if epoch % 100 == 0:
                print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch + 1, self.loss_phi_trace[-1], self.loss_h_trace[-1]))
    
    def verify_model(self, test_data: list[str], is_back2back=False):
        self.phi_net.eval()
        with torch.no_grad():
            if is_back2back:
                phi_out = self.phi_net(torch.tensor(data_manager.prepare_back2back_datasets(test_data)[0].input))
            else:
                phi_out = self.phi_net(torch.tensor(data_manager.prepare_datasets(test_data)[0].input))
            groundtruth = torch.tensor(self.dataset[0].output)
            a = self.get_optimal_a(phi_out, groundtruth)
            prediction = self.get_prediction(phi_out, a)
            error = groundtruth - prediction
        self.plot_prediction_error(error, groundtruth, prediction)
        return error, groundtruth, prediction

    def plot_prediction_error(self, error, groundtruth, prediction):
        fig, axs = plt.subplots(3, 2)
        axs[0, 0].plot(groundtruth[:, 0])
        axs[1, 0].plot(groundtruth[:, 1])
        axs[2, 0].plot(groundtruth[:, 2])
        axs[0, 0].plot(prediction[:, 0])
        axs[1, 0].plot(prediction[:, 1])
        axs[2, 0].plot(prediction[:, 2])
        axs[0, 0].legend(["groundtruth", "prediction"]) 
        axs[0, 0].set_ylabel('f_disturb_x')
        axs[1, 0].set_ylabel('f_disturb_y')
        axs[2, 0].set_ylabel('f_disturb_z')

        axs[0, 1].plot(error[:, 0])
        axs[1, 1].plot(error[:, 1])
        axs[2, 1].plot(error[:, 2])

    def plot_loss(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_phi_trace)
        axs[0].set_ylabel('loss_phi_trace [N]')
        axs[1].plot(self.loss_h_trace)
        axs[1].set_ylabel('loss_h_trace')
        axs[1].set_xlabel('epoch')


if __name__ == "__main__":
    # data_list = ["test_air_drag_0.csv", "test_air_drag_1.csv"]
    # config = model_config.ModelConfig(len(data_list))
    # trainer = Trainer(config)
    # trainer.train_model(data_list)

    data_list = ['custom_random3_baseline_10wind.csv',
                'custom_random3_baseline_20wind.csv',
                'custom_random3_baseline_30wind.csv',
                'custom_random3_baseline_40wind.csv',
                'custom_random3_baseline_50wind.csv',
                'custom_random3_baseline_nowind.csv']
    config = model_config.ModelConfig(len(data_list))
    trainer = Trainer(config)
    trainer.train_model(data_list, True)

    trainer.plot_loss()
    trainer.verify_model(['custom_random3_baseline_10wind.csv'], True)
    plt.show()




    

