import numpy as np
import matplotlib.pyplot as plt
import os
import yaml

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
    def __init__(self, data_menu: list, input_label_map_file: str, column_map_file: str) -> None:
        self.data_manager_instance = data_manager.DataManager(
            data_menu, input_label_map_file, column_map_file
        )
        self.loaderset_phi, self.loaderset_a = self.data_manager_instance.get_data()
        self.dim_of_input = len(self.data_manager_instance.input_columns)
        self.dim_of_label = len(self.data_manager_instance.label_columns)
        self.num_of_conditions = len(data_menu)
        self.model_factory_instance = model.ModelFactory(
            self.num_of_conditions,
            self.dim_of_input,
            self.data_manager_instance.input_mean_vector,
            self.data_manager_instance.input_scale_vector,
            self.data_manager_instance.label_mean_vector,
            self.data_manager_instance.label_scale_vector,
        )
        self.phi_net, self.h_net, self.dim_of_feature = self.model_factory_instance.generate_nets()
        self.config = Trainer.load_config("trainer_config.yaml")

        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=self.config["learning_rate"])
        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=self.config["learning_rate"])
        # initialization
        self.optimizer_phi.zero_grad()
        self.optimizer_h.zero_grad()
        self.loss_phi_trace = []
        self.loss_h_trace = []
        self.a_trace =[np.zeros((0, self.dim_of_feature, self.dim_of_label))] * (self.num_of_conditions)   # a list of 3D array, a_trace[condition_ID][iteration] = a where a is the a matrix

    @staticmethod
    def load_config(config_file: str):
        """Load model configuration from YAML file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(current_dir, config_file)
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config

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
            raise ValueError(f"a is too small, a = {a}")
        else:
            a = a / torch.norm(a, 'fro') * self.config["gamma"]
        return a

    @staticmethod
    def get_prediction(phi, a):
        return torch.mm(phi, a)
    
    def can_train_h_net(self) -> bool:
        """randomly insert h_net training"""
        can_do = np.random.rand() <= 1.0 / self.config["frequency_h"]
        return can_do

    def step_training(self, epoch: int) -> tuple[float, float]:
        # Randomize the order in which we train over the subdatasets
        randomized_cases = np.arange(self.num_of_conditions)
        np.random.shuffle(randomized_cases)
        
        loss_phi_sum = 0.0
        loss_h_sum = 0.0
        a_trace =[np.zeros((0, self.dim_of_feature, self.dim_of_label))] * (self.num_of_conditions)
        for case_num in randomized_cases:
            with torch.no_grad():
                batch_phi = next(iter(self.loaderset_phi[case_num]))
                batch_a = next(iter(self.loaderset_a[case_num]))
            self.optimizer_phi.zero_grad()   # reset gradient before each training section starts
            phi_output = self.phi_net(batch_phi['input'])
            if self.config["is_dynamic_environment"]:
                a = self.get_optimal_a(phi_output, batch_phi['output'])
            else:
                a = torch.ones((self.dim_of_feature, self.dim_of_label))
                a[self.dim_of_feature-1,:] = torch.zeros(self.dim_of_label)
            prediction = self.get_prediction(phi_output, a)
            loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
            loss_phi = 10000*self.criterion(prediction, batch_phi['output']) - self.config["alpha"]*loss_h
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
            if self.config["spectral_norm"] > 0:
                for param in self.phi_net.parameters():
                    M = param.detach().numpy()
                    if M.ndim > 1:
                        s = np.linalg.norm(M, 2)
                        if s > self.config["spectral_norm"]:
                            param.data = param / s * self.config["spectral_norm"]
            '''
            record loss trace
            '''
            loss_phi_sum += loss_phi.item()
            loss_h_sum += loss_h.item()
            a_np = np.copy(a.detach().numpy())
            a_trace[case_num] = np.concatenate((a_trace[case_num], np.array([a_np])), axis=0)
        return loss_phi_sum, loss_h_sum, a_trace


    def train_model(self):
        self.loss_phi_trace = []
        self.loss_h_trace = []
        for epoch in range(self.config["num_epochs"]):
            loss_phi, loss_h, a_trace = self.step_training(epoch)
            self.loss_phi_trace.append(loss_phi/self.num_of_conditions)
            self.loss_h_trace.append(loss_h/self.num_of_conditions)
            for case_num in range(len(a_trace)):
                self.a_trace[case_num] = np.concatenate((self.a_trace[case_num], a_trace[case_num]), axis=0)
            if epoch % 100 == 0:
                print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch + 1, self.loss_phi_trace[-1], self.loss_h_trace[-1]))
    
    def verify_model(self, test_data: list[str]):
        self.phi_net.eval()
        with torch.no_grad():
            dataset = self.data_manager_instance.prepare_datasets(test_data)
            phi_out = self.phi_net(torch.tensor(dataset[0].input))
            groundtruth = torch.tensor(dataset[0].output)
            a = self.get_optimal_a(phi_out, groundtruth)
            print(f"a = {a}")
            prediction = self.get_prediction(phi_out, a)
            error = groundtruth - prediction
        self.plot_prediction_error(error, groundtruth, prediction)
        self.plot_phi_out(phi_out)
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
        axs[0, 1].legend(["error"]) 

    def plot_phi_out(self, out):
        fig, axs = plt.subplots(3, 1)
        axs[0].plot(out[:, 0])
        axs[1].plot(out[:, 1])
        axs[2].plot(out[:, 2])
        axs[0].set_ylabel('phi_out_x')
        axs[1].set_ylabel('phi_out_y')
        axs[2].set_ylabel('phi_out_z')
        axs[2].set_xlabel('epoch')        
        
    def plot_loss(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_phi_trace)
        axs[0].set_ylabel('loss_phi_trace [N]')
        axs[1].plot(self.loss_h_trace)
        axs[1].set_ylabel('loss_h_trace')
        axs[1].set_xlabel('epoch')

    def plot_a(self):
        for i in range(len(self.a_trace)):
            _, row, col = self.a_trace[i].shape
            fig, axs = plt.subplots(row, col)
            for j in range(row):
                for k in range(col):
                    axs[j,k].plot(self.a_trace[i][:, j, k])

    def save_model(self, name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        file_path = os.path.join(current_dir, "model", name + ".pth")
        torch.save({"phi": self.phi_net.state_dict(),
                    "h": self.h_net.state_dict(),
                    "config": self.gather_model_config()}, 
                    file_path)
        
    def gather_model_config(self):
        """Extract model configuration from the model factory and data manager for later reconstruct the model in application.
        Model config contains how to recreate the model and how to use it"""
        config = {"phi_net_input_args": {},
                  "h_net_input_args": {},
                  "phi_net_io_fields": {}}
        config["phi_net_input_args"]["dim_of_input "] = self.dim_of_input
        config["phi_net_input_args"]["dim_of_output"] = self.dim_of_feature
        config["phi_net_input_args"]["dim_of_layers"] = self.model_factory_instance.config["PhiNet"]["hidden_layer_dimensions"]
        config["phi_net_input_args"]["input_mean"] = self.data_manager_instance.input_mean_vector
        config["phi_net_input_args"]["input_scale"] = self.data_manager_instance.input_scale_vector
        config["phi_net_input_args"]["output_mean"] = self.data_manager_instance.label_mean_vector
        config["phi_net_input_args"]["output_scale"] = self.data_manager_instance.label_scale_vector
        config["phi_net_io_fields"] = self.data_manager_instance.input_label_map
        
        config["h_net_input_args"]["dim_of_input"] = self.dim_of_feature
        config["h_net_input_args"]["dim_of_output"] = self.num_of_conditions
        config["h_net_input_args"]["dim_of_layers"] = self.model_factory_instance.config["HNet"]["hidden_layer_dimensions"]
        return config
    

def load_model(name) -> tuple[model.PhiNet, model.HNet, model_config.ModelConfig]:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "model", name + ".pth")       
    package = torch.load(file_path)
    config = package["config"]
    # generate an empty phi net
    phi = model.PhiNet(config["phi_net_input_args"]["dim_of_input "], 
                       config["phi_net_input_args"]["dim_of_output"], 
                       config["phi_net_input_args"]["dim_of_layers"], 
                       config["phi_net_input_args"]["input_mean"], 
                       config["phi_net_input_args"]["input_scale"], 
                       config["phi_net_input_args"]["output_mean"], 
                       config["phi_net_input_args"]["output_scale"])
    phi.load_state_dict(package["phi"])
    phi.eval()  # set to eval mode
    # similar for h net
    h = model.HNet(config["h_net_input_args"]["dim_of_input"],
                   config["h_net_input_args"]["dim_of_output"],
                   config["h_net_input_args"]["dim_of_layers"])
    h.load_state_dict(package["h"])
    h.eval()
    return phi, h, config
        

if __name__ == "__main__":
    is_back2back = False
    if not is_back2back:
        data_list = ["test_wall_effect_h_1_5r_dh_0.csv",
                    "test_wall_effect_h_2r_dh_0.csv",
                    "test_wall_effect_h_3r_dh_0.csv"]
                    # "test_wall_effect_h_4r_dh_0.csv"]
        config = model_config.ModelConfig(len(data_list), dim_of_feature=5)
        trainer = Trainer(config)
        trainer.train_model(data_list)
        trainer.verify_model(["test_wall_effect_h_1_8r_dh_0.csv"])
    else:
        data_list = ['custom_random3_baseline_10wind.csv',
                    'custom_random3_baseline_20wind.csv',
                    'custom_random3_baseline_30wind.csv',
                    'custom_random3_baseline_40wind.csv',
                    'custom_random3_baseline_50wind.csv',
                    'custom_random3_baseline_nowind.csv']
        config = model_config.ModelConfig(len(data_list))
        trainer = Trainer(config)
        trainer.train_model(data_list, True)
        trainer.verify_model(['custom_random3_baseline_10wind.csv'], True)
    trainer.plot_loss()
    trainer.plot_a()

    trainer.save_model("wall_effect")
    phi, h, config1 = load_model("wall_effect")
    trainer1 = Trainer(config1)
    trainer1.phi_net = phi
    trainer1.h_net = h
    trainer1.verify_model(['test_wall_effect_h_1_8r_dh_0.csv'])
    plt.show()    





    

