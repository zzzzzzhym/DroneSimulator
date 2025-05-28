import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
from sklearn.manifold import TSNE
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataset import random_split

import model

class Trainer:
    def __init__(self,
                 phi_net: model.MultilayerNet,
                 h_net: model.MultilayerNet,
                 loaderset_phi: list[DataLoader],
                 loaderset_a: list[DataLoader],
                 dim_of_label: int
                 ) -> None:
        self.phi_net = phi_net
        self.h_net = h_net
        self.loaderset_phi = loaderset_phi
        self.loaderset_a = loaderset_a
        self.dim_of_label = dim_of_label
        self.num_of_conditions = self.h_net.dim_of_output   # should match the length of loaderset_phi and loaderset_a
        if self.num_of_conditions != len(loaderset_phi) or self.num_of_conditions != len(loaderset_a):
            raise ValueError("Number of conditions does not match the length of loaderset_phi and loaderset_a")
        self.config = Trainer.load_config("trainer_config.yaml")

        self.criterion = nn.MSELoss()
        self.criterion_h = nn.CrossEntropyLoss()
        self.optimizer_h = optim.Adam(self.h_net.parameters(), lr=self.config["learning_rate_h"])
        self.optimizer_phi = optim.Adam(self.phi_net.parameters(), lr=self.config["learning_rate_phi"])
        # initialization
        self.optimizer_phi.zero_grad()
        self.optimizer_h.zero_grad()
        self.loss_phi_trace = []
        self.loss_h_trace = []
        self.a_trace = [
            np.zeros((self.config["num_epochs"], self.phi_net.dim_of_output, self.dim_of_label))
            for _ in range(self.num_of_conditions)
        ]   # a list of 3D array, a_trace[condition_ID][iteration] = adapter matrix, a

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
        a_trace_per_step = [
            np.zeros((0, self.a_trace[0].shape[1], self.a_trace[0].shape[2]))
            for _ in range(len(self.a_trace))
        ]
        alpha = 0.0
        if epoch > self.config["warmup_epoch"]:
            alpha = self.config["alpha"]
        for case_num in randomized_cases:
            with torch.no_grad():
                batch_phi = next(iter(self.loaderset_phi[case_num]))
                batch_a = next(iter(self.loaderset_a[case_num]))
            self.optimizer_phi.zero_grad()   # reset gradient before each training section starts
            prediction_on_adapter_data = self.phi_net(batch_a['input'])
            if self.config["is_dynamic_environment"]:
                a = self.get_optimal_a(prediction_on_adapter_data, batch_a['output'])
            else:
                a = torch.ones(self.a_trace[case_num].shape[1], self.a_trace[case_num].shape[2])
                a[-1,:] = torch.zeros(self.a_trace[case_num].shape[2])
            phi_output = self.phi_net(batch_phi['input'])
            prediction = self.get_prediction(phi_output, a)
            loss_h = self.criterion_h(self.h_net(phi_output), batch_phi['c'])
            loss_phi = self.criterion(prediction, batch_phi['output']) - alpha*loss_h
            # trial code -> 
            # loss_mse = self.criterion(prediction, batch_phi['output'])
            # self.optimizer_phi.zero_grad()
            # loss_mse.backward(retain_graph=True)
            # mse_grad_norm = self.compute_grad_norm(self.phi_net)
            # self.optimizer_phi.zero_grad()
            # loss_h.backward(retain_graph=True)
            # ce_grad_norm = self.compute_grad_norm(self.phi_net)
            # epsilon = 1e-8
            # alpha = float(mse_grad_norm / (ce_grad_norm + epsilon))*0.002
            # alpha = max(0.0000001, min(alpha, 0.0000002))  # Clamp to safe range
            # self.optimizer_phi.zero_grad()
            # loss_phi = loss_mse - alpha * loss_h
            # trial code <-
            loss_phi.backward()
            self.optimizer_phi.step()

            if self.can_train_h_net() or self.config["fine_tune_epoch_h"] < epoch:
                self.optimizer_h.zero_grad() # remove h_net gradient gained from loss_phi 
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
            a_trace_per_step[case_num] = np.concatenate((a_trace_per_step[case_num], np.array([a_np])), axis=0)
        return loss_phi_sum, loss_h_sum, a_trace_per_step

    def compute_grad_norm(self, model):
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** 0.5


    def train_model(self):
        self.loss_phi_trace = []
        self.loss_h_trace = []
        for epoch in range(self.config["num_epochs"]):
            loss_phi, loss_h, a_trace = self.step_training(epoch)
            self.loss_phi_trace.append(loss_phi/self.num_of_conditions)
            self.loss_h_trace.append(loss_h/self.num_of_conditions)
            for case_num in range(len(a_trace)):
                # self.a_trace[case_num] = np.concatenate((self.a_trace[case_num], a_trace[case_num]), axis=0)
                self.a_trace[case_num][epoch] = a_trace[case_num]
            if epoch % 100 == 0:
                print('[%d] loss_f: %.2f loss_c: %.2f' % (epoch + 1, self.loss_phi_trace[-1], self.loss_h_trace[-1]))
    
    def verify_model(self, test_data_menu: list[str], is_x_y = True):
        self.phi_net.eval()
        with torch.no_grad():
            dataset = self.data_factory_instance.prepare_datasets(test_data_menu)
            for data, name in zip(dataset, test_data_menu):
                phi_out = self.phi_net(torch.tensor(data.input))
                print(f"phi_out: {phi_out}")
                groundtruth = torch.tensor(data.output)
                a = self.get_optimal_a(phi_out, groundtruth)
                print(f"a = {a}")
                prediction = self.get_prediction(phi_out, a)
                error = groundtruth - prediction
                self.plot_prediction_error(error, groundtruth, prediction, name)
                self.plot_phi_out(phi_out)
    
    def inspect_data(self, test_data: list[str]):
        with torch.no_grad():
            dataset = self.data_factory_instance.prepare_datasets(test_data)
            for data in dataset:
                groundtruth = torch.tensor(data.output)
                fig, axs = plt.subplots(3, 1)
                axs[0].plot(groundtruth[:, 0])
                axs[1].plot(groundtruth[:, 1])
                axs[2].plot(groundtruth[:, 2])
                axs[0].legend(["groundtruth", "prediction"]) 
                axs[0].set_ylabel('f_disturb_x')
                axs[1].set_ylabel('f_disturb_y')
                axs[2].set_ylabel('f_disturb_z')


    def plot_prediction_error(self, error, groundtruth, prediction, title):
        """assumes the output is 3d (fx,fy,fz)"""
        abs_error = np.abs(error.detach().cpu().numpy()) if isinstance(error, torch.Tensor) else np.abs(error)
        p90 = np.percentile(abs_error, 90, axis=0)
        rms_per_dim = np.sqrt(np.mean(abs_error**2, axis=0)) 
        print(f"RMS: {rms_per_dim}")

        fig, axs = plt.subplots(3, 2, figsize=(10, 6))
        fig.suptitle(f"{title}\n P90 Error: "
                    f"fx={p90[0]:.3f} N, fy={p90[1]:.3f} N, fz={p90[2]:.3f} N", fontsize=12)

        # Plot ground truth vs prediction
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

        # Plot errors
        axs[0, 1].plot(error[:, 0])
        axs[1, 1].plot(error[:, 1])
        axs[2, 1].plot(error[:, 2])
        axs[0, 1].legend(["error"])
        axs[0, 1].set_ylabel('error_x')
        axs[1, 1].set_ylabel('error_y')
        axs[2, 1].set_ylabel('error_z')

        # Add P90 text annotation to each error plot
        axs[0, 1].text(0.95, 0.95, f"P90={p90[0]:.3f}", ha='right', va='top', transform=axs[0, 1].transAxes)
        axs[1, 1].text(0.95, 0.95, f"P90={p90[1]:.3f}", ha='right', va='top', transform=axs[1, 1].transAxes)
        axs[2, 1].text(0.95, 0.95, f"P90={p90[2]:.3f}", ha='right', va='top', transform=axs[2, 1].transAxes)

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle        

    def plot_phi_out(self, out: torch.Tensor):
        dim = out.shape[-1]
        fig, axs = plt.subplots(dim, 1)
        for i in range(dim):
            axs[i].plot(out[:, i])
            axs[i].set_ylabel(f"phi_out_{i}")
        axs[-1].set_xlabel('epoch')        
        
    def plot_loss(self):
        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.loss_phi_trace)
        axs[0].set_ylabel('loss_phi_trace [N]')
        axs[1].plot(self.loss_h_trace)
        axs[1].set_ylabel('loss_h_trace')
        axs[1].set_xlabel('epoch')

    def plot_a(self):
        # a_trace shape: (num_of_conditions, num_of_iterations, dim_of_feature, dim_of_label)
        for i in range(len(self.a_trace)):
            _, row, col = self.a_trace[i].shape
            fig, axs = plt.subplots(row, col)
            for j in range(row):
                for k in range(col):
                    axs[j,k].plot(self.a_trace[i][:, j, k])

    def plot_tsne_of_a(self, selected_epochs: list[int]):
        """
        Visualize the t-SNE of all learned 'a' vectors.
        Assumes self.a_trace[domain_id] is shape [N, dim_a, dim_y] per domain.

        Args:
            title (str): Plot title
            max_points_per_domain (int): To subsample for large traces
        """
        all_a = []
        all_labels = []
        for domain_id, a_array in enumerate(self.a_trace):
            # a_array: shape (N, dim_a, dim_y), flatten over last dim
            valid_indices = [i for i in selected_epochs if i < len(a_array)]

            a_flat = a_array[valid_indices].reshape(len(valid_indices), -1)
            all_a.append(a_flat)
            all_labels.append(np.full(len(valid_indices), domain_id))

        X = np.concatenate(all_a, axis=0)
        Y = np.concatenate(all_labels, axis=0)

        # Run t-SNE
        X_tsne = TSNE(n_components=2, perplexity=5, learning_rate='auto', init='pca', random_state=0).fit_transform(X)

        # Plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=Y, palette='tab10', s=25)
        plt.title("t-SNE of a")
        plt.xlabel("t-SNE dim 1")
        plt.ylabel("t-SNE dim 2")
        plt.legend(title="Domain", loc='best')
        plt.grid(True)
        plt.tight_layout()

        





    

