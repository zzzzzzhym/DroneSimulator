import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import model
import data_factory


class Evaluator:
    @staticmethod
    def denormalize(normalized: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """denormalize the output to get disturbance"""
        return normalized / scale + mean


class DiamlEvaluator(Evaluator):
    def __init__(self) -> None:
        self.phi_net = None
        self.h_net = None
        self.datasets = None
        self.condition_labels = None

    def load_model(self, phi_net: model.MultilayerNet, h_net: model.MultilayerNet):
        self.phi_net = phi_net
        self.h_net = h_net

    def load_dataset(self, datasets: list[data_factory.DiamlDataset]):
        self.datasets = datasets
        self.condition_labels = [dataset.source_file for dataset in datasets]

    def evaluate_model(self, can_show_plot=False):
        is_in_training = self.phi_net.training
        if is_in_training:
            self.phi_net.eval()
        with torch.no_grad():
            mse = []
            for dataset in self.datasets:
                phi_out = self.phi_net(torch.tensor(dataset.input))
                groundtruth = torch.tensor(dataset.output)
                a = self.get_optimal_a(phi_out, groundtruth)
                prediction = self.get_prediction(phi_out, a)
                error = groundtruth - prediction
                denormalized_error = Evaluator.denormalize(
                    error.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                denormalized_groundtruth = Evaluator.denormalize(
                    groundtruth.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                denormalized_prediction = Evaluator.denormalize(
                    prediction.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                mse_per_dataset = self.get_mse_per_label(denormalized_error)
                mse.append(mse_per_dataset)

                if can_show_plot:
                    print(f"phi_out: {phi_out}")
                    print(f"a = {a}")
                    self.plot_prediction_error(denormalized_error, denormalized_groundtruth, denormalized_prediction, dataset.source_file)
                    self.plot_nn_out(phi_out)
                    plt.show()            
        if is_in_training:
            self.phi_net.train()
        return mse

    def get_zero_prediction_mse(self):
        """MSE if model output is zero"""
        mse = []
        for dataset in self.datasets:
            groundtruth = torch.tensor(dataset.output)
            mse_per_dataset = self.get_mse_per_label(groundtruth)
            mse.append(mse_per_dataset)
        return mse

    def test_model(self):
        mse = self.evaluate_model(can_show_plot=True)
        zero_prediction_mse = self.get_zero_prediction_mse()
        rms = np.sqrt(np.array(mse))  # convert mse to rms
        self.plot_rms_grouped_by_dimension(rms)
        self.plot_mse_grouped_by_dimension(mse, zero_prediction_mse)

    def callback_validation(self):
        mse = self.evaluate_model(can_show_plot=False)
        result = 0.0
        for mse_per_dataset in mse:
            result += np.mean(mse_per_dataset)
        return result / len(mse)



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
        # do not normalize a with config['gamma'] as in the training
        return a
    
    @staticmethod
    def get_prediction(phi, a):
        return torch.mm(phi, a)    
    
    def get_mse_per_label(self, error):
        if isinstance(error, torch.Tensor):
            error = error.detach().cpu().numpy()
        mse = np.mean(error**2, axis=0)
        return mse

    def get_rms_error(self, error):        
        rms_per_dim = np.sqrt(self.get_mse_per_label(error)) 
        return rms_per_dim

    def plot_rms_grouped_by_dimension(self, rms_list):
        rms_array = np.array(rms_list)  # shape: (num_conditions, num_dims)
        num_conditions, num_dims = rms_array.shape

        x = np.arange(num_conditions)  # x axis positions per condition
        width = 0.8 / num_dims  # space out bars within a group

        plt.figure(figsize=(12, 6))
        for dim in range(num_dims):
            print(f"average rms for dim {dim}: {np.mean(rms_array[:, dim])}")
            plt.bar(x + dim * width, rms_array[:, dim], width=width, label=f'Dim {dim}')

        # Labeling
        plt.xticks(
            x + width * (num_dims - 1) / 2,
            self.condition_labels,
            rotation=90
        )

        plt.title("RMS Error per Condition (Grouped by Output Dimension)")
        plt.xlabel("Condition")
        plt.ylabel("RMS Error")
        plt.legend(title="Output Dim")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_mse_grouped_by_dimension(self, mse_list, zero_prediction_mse_list):
        mse_array = np.array(mse_list)  # shape: (num_conditions, num_dims)
        zero_prediction_mse_array = np.array(zero_prediction_mse_list)
        num_conditions, num_dims = mse_array.shape

        x = np.arange(num_conditions)  # x axis positions per condition
        width = 0.8 / num_dims  # space out bars within a group
        colors = plt.cm.tab10.colors  # 10 distinct colors (cycle if >10 dims)

        plt.figure(figsize=(12, 6))
        for dim in range(num_dims):
            color = colors[dim % len(colors)]
            print(f"average mse for dim {dim}: {np.mean(mse_array[:, dim])}")
            plt.bar(x + dim * width,
                    mse_array[:, dim],
                    width=width,
                    label=f'Dim {dim}',
                    color=color)
            plt.scatter(x + dim * width,
                        zero_prediction_mse_array[:, dim],
                        label=f'Dim {dim} (mse before training)',
                        color=color,
                        marker='o',
                        edgecolors='black')  # optional outline for visibility

        # Labeling
        plt.xticks(
            x + width * (num_dims - 1) / 2,
            self.condition_labels,
            rotation=90
        )
        plt.title("MSE Error per Condition (Grouped by Output Dimension)")
        plt.xlabel("Condition")
        plt.ylabel("MSE Error")
        plt.legend(title="Output Dim")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_prediction_error(self, error, groundtruth, prediction, title):
        """assumes the output is 3d (fx,fy,fz)"""
        # check dimensions
        dim = groundtruth.shape[-1]

        fig, axs = plt.subplots(3, 2, figsize=(10, 6))

        fig.suptitle(f"{title}", fontsize=12)

        # Plot ground truth vs prediction and errors using a loop
        labels = ['x', 'y', 'z']

        for i in range(dim):
            # Ground truth vs prediction
            axs[i, 0].plot(groundtruth[:, i])
            axs[i, 0].plot(prediction[:, i])
            axs[i, 0].legend(["groundtruth", "prediction"])
            axs[i, 0].set_ylabel(f'f_disturb_{labels[i]}')

            # Error
            axs[i, 1].plot(error[:, i])
            axs[i, 1].legend(["error"])
            axs[i, 1].set_ylabel(f'error_{labels[i]}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle        

    def plot_nn_out(self, out: torch.Tensor):
        dim = out.shape[-1]
        fig, axs = plt.subplots(dim, 1)
        for i in range(dim):
            if dim == 1:
                ax = axs
            else:
                ax = axs[i]
            ax.plot(out[:, i])
            ax.set_ylabel(f"phi_out_{i}")
        ax.set_xlabel('epoch')


class SimpleEvaluator(Evaluator):
    def __init__(self) -> None:
        self.simple_net = None
        self.datasets = None
        self.condition_labels = None

    def load_model(self, simple_net: model.MultilayerNet):
        self.simple_net = simple_net

    def load_dataset(self, datasets: list[data_factory.DiamlDataset]):
        self.datasets = datasets
        self.condition_labels = [dataset.source_file for dataset in datasets]

    def evaluate_model(self, can_show_plot=False):
        is_in_training = self.simple_net.training
        if is_in_training:
            self.simple_net.eval()
        with torch.no_grad():
            mse = []
            for dataset in self.datasets:
                prediction = self.simple_net(torch.tensor(dataset.input))
                groundtruth = torch.tensor(dataset.output)
                error = groundtruth - prediction
                denormalized_error = Evaluator.denormalize(
                    error.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                denormalized_groundtruth = Evaluator.denormalize(
                    groundtruth.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                denormalized_prediction = Evaluator.denormalize(
                    prediction.numpy(),
                    dataset.label_mean_vector,
                    dataset.label_scale_vector
                )
                mse_per_dataset = self.get_mse_per_label(denormalized_error)
                mse.append(mse_per_dataset)

                if can_show_plot:
                    print(f"prediction: {prediction}")
                    self.plot_prediction_error(denormalized_error, denormalized_groundtruth, denormalized_prediction, dataset.source_file)
                    self.plot_nn_output(prediction)
                    plt.show()            
        if is_in_training:
            self.simple_net.train()
        return mse
    
    def plot_nn_output(self, out: torch.Tensor):
        dim = out.shape[-1]
        fig, axs = plt.subplots(dim, 1, sharex=True)
        for i in range(dim):
            if dim == 1:
                ax = axs
            else:
                ax = axs[i]
            ax.plot(out[:, i])
            ax.set_ylabel(f"nn_{i}")
        ax.set_xlabel('epoch')    


    def plot_prediction_error(self, error, groundtruth, prediction, title):
        """assumes the output is 3d (fx,fy,fz)"""
        # check dimensions
        dim = groundtruth.shape[-1]

        fig, axs = plt.subplots(3, 2, figsize=(10, 6), sharex=True)

        fig.suptitle(f"{title}", fontsize=12)

        # Plot ground truth vs prediction and errors using a loop
        labels = ['x', 'y', 'z']

        for i in range(dim):
            # Ground truth vs prediction
            axs[i, 0].plot(groundtruth[:, i])
            axs[i, 0].plot(prediction[:, i])
            axs[i, 0].legend(["groundtruth", "prediction"])
            axs[i, 0].set_ylabel(f'f_disturb_{labels[i]}')

            # Error
            axs[i, 1].plot(error[:, i])
            axs[i, 1].legend(["error"])
            axs[i, 1].set_ylabel(f'error_{labels[i]}')

        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle           

    
    def get_mse_per_label(self, error):
        if isinstance(error, torch.Tensor):
            error = error.detach().cpu().numpy()
        mse = np.mean(error**2, axis=0)
        return mse

    def get_rms_error(self, error):        
        rms_per_dim = np.sqrt(self.get_mse_per_label(error)) 
        return rms_per_dim

    def plot_rms_grouped_by_dimension(self, rms_list):
        rms_array = np.array(rms_list)  # shape: (num_conditions, num_dims)
        num_conditions, num_dims = rms_array.shape

        x = np.arange(num_conditions)  # x axis positions per condition
        width = 0.8 / num_dims  # space out bars within a group

        plt.figure(figsize=(12, 6))
        for dim in range(num_dims):
            print(f"average rms for dim {dim}: {np.mean(rms_array[:, dim])}")
            plt.bar(x + dim * width, rms_array[:, dim], width=width, label=f'Dim {dim}')

        # Labeling
        plt.xticks(
            x + width * (num_dims - 1) / 2,
            self.condition_labels,
            rotation=90
        )

        plt.title("RMS Error per Condition (Grouped by Output Dimension)")
        plt.xlabel("Condition")
        plt.ylabel("RMS Error")
        plt.legend(title="Output Dim")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_mse_grouped_by_dimension(self, mse_list, zero_prediction_mse_list):
        mse_array = np.array(mse_list)  # shape: (num_conditions, num_dims)
        zero_prediction_mse_array = np.array(zero_prediction_mse_list)
        num_conditions, num_dims = mse_array.shape

        x = np.arange(num_conditions)  # x axis positions per condition
        width = 0.8 / num_dims  # space out bars within a group
        colors = plt.cm.tab10.colors  # 10 distinct colors (cycle if >10 dims)

        plt.figure(figsize=(12, 6))
        for dim in range(num_dims):
            color = colors[dim % len(colors)]
            print(f"average mse for dim {dim}: {np.mean(mse_array[:, dim])}")
            plt.bar(x + dim * width,
                    mse_array[:, dim],
                    width=width,
                    label=f'Dim {dim}',
                    color=color)
            plt.scatter(x + dim * width,
                        zero_prediction_mse_array[:, dim],
                        label=f'Dim {dim} (mse before training)',
                        color=color,
                        marker='o',
                        edgecolors='black')  # optional outline for visibility

        # Labeling
        plt.xticks(
            x + width * (num_dims - 1) / 2,
            self.condition_labels,
            rotation=90
        )
        plt.title("MSE Error per Condition (Grouped by Output Dimension)")
        plt.xlabel("Condition")
        plt.ylabel("MSE Error")
        plt.legend(title="Output Dim")
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()        

    def get_zero_prediction_mse(self):
        """MSE if model output is zero"""
        mse = []
        for dataset in self.datasets:
            groundtruth = torch.tensor(dataset.output)
            mse_per_dataset = self.get_mse_per_label(groundtruth)
            mse.append(mse_per_dataset)
        return mse

    def test_model(self):
        mse = self.evaluate_model(can_show_plot=True)
        zero_prediction_mse = self.get_zero_prediction_mse()
        rms = np.sqrt(np.array(mse))  # convert mse to rms
        self.plot_rms_grouped_by_dimension(rms)
        self.plot_mse_grouped_by_dimension(mse, zero_prediction_mse)
        
    def callback_validation(self):
        mse = self.evaluate_model(can_show_plot=False)
        result = 0.0
        for mse_per_dataset in mse:
            result += np.mean(mse_per_dataset)
        return result / len(mse)