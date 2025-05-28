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

class Validator:
    def __init__(self,
                 config: dict,
                 ) -> None:
        self.phi_net = None
        self.h_net = None
        self.config = config
        self.dataset = None

    def load_model(self, phi_net: model.MultilayerNet, h_net: model.MultilayerNet):
        self.phi_net = phi_net
        self.h_net = h_net

    def load_dataset(self, dataset: data_factory.LearningDataset):
        self.dataset = dataset

    def validate_model(self):
        self.phi_net.eval()
        with torch.no_grad():
            for data in self.dataset:
                phi_out = self.phi_net(torch.tensor(data.input))
                print(f"phi_out: {phi_out}")
                groundtruth = torch.tensor(data.output)
                a = self.get_optimal_a(phi_out, groundtruth)
                print(f"a = {a}")
                prediction = self.get_prediction(phi_out, a)
                error = groundtruth - prediction
                denormalized_error = self.denormalize(
                    error.numpy(),
                    data.label_mean_vector,
                    data.label_scale_vector
                )
                denormalized_groundtruth = self.denormalize(
                    groundtruth.numpy(),
                    data.label_mean_vector,
                    data.label_scale_vector
                )
                denormalized_prediction = self.denormalize(
                    prediction.numpy(),
                    data.label_mean_vector,
                    data.label_scale_vector
                )
                self.plot_prediction_error(denormalized_error, denormalized_groundtruth, denormalized_prediction, data.source_file)
                self.plot_phi_out(phi_out)
                plt.show()

    @staticmethod
    def denormalize(normalized: np.ndarray, mean: np.ndarray, scale: np.ndarray) -> np.ndarray:
        """denormalize the output to get disturbance"""
        return normalized / scale + mean

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
    
    def plot_prediction_error(self, error, groundtruth, prediction, title):
        """assumes the output is 3d (fx,fy,fz)"""
        # check dimensions
        dim = groundtruth.shape[-1]
        abs_error = np.abs(error.detach().cpu().numpy()) if isinstance(error, torch.Tensor) else np.abs(error)
        p90 = np.percentile(abs_error, 90, axis=0)
        rms_per_dim = np.sqrt(np.mean(abs_error**2, axis=0)) 
        print(f"RMS: {rms_per_dim}")
        print(f"P90: {p90}")

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

    def plot_phi_out(self, out: torch.Tensor):
        dim = out.shape[-1]
        fig, axs = plt.subplots(dim, 1)
        for i in range(dim):
            axs[i].plot(out[:, i])
            axs[i].set_ylabel(f"phi_out_{i}")
        axs[-1].set_xlabel('epoch')        