import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
torch.set_default_tensor_type('torch.DoubleTensor')
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

import model

class Validator:
    def __init__(self,
                 config: dict,
                 ) -> None:
        self.phi_net = None
        self.h_net = None
        self.config = config

    def load_model(self, phi_net: model.MultilayerNet, h_net: model.MultilayerNet):
        self.phi_net = phi_net
        self.h_net = h_net

    def load_dataset(self, dataset: Dataset):
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
                self.plot_prediction_error(error, groundtruth, prediction, "tbd")
                self.plot_phi_out(phi_out)

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