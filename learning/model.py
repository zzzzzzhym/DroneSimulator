import torch 
import torch.nn as nn
import torch.nn.functional as F
torch.set_default_tensor_type('torch.DoubleTensor')
import yaml
import os


class MultilayerNet(nn.Module):
    """Takes config that specifies the input, hidden layer and output dimensions and makes the layers accordingly.
    The forward function need to be defined as requested by nn.Module. But it's not shown in this base class."""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, can_append_bias: bool = False):
        super().__init__()
        self.dim_of_input = dim_of_input
        self.dim_of_output = dim_of_output

        self.layers = nn.ModuleList()
        if can_append_bias:
            nn_output = dim_of_output - 1 # append bias appended in forward
        else:
            nn_output = dim_of_output
        if len(dim_of_layers) == 0: # No hidden layers, direct input to output    
            self.layers.append(nn.Linear(dim_of_input, nn_output))
        else:
            self.layers.append(nn.Linear(dim_of_input, dim_of_layers[0]))
            for i in range(1, len(dim_of_layers)):
                self.layers.append(nn.Linear(dim_of_layers[i - 1], dim_of_layers[i]))
            self.layers.append(nn.Linear(dim_of_layers[-1], nn_output))    

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x 

class PhiNet(MultilayerNet):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list, input_mean, input_scale, output_mean, output_scale):
        super().__init__(dim_of_input, dim_of_output, dim_of_layers, can_append_bias=True)
        # Normalization: use register_buffer to avoid being changed by optimizer
        self.register_buffer("input_mean", torch.tensor(input_mean, dtype=torch.float32))
        self.register_buffer("input_scale", torch.tensor(input_scale, dtype=torch.float32))

        # output normalization is used as (a*phi(x) - output_mean)*output_scale
        # it's not used when outputing the kernel, but used after applying adaptation coefficients
        self.register_buffer("output_mean", torch.tensor(output_mean, dtype=torch.float32))
        self.register_buffer("output_scale", torch.tensor(output_scale, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input, assuming x is raw input
        x = (x - self.input_mean) * self.input_scale

        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        if len(x.shape) == 1:
            # single input (batch size == 1)
            return torch.cat([x, torch.ones(1)])
        else:
            # batch input for training (assuming that x is at most 2D)
            return torch.cat([x, torch.ones([x.shape[0], 1])], dim=-1)
        

class HNet(MultilayerNet):
    """Follow NeuroFly Nomenclature"""
    def __init__(self, dim_of_input, dim_of_output, dim_of_layers: list):
        super().__init__(dim_of_input, dim_of_output, dim_of_layers, can_append_bias=False)        

    
class ModelFactory:
    def __init__(self,
                 num_of_conditions,
                 dim_of_input,
                 input_mean,
                 input_scale,
                 label_mean,
                 label_scale,
                 customized_config: dict = None
                 ):
        """consider the following example to predict disturbance forces:
        (v_0 v_1 v_2 q_0 q_1 q_2 q_3 pwm_0 pwm_1 pwm_2 pwm_3)
        velocity (3) + quaternion (4) + input (4) = 11
        dim_of_input: input dimension (11 in this case)
        dim_of_feature: features that phi_net need to capture (not necessarily the same as dim of disturbance forces because it needs to interact with adaptation coefficients)
        From the paper, the feature dimension is 5.
        dim_of_label: dimension of label vector. Because we are trying to compare the disturbance force in x y z direction,
        the label vector is a 3-dim vector because disturbance torque is not estimated in the paper.
        """               
        self.config = self.set_up_config(customized_config)
        self.num_of_conditions = num_of_conditions
        self.dim_of_input = dim_of_input
        self.input_mean = input_mean
        self.input_scale = input_scale
        self.label_mean = label_mean
        self.label_scale = label_scale

    def generate_nets(self):
        phi_net = PhiNet(
            self.dim_of_input,
            self.config["dim_of_feature"],
            self.config["PhiNet"]["hidden_layer_dimensions"],
            self.input_mean,
            self.input_scale,
            self.label_mean,
            self.label_scale
        )
        h_net = HNet(
            self.config["dim_of_feature"],
            self.num_of_conditions,
            self.config["HNet"]["hidden_layer_dimensions"]
        )
        return phi_net, h_net

    @staticmethod
    def set_up_config(customized_config: dict = None):
        """Set up the model configuration. If customized_config is None, load the default config."""
        if customized_config is None:
            config_path = ModelFactory.get_default_config_path()
            config = ModelFactory.load_config(config_path)
        else:
            config = customized_config  # in case of loading the trained model, use the documented config at generation time
        return config

    @staticmethod
    def get_default_config_path():
        """Get the default path of the model configuration file"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(current_dir, "model_config.yaml")

    @staticmethod
    def load_config(config_path: str):
        """Load model configuration from YAML file"""
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        return config
    
    def generate_self_config(self) -> dict:
        """Generate a self-contained configuration dictionary. 
        Use this as input argument to recreate the same factory and produce the exact model later."""
        return {
            "num_of_conditions": self.num_of_conditions,
            "dim_of_input": self.dim_of_input,
            "input_mean": self.input_mean,
            "input_scale": self.input_scale,
            "label_mean": self.label_mean,
            "label_scale": self.label_scale,
            "customized_config": self.config
        }
