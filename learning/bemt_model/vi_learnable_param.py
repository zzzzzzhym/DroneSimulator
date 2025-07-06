import torch
import torch.nn as nn

class LearnableInflowSpeed(nn.Module):
    """v_i is 0D tensor, representing a single learnable parameter for inflow speed"""
    def __init__(self, init_val: torch.Tensor):
        """
        Args:
            init_val (torch.Tensor): 0D initial value for the inflow speed, should be a scalar tensor
        """
        super().__init__()
        self.v_i = nn.Parameter(init_val)

    def forward(self):
        """
        Args:
            y_index (torch.Tensor): index or indices into the blade element (e.g., 0 to N-1)
        Returns:
            torch.Tensor: corresponding v_i values
        """
        return self.v_i
