import torch
import numpy as np
import matplotlib.pyplot as plt

class SigmaFunction:
    def __init__(self, alpha_0=torch.tensor(np.radians(25)), M=25):
        """
        Initialize the SigmaFunction with parameters and caps.

        Parameters:
        - alpha0: Stall angle (radians)
        - M: Sharpness parameter
        - alpha_cap: Tuple of (min_alpha, max_alpha) for capping alpha values
        - M_cap: Tuple of (min_M, max_M) for capping M values
        """
        self.alpha_0 = torch.clip(alpha_0, torch.tensor(-30, dtype=alpha_0.dtype), torch.tensor(30, dtype=alpha_0.dtype))
        self.M = np.clip(M, 10, 50)
        self.alpha_cap = torch.pi/2

    def compute(self, alpha):
        """
        Compute the sigma function for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - Sigma value (float)
        """

        alpha = torch.clip(alpha, -self.alpha_cap, self.alpha_cap)
        # Compute sigma
        exp1 = torch.exp(-self.M * (alpha - self.alpha_0))
        exp2 = torch.exp(self.M * (alpha + self.alpha_0))
        numerator = 1 + exp1 + exp2
        denominator = (1 + exp1) * (1 + exp2)
        return numerator / denominator


if __name__ == "__main__":
    # Example usage
    alpha0 = np.radians(20)  # Stall angle in radians
    M = 25                   # Sharpness parameter
    sigma = SigmaFunction(alpha_0=alpha0, M=M)


    # Define parameters
    alpha = np.linspace(-30, 30, 500)  # Angle of attack in degrees
    alpha_rad = np.radians(alpha)     # Convert to radians
    alpha0 = np.radians(20)           # Stall angle in radians
    M_values = [10, 25, 50]           # Different sharpness parameters

    # Plot the sigma function for different values of M
    plt.figure(figsize=(10, 6))
    for M in M_values:
        sigma.M = M
        sigma_values = sigma.compute(alpha_rad)
        plt.plot(alpha, sigma_values, label=f'M = {M}')

    # Customize the plot
    plt.title("Sigma Function Profile", fontsize=16)
    plt.xlabel("Angle of Attack (degrees)", fontsize=14)
    plt.ylabel("Sigma Value", fontsize=14)
    plt.axvline(np.degrees(alpha0), color='r', linestyle='--', label=r'$\alpha_0$ (Stall Angle)')
    plt.legend(fontsize=12)
    plt.grid(True)
    plt.show()