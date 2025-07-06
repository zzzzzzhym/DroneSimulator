import numpy as np
import matplotlib.pyplot as plt
import torch

import airfoil.sigma_function as sigma_function
from airfoil.air import Air



class Coeffecients:
    def __init__(self, cl_1=5.3, cl_2=1.7, alpha_0=torch.tensor(np.radians(20.6)), cd=1.8, cd_0=0.01, cp=1.328):
        """
        Initialize the Coeffecients with parameters.

        Parameters:
        - cl_1: Lift coefficient parameter
        - cl_2: Lift coefficient parameter
        - cd: Drag coefficient parameter
        - alpha0: Stall angle in radians
        """
        self.cl_1 = cl_1
        self.cl_2 = cl_2
        self.alpha_0 = alpha_0
        self.cd = cd
        self.cd_0 = cd_0
        self.cp = cp
        self.sigma = sigma_function.SigmaFunction(alpha_0=self.alpha_0)


    def get_cl(self, alpha):
        """
        Compute the lift coefficient for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - CL value (float)
        """
        CL = (1 - self.sigma.compute(alpha))*self.cl_1 * alpha + self.sigma.compute(alpha)*self.cl_2 * torch.sin(alpha) * torch.cos(alpha)

        return CL
    
    def get_cd(self, alpha, u, chord):
        """
        Compute the drag coefficient for a given angle of attack (alpha).

        Parameters:
        - alpha: Angle of attack (radians)

        Returns:
        - CD value (float)
        """
        rn_clamped = torch.maximum(Coeffecients.get_reynolds_number(u, chord), torch.tensor(1.0))
        CD = self.cd * torch.sin(alpha)**2 + 2*1.02*self.cp / torch.sqrt(rn_clamped) + self.cd_0
        return CD
    
    @staticmethod
    def get_reynolds_number(u, chord):
        """
        Compute the Reynolds number for a given velocity and chord length.

        Parameters:
        - U: Velocity (m/s)
        - chord: Chord length (m)

        Returns:
        - Reynolds number (float)
        """
        rn = Air.rho * u * chord / Air.mu
        return rn
    
    
if __name__ == "__main__":
    alpha = np.linspace(-180, 180, 500)
    coeff = Coeffecients()
    cl = coeff.get_cl(np.radians(alpha))
    cd = coeff.get_cd(np.radians(alpha), 10, 0.1)

    plt.figure(figsize=(10, 6))
    plt.plot(alpha, cl, label='CL')
    plt.plot(alpha, cd, label='CD')
    plt.plot(alpha, cl/cd, label='CL/CD')

    plt.xlabel('Angle of Attack (degrees)')
    plt.ylabel('Coefficients')
    plt.legend()
    plt.show()
