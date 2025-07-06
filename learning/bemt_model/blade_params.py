import numpy as np
from scipy.interpolate import interp1d
import torch


class Blade:
    def __init__(self, num_of_blades=2, y_max=0.1, cl_1=5.3, cl_2=1.7, alpha_0=torch.tensor(np.radians(20.6)), cd=1.8, cd_0=0.01):
        self.num_of_blades = num_of_blades
        self.y_max = y_max
        self.y_min = 0
        self.cl_1 = cl_1
        self.cl_2 = cl_2
        self.alpha_0 = alpha_0
        self.cd = cd
        self.cd_0 = cd_0
        self.name = "NA"

    def get_chord(self, y: float):
        raise NotImplementedError
            
    def get_blade_pitch(self, y: float):
        """blade torsion angle relative to the rotation disk"""
        raise NotImplementedError


class APC_8x6(Blade):
    """
        Gill, Rajan, and Raffaello D'andrea. "Propeller thrust and drag in forward flight." 
        2017 IEEE Conference on control technology and applications (CCTA). IEEE, 2017.
    """  
    def __init__(self):
        super().__init__(num_of_blades=2, y_max=0.1, cl_1=5.3, cl_2=1.7, alpha_0=torch.tensor(np.radians(20.6)), cd=1.8, cd_0=0.01)
        x_table = [0.0, 0.3, 1.0]
        y_table = [0.8, 1.0, 0.0]
        self.interp_func = interp1d(x_table, y_table, kind="linear")
    
    def get_chord(self, y: float):
        """Get the chord length at a given y position along the blade."""
        min_length = 0.005
        max_length = 0.024
        weight = self.interp_func(y/self.y_max)
        length = (1 - weight)*min_length + weight*max_length
        return length

    def get_blade_pitch(self, y: float):
        min_pitch = torch.tensor(np.radians(17))
        max_pitch = torch.tensor(np.radians(45))
        pitch = self.linear_interpolate(max_pitch, min_pitch, y)
        return pitch
    
    def linear_interpolate(self, val_root, val_tip, y):
        weight = y/self.y_max
        result = (1 - weight)*val_root + weight*val_tip
        return result

class APC_8x6_OfficialData:
    # pad official data with 0.0 at the beginning
    OMEGA_APC8X6_OFFICIAL_DATA_RPM = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 11000, 12000, 13000, 14000, 15000, 16000, 17000, 18000, 19000, 20000, 21000, 22000, 23000, 24000, 25000]) # rpm
    THRUST_APC8X6_OFFICIAL_DATA_LBF = np.array([0.0, 0.015, 0.061, 0.138, 0.246, 0.385, 0.555, 0.757, 0.990, 1.256, 1.553, 1.884, 2.247, 2.643, 3.073, 3.537, 4.035, 4.569, 5.138, 5.743, 6.385, 7.063, 7.779, 8.532, 9.321, 10.146]) # pound force
    @staticmethod
    def get_omega_range():
        # convert the official data to ISO units
        rpm_to_rad_per_sec = 2*np.pi/60
        return (APC_8x6_OfficialData.OMEGA_APC8X6_OFFICIAL_DATA_RPM * rpm_to_rad_per_sec)
    
    @staticmethod
    def get_thrust_range():
        # convert the official data to ISO units
        pound_force_to_newton = 4.44822
        return (APC_8x6_OfficialData.THRUST_APC8X6_OFFICIAL_DATA_LBF * pound_force_to_newton)
    
