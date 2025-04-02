import numpy as np
import matplotlib.pyplot as plt

class Propeller:
    """This class manages the relationship between rotation speed and thrust of a propeller and does not consider rotor and its relationship to the drone.
    Lookup table according to
    de Pierrepont Franzetti, David Du Mutel, et al. "Ground, Ceiling and Wall Effect Evaluation of Small Quadcopters in Pressure-controlled Environments." Journal of Intelligent & Robotic Systems 110.3 (2024): 1-19.
    """    
    def __init__(self, n_inf: np.ndarray,  f_z_inf: np.ndarray, diameter: float) -> None:
        """
        n_inf: np.ndarray   # rotation speed [rpm]
        f_z_inf: np.ndarray # thrust in z direction [N]
        diameter: float     # rotor diameter
        coeffs: np.ndarray  # f = a0 n^2 + a1 n
        Note: in the paper, _inf means the propeller is away from the ground, a ceiling, or a wall
        """
        self.n_inf = n_inf
        self.f_z_inf = f_z_inf
        self.diameter = diameter
        self.coeffs = self.fit_n_f_relationship(n_inf, f_z_inf)

    @staticmethod
    def fit_n_f_relationship(n: np.ndarray, f: np.ndarray):
        """get n-f quadratic function using data samples
        a0 n^2 + a1 n = f
        """
        n_m = np.vstack([n**2, n]).T
        a, residuals, rank, s = np.linalg.lstsq(n_m, f, rcond=None)
        return a

    def get_thrust(self, n: float) -> float:
        """Interpolates the thrust for a given rotation speed
        f = a0 n^2 + a1 n
        """
        # Clip the input to be within the range of n_inf
        n_clamped = np.clip(n, 0, 20000)
        return self.coeffs[0]*n_clamped**2 + self.coeffs[1]*n_clamped
    
    def get_rotation_speed(self, f_z: float) -> float:
        """Interpolates the speed (rps) for a given thrust
        sovle n from a0 n^2 + a1 n = f
        Returns:
            float: rotation speed in rpm
        """
        # Clip the input to be within the range of f_z_inf
        f_z_clamped = np.clip(f_z, 0, 80)
        n = (-self.coeffs[1] + np.sqrt(self.coeffs[1]**2 + 4*self.coeffs[0]*f_z_clamped)) / (2*self.coeffs[0])  # rpm
        return n

    def plot_fitted_curve(self):
        x = np.arange(-9000, 9000)
        plt.plot(x, self.get_thrust(x))
        y = np.arange(-80, 80)
        plt.plot(self.get_rotation_speed(y), y)
        plt.plot(self.n_inf, self.f_z_inf, marker='x', linestyle="None")
        plt.grid(True)
        plt.xlabel("rotor speed rpm")
        plt.ylabel("thrust N")


def convert_rpm_to_rps(rpm):
    return rpm/60

"""propeller_size [inch]"""
"""from Ground, Ceiling and Wall Effect Evaluation of Small Quadcopters in Pressure-controlled Environments"""
prop_12x5 = Propeller(
    n_inf=np.array([0, 2000, 2600, 3700, 4000]), 
    f_z_inf=np.array([0, 3.5, 6.4, 12.8, 15.5]),
    diameter=0.3048
)

prop_10x4_7 = Propeller(
    n_inf=np.array([0, 2000, 3386, 4000, 4800]),
    f_z_inf=np.array([0, 1.8, 6.4, 8.4, 12.8]),
    diameter=0.254
)

prop_10x4_5 = Propeller(
    n_inf=np.array([0, 3000, 4000, 4400, 4800]),
    f_z_inf=np.array([0, 4.3, 8, 10, 12]),
    diameter=0.254
)

prop_12x6 = Propeller(
    n_inf=np.array([0, 3000, 3600, 4000, 4400]),
    f_z_inf=np.array([0, 5.5, 8, 10, 12]),
    diameter=0.3048
)

prop_13x6_5 = Propeller(
    n_inf=np.array([0, 2500, 2800, 3000, 3100]),
    f_z_inf=np.array([0, 8, 10, 12, 12.5]),
    diameter=0.3302
)
"""virtual properller estimated from the paper"""
prop_15x5 = Propeller(
    n_inf=np.array([0, 2000, 2600, 3700, 4000]), 
    f_z_inf=np.array([0, 3.5, 6.4, 12.8, 15.5])*(15/12)**2,
    diameter=0.3048*15/12
)

"""KDE direct motor-propeller combination
motor versin: KDE4215XF-465kv
voltage: 25.2V 6s
propeller: 15.5x5.3 DUAL-EDN(KDE) (dual-blade)
https://www.kdedirect.com/products/kde4215xf-465?_pos=1&_sid=b2fc4089f&_ss=r
'Performance Data' tab in the link above
"""
prop_kde4215xf465_6s_15_5x5_3_dual = Propeller(
    n_inf=np.array([0, 3480, 4340, 5280, 6300, 7210, 8160, 8800]),
    f_z_inf=np.array([0.0, 7.20, 10.77, 16.13, 23.33, 30.64, 38.56, 46.02]),
    diameter=0.3048*15/12
)

"""APC_8x6 propeller from inflow model (not ready)"""
apc_8x6 = Propeller(
    n_inf=np.array([0, 200, 300, 400, 500, 600, 700, 800, 900, 1000])*60/(2*np.pi),     # rad/s to rpm
    f_z_inf=np.array([0.07323629, 0.31446133, 0.55298213, 0.88470807, 1.3132933, 1.84148013, 2.47178404, 3.20659931, 4.04818159, 4.99853133]),
    diameter=0.2032
)

if __name__ == "__main__":
    # prop_12x5.plot_fitted_curve()
    # prop_kde4215xf465_6s_15_5x5_3_dual.plot_fitted_curve()
    apc_8x6.plot_fitted_curve()
    plt.show()
    print(prop_15x5.get_rotation_speed(6.3))