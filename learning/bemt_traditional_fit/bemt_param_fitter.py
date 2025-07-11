import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from inflow_model.bet import BladeElementTheory
from inflow_model.blade_params import Blade
from drone import utils
from drone import parameters
import data_factory

class FittedBlade(Blade):
    """
        Gill, Rajan, and Raffaello D'andrea. "Propeller thrust and drag in forward flight." 
        2017 IEEE Conference on control technology and applications (CCTA). IEEE, 2017.
    """  
    def __init__(self):
        super().__init__(num_of_blades=2, y_max=0.1, cl_1=5.3, cl_2=1.7, alpha_0=np.radians(20.6), cd=1.8, cd_0=0.01)
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
        min_pitch = np.radians(17)
        max_pitch = np.radians(45)
        pitch = self.linear_interpolate(max_pitch, min_pitch, y)
        return pitch
    
    def linear_interpolate(self, val_root, val_tip, y):
        weight = y/self.y_max
        result = (1 - weight)*val_root + weight*val_tip
        return result

def compute_model_thrust(u_free, v_forward, r_disk, omega, is_ccw_blade, bet_instance: BladeElementTheory):
    f_x, f_y, f_z, v_i = bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
    return np.array([f_x, f_y, f_z])

def get_loss(x, dataset: data_factory.FittingDataset, blade: FittedBlade, can_collect_data_to_plot=False):
    loss = 0.0
    params = parameters.PennStateARILab550()
    cl_1, cl_2, cd, alpha_0 = x
    blade.cl_1 = cl_1
    blade.cl_2 = cl_2
    blade.cd = cd
    blade.alpha_0 = alpha_0    
    data_len = len(dataset.u_free_0)
    sample_distance = 100
    bet_instance = BladeElementTheory(blade, num_of_elements=100, num_of_rotation_segments=90)
    if can_collect_data_to_plot:
        data = {"i": [], "f_0": [], "f_1": [], "f_2": [], "f_3": [], "f_total": []}
    for i in range(0, data_len, sample_distance):
        f_0 = compute_model_thrust(dataset.u_free_0[i], 
                                 dataset.v_forward_0[i], 
                                 dataset.shared_r_disk[i], 
                                 dataset.omega_0[i], 
                                 params.is_ccw_blade[0],
                                 bet_instance)
        # torque_0 = np.cross(dataset.relative_position_inertial_frame_0[i], f_0)

        f_1 = compute_model_thrust(dataset.u_free_1[i], 
                                 dataset.v_forward_1[i], 
                                 dataset.shared_r_disk[i], 
                                 dataset.omega_1[i], 
                                 params.is_ccw_blade[1],
                                 bet_instance)
        # torque_1 = np.cross(dataset.relative_position_inertial_frame_1[i], f_1)

        f_2 = compute_model_thrust(dataset.u_free_2[i], 
                                 dataset.v_forward_2[i], 
                                 dataset.shared_r_disk[i], 
                                 dataset.omega_2[i], 
                                 params.is_ccw_blade[2],
                                 bet_instance)
        # torque_2 = np.cross(dataset.relative_position_inertial_frame_2[i], f_2)

        f_3 = compute_model_thrust(dataset.u_free_3[i], 
                                 dataset.v_forward_3[i], 
                                 dataset.shared_r_disk[i], 
                                 dataset.omega_3[i], 
                                 params.is_ccw_blade[3],
                                 bet_instance)
        # torque_3 = np.cross(dataset.relative_position_inertial_frame_3[i], f_3)

        f = f_0 + f_1 + f_2 + f_3
        f = utils.FrdFluConverter.flip_vector(f)
        if can_collect_data_to_plot:
            data["i"].append(i)
            data["f_0"].append(f_0)
            data["f_1"].append(f_1)
            data["f_2"].append(f_2)
            data["f_3"].append(f_3)
            data["f_total"].append(f)
        loss_f = np.linalg.norm(utils.FrdFluConverter.flip_vector(dataset.shared_r_disk[i]@f) + params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) - params.m * dataset.dv[i])**2
        loss += loss_f
    if can_collect_data_to_plot:
        plot_the_fit(data, dataset)
    loss = loss/data_len*sample_distance
    print(f"Current loss: {loss}")
    return loss

def fit_params(dataset, blade: FittedBlade):
    
    # Initial guess for cl_1, cl_2, cd, alpha_0
    initial_guess = np.array([blade.cl_1, blade.cl_2, blade.cd, blade.alpha_0])*1.0
    print("Initial guess:", initial_guess)
    # Define the bounds for each parameter
    bounds = [(0.0, 10.0), (0.0, 10.0), (0.0, 5.0), (np.radians(0), np.radians(45))]

    trace = []
    step_counter = {"count": 0}  # mutable so it can persist across callback calls

    def record_and_print(xk):
        trace.append(xk.copy())
        step_counter["count"] += 1
        if step_counter["count"] % 1 == 0:
            print(f"Step {step_counter['count']}: x = {xk}")

    # Optimize the parameters using a minimization algorithm
    result = minimize(
        get_loss,
        initial_guess,
        args=(dataset, blade),
        bounds=bounds,
        callback=record_and_print,
        method='Nelder-Mead',
        options={'disp': True, 'maxiter': 200}
    )

    if result.success:
        fitted_params = result.x
        print("Fitted parameters:", fitted_params)
        return fitted_params
    else:
        print("Optimization failed:", result.message)
        return None



def plot_the_fit(data, dataset: data_factory.FittingDataset):
    params = parameters.PennStateARILab550()
    f_gt = [
        -params.m * parameters.Environment.g * np.array([0.0, 0.0, 1.0]) + params.m * dataset.dv[i]
        for i in range(len(dataset.dv))
    ]

    fig, axs = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    axs[0].plot(data["i"], [f[0] for f in data["f_total"]], label='Fitted Fx', linestyle='None', marker='.')
    axs[0].plot([f[0] for f in f_gt], label='GT Fx', linestyle='-', marker='.')
    axs[0].legend()
    axs[0].set_ylabel("Force X")

    axs[1].plot(data["i"], [f[1] for f in data["f_total"]], label='Fitted Fy', linestyle='None', marker='.')
    axs[1].plot([f[1] for f in f_gt], label='GT Fy', linestyle='-', marker='.')
    axs[1].legend()
    axs[1].set_ylabel("Force Y")

    axs[2].plot(data["i"], [f[2] for f in data["f_total"]], label='Fitted Fz', linestyle='None', marker='.')
    axs[2].plot([f[2] for f in f_gt], label='GT Fz', linestyle='-', marker='.')
    axs[2].legend()
    axs[2].set_ylabel("Force Z")
    axs[2].set_xlabel("Sample Index")

    fig.tight_layout()
    return fig
