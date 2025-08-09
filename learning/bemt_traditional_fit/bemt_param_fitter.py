import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.optimize import minimize
import matplotlib.pyplot as plt

from inflow_model.bet import BladeElementTheory
from inflow_model.blade_params import Blade
from inflow_model.propeller_lookup_table import PropellerLookupTable
import inflow_model.blade_params
import inflow_model.propeller_lookup_table as propeller_lookup_table
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

class BemtParamFitter:
    def __init__(self):
        self.blade = FittedBlade()
        self.bet_instance = BladeElementTheory(self.blade)
        self.params = parameters.PennStateARILab550()
        self.logger = {}
        self.sample_distance = None
        self.horizontal_weight = None
        self.fitted_params = None

    def initialize_logger(self):
        self.logger = {"i": [], "f_0": [], "f_1": [], "f_2": [], "f_3": [], "f_total": [], "f_inertial_frame_frd": []}

    def make_lookup_table(self, fitted_params):
        blade = inflow_model.blade_params.APC_8x6()
        blade.cl_1, blade.cl_2, blade.cd, blade.alpha_0 = fitted_params  # fitted parameters
        PropellerLookupTable.Maker.make_propeller_lookup_table("apc_8x6_fitted", blade,)

    def adjust_resolution(self, is_fine_tune):
        if is_fine_tune:
            # self.sample_distance = 20
            # self.horizontal_weight = 1.0
            # num_of_elements = 20
            # num_of_rotation_segments = 18
            self.sample_distance = 500
            self.horizontal_weight = 1.0
            num_of_elements = 100
            num_of_rotation_segments = 90
        else:
            self.sample_distance = 100
            self.horizontal_weight = 100        
            num_of_elements = 5
            num_of_rotation_segments = 6
        self.bet_instance.set_integration_resolution(num_of_elements, num_of_rotation_segments)

    def compute_total_force_inertial_frame_frd(self, dataset: data_factory.FittingDataset, i: int, can_log=False):
        f_0 = self.compute_model_thrust(dataset.u_free_0[i], 
                                dataset.v_forward_0[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_0[i], 
                                self.params.is_ccw_blade[0])
        # torque_0 = np.cross(dataset.relative_position_inertial_frame_0[i], f_0)

        f_1 = self.compute_model_thrust(dataset.u_free_1[i], 
                                dataset.v_forward_1[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_1[i], 
                                self.params.is_ccw_blade[1])
        # torque_1 = np.cross(dataset.relative_position_inertial_frame_1[i], f_1)

        f_2 = self.compute_model_thrust(dataset.u_free_2[i], 
                                dataset.v_forward_2[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_2[i], 
                                self.params.is_ccw_blade[2])
        # torque_2 = np.cross(dataset.relative_position_inertial_frame_2[i], f_2)

        f_3 = self.compute_model_thrust(dataset.u_free_3[i], 
                                dataset.v_forward_3[i], 
                                dataset.shared_r_disk[i], 
                                dataset.omega_3[i], 
                                self.params.is_ccw_blade[3])
        # torque_3 = np.cross(dataset.relative_position_inertial_frame_3[i], f_3)

        f = f_0 + f_1 + f_2 + f_3
        f_inertial_frame_frd = utils.FrdFluConverter.flip_vector(dataset.shared_r_disk[i]@f)

        if can_log:
            self.logger["i"].append(i)
            self.logger["f_0"].append(f_0)
            self.logger["f_1"].append(f_1)
            self.logger["f_2"].append(f_2)
            self.logger["f_3"].append(f_3)
            self.logger["f_total"].append(f)
            self.logger["f_inertial_frame_frd"].append(f_inertial_frame_frd)
        return f_inertial_frame_frd

    def compute_total_force_inertial_frame_frd_with_lookup_table(self, dataset: data_factory.FittingDataset, i: int, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader, can_log=False):
        f_0 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_0[i],
            dataset.v_forward_0[i],
            dataset.shared_r_disk[i],
            dataset.omega_0[i],
            self.params.is_ccw_blade[0],
            lookup_table
        )

        f_0_debug = self.compute_model_thrust(
            dataset.u_free_0[i],
            dataset.v_forward_0[i],
            dataset.shared_r_disk[i],
            dataset.omega_0[i],
            self.params.is_ccw_blade[0]
        )


        f_1 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_1[i],
            dataset.v_forward_1[i],
            dataset.shared_r_disk[i],
            dataset.omega_1[i],
            self.params.is_ccw_blade[1],
            lookup_table
        )
        
        f_1_debug = self.compute_model_thrust(
            dataset.u_free_1[i],
            dataset.v_forward_1[i],
            dataset.shared_r_disk[i],
            dataset.omega_1[i],
            self.params.is_ccw_blade[1]
        )


        f_2 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_2[i],
            dataset.v_forward_2[i],
            dataset.shared_r_disk[i],
            dataset.omega_2[i],
            self.params.is_ccw_blade[2],
            lookup_table
        )

        f_2_debug = self.compute_model_thrust(
            dataset.u_free_2[i],
            dataset.v_forward_2[i],
            dataset.shared_r_disk[i],
            dataset.omega_2[i],
            self.params.is_ccw_blade[2]
        )


        f_3 = BemtParamFitter.compute_thrust_with_lookup_table(
            dataset.u_free_3[i],
            dataset.v_forward_3[i],
            dataset.shared_r_disk[i],
            dataset.omega_3[i],
            self.params.is_ccw_blade[3],
            lookup_table
        )

        f_3_debug = self.compute_model_thrust(
            dataset.u_free_3[i],
            dataset.v_forward_3[i],
            dataset.shared_r_disk[i],
            dataset.omega_3[i],
            self.params.is_ccw_blade[3]
        )

        print(f"dataset.u_free_0[i]: {dataset.u_free_0[i]}, dataset.v_forward_0[i]: {dataset.v_forward_0[i]}, dataset.shared_r_disk[i]: {dataset.shared_r_disk[i]}, dataset.omega_0[i]: {dataset.omega_0[i]}, is_ccw_blade: {self.params.is_ccw_blade[0]}")
        print(f"dataset.u_free_1[i]: {dataset.u_free_1[i]}, dataset.v_forward_1[i]: {dataset.v_forward_1[i]}, dataset.shared_r_disk[i]: {dataset.shared_r_disk[i]}, dataset.omega_1[i]: {dataset.omega_1[i]}, is_ccw_blade: {self.params.is_ccw_blade[1]}")
        print(f"dataset.u_free_2[i]: {dataset.u_free_2[i]}, dataset.v_forward_2[i]: {dataset.v_forward_2[i]}, dataset.shared_r_disk[i]: {dataset.shared_r_disk[i]}, dataset.omega_2[i]: {dataset.omega_2[i]}, is_ccw_blade: {self.params.is_ccw_blade[2]}")
        print(f"dataset.u_free_3[i]: {dataset.u_free_3[i]}, dataset.v_forward_3[i]: {dataset.v_forward_3[i]}, dataset.shared_r_disk[i]: {dataset.shared_r_disk[i]}, dataset.omega_3[i]: {dataset.omega_3[i]}, is_ccw_blade: {self.params.is_ccw_blade[3]}")

        print(f"f_0: {f_0}, f_0_debug: {dataset.shared_r_disk[i] @ f_0_debug}")
        print(f"f_1: {f_1}, f_1_debug: {dataset.shared_r_disk[i] @ f_1_debug}")
        print(f"f_2: {f_2}, f_2_debug: {dataset.shared_r_disk[i] @ f_2_debug}")
        print(f"f_3: {f_3}, f_3_debug: {dataset.shared_r_disk[i] @ f_3_debug}")

        f_inertial_frame_flu = f_0 + f_1 + f_2 + f_3
        f_inertial_frame_frd = utils.FrdFluConverter.flip_vector(f_inertial_frame_flu)
        return f_inertial_frame_frd

    def compute_residual_force(self, f_inertial_frame_frd, a_groundtruth):
        """Compute the residual force between the model thrust and the ground truth thrust.
        Assumes a_groundtruth is in the inertial frame (FRD).
        """
        f_gt_inertial_frame = -self.params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) + self.params.m * a_groundtruth
        f_residual = f_inertial_frame_frd - f_gt_inertial_frame
        print(f"f_inertial_frame_frd: {f_inertial_frame_frd}")
        print(f"f_gt_inertial_frame: {f_gt_inertial_frame}")
        return f_residual

    def get_residual_force(self, dataset: data_factory.FittingDataset, i: int, can_log=False, lookup_table=None, is_using_lookup_table=False):
        if is_using_lookup_table:
            f_inertial_frame_frd = self.compute_total_force_inertial_frame_frd_with_lookup_table(dataset, i, lookup_table, can_log)
        else:
            f_inertial_frame_frd = self.compute_total_force_inertial_frame_frd(dataset, i, can_log)
        f_residual = self.compute_residual_force(f_inertial_frame_frd, dataset.dv[i])
        return f_residual

    def compute_model_thrust(self, u_free, v_forward, r_disk, omega, is_ccw_blade):
        if is_ccw_blade:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        else:
            f_x, f_y, f_z, v_i = self.bet_instance.get_rotor_forces(u_free, v_forward, r_disk, -omega, is_ccw_blade) # negative omega for CW rotation, this is an interface mismatch
        return np.array([f_x, f_y, f_z])

    @staticmethod
    def compute_thrust_with_lookup_table(u_free, v_forward, r_disk, omega, is_ccw_blade, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
        thrust, _ = lookup_table.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
        return thrust
    
    def get_loss(self, x, datasets: list[data_factory.FittingDataset], can_log=False, lookup_table=None, is_using_lookup_table=False):
        
        self.blade.cl_1, self.blade.cl_2, self.blade.cd, self.blade.alpha_0 = x 

        loss = 0.0
        for dataset in datasets:
            self.initialize_logger()
            data_len = len(dataset.u_free_0)
            loss_per_data_set = 0.0
            num_of_samples_per_dataset = data_len // self.sample_distance
            for i in range(0, data_len, self.sample_distance):
                f_residual = self.get_residual_force(dataset, i, can_log, lookup_table, is_using_lookup_table)
                loss_f = self.horizontal_weight*(f_residual[0]**2 + f_residual[1]**2) + f_residual[2]**2
                loss_per_data_set += loss_f
            loss_per_data_set = loss_per_data_set / num_of_samples_per_dataset
            loss += loss_per_data_set
            if can_log:
                self.plot_the_fit(dataset)
                self.initialize_logger()
        loss = loss / len(datasets)
        print(f"Current loss: {loss}")
        return loss

    def fit_params(self, datasets: list[data_factory.FittingDataset], blade: FittedBlade, initial_guess=None, is_fine_tune=False):
        cl_1_range = [3.0, 4.0, 5.0, 6.0]
        cl_2_range = [1.0, 1.5, 2.0]
        cd_range = [1.0, 1.5, 2.0]
        alpha_0_range = [np.radians(15), np.radians(20), np.radians(25)]
        if initial_guess is None:
            # Initial guess for cl_1, cl_2, cd, alpha_0
            initial_guess = np.array([5.0, 1.5, 1.5, 0.3])*1.5
        print("Initial guess:", initial_guess)
        # Define the bounds for each parameter
        bounds = [(2.0, 10.0), (0.0, 5.0), (0.0, 5.0), (np.radians(10), np.radians(40))]
        if is_fine_tune:
            maxiter = 50
        else:
            maxiter = 200
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
            args=(datasets, blade, is_fine_tune),
            bounds=bounds,
            callback=record_and_print,
            method='Nelder-Mead',
            options={'disp': True, 
                    'maxiter': maxiter,
                    'fatol': 1e-1}
        )

        if result.success:
            fitted_params = result.x
            print("Fitted parameters:", fitted_params)
            return fitted_params
        else:
            print("Optimization failed:", result.message)
            return None

    def plot_the_fit(self, dataset):
        f_total_intertial_frame_gt = [
            (-self.params.m * parameters.Environment.g * np.array([0.0, 0.0, 1.0]) + self.params.m * dataset.dv[i])
            for i in range(len(dataset.dv))
        ]
        f_0_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_0_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
            ]
        f_1_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_1_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_1_f_rotor_inertial_frame))
            ]
        f_2_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_2_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_2_f_rotor_inertial_frame))
            ]
        f_3_body_frame_gt = [
            dataset.shared_r_disk[i].T@dataset.rotor_3_f_rotor_inertial_frame[i]    # back to body frame/disk frame
            for i in range(len(dataset.rotor_3_f_rotor_inertial_frame))
            ]
        f_total_from_rotor_inertial_frame_gt = [
            utils.FrdFluConverter.flip_vector(dataset.rotor_0_f_rotor_inertial_frame[i] + dataset.rotor_1_f_rotor_inertial_frame[i] + \
            dataset.rotor_2_f_rotor_inertial_frame[i] + dataset.rotor_3_f_rotor_inertial_frame[i])
            for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
            ]
        
        fig0, axs0 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

        force_labels = ["Force X", "Force Y", "Force Z"]
        for i in range(3):
            axs0[i].plot(self.logger["i"], [f[i] for f in self.logger["f_inertial_frame_frd"]], label=f'Fitted F{["x","y","z"][i]}', linestyle='None', marker='.')
            axs0[i].plot([f[i] for f in f_total_intertial_frame_gt], label=f'GT F{["x","y","z"][i]}', linestyle='-', marker='.')
            axs0[i].plot([f[i] for f in f_total_from_rotor_inertial_frame_gt], label=f'GT from Rotor F{["x","y","z"][i]}', linestyle='-', marker='.')
            axs0[i].legend()
            axs0[i].set_ylabel(force_labels[i])
        axs0[2].set_xlabel("Sample Index")

        fig0.tight_layout()

        fig1, axs1 = plt.subplots(4, 3, figsize=(12, 8), sharex=True)

        for i in range(3):
            axs1[0, i].plot(self.logger["i"], [f[i] for f in self.logger["f_0"]], label='Fitted F0', linestyle='None', marker='.')
            axs1[0, i].plot([f[i] for f in f_0_body_frame_gt], label='GT F0', linestyle='-')
            axs1[1, i].plot(self.logger["i"], [f[i] for f in self.logger["f_1"]], label='Fitted F1', linestyle='None', marker='.')
            axs1[1, i].plot([f[i] for f in f_1_body_frame_gt], label='GT F1', linestyle='-')
            axs1[2, i].plot(self.logger["i"], [f[i] for f in self.logger["f_2"]], label='Fitted F2', linestyle='None', marker='.')
            axs1[2, i].plot([f[i] for f in f_2_body_frame_gt], label='GT F2', linestyle='-')
            axs1[3, i].plot(self.logger["i"], [f[i] for f in self.logger["f_3"]], label='Fitted F3', linestyle='None', marker='.')
            axs1[3, i].plot([f[i] for f in f_3_body_frame_gt], label='GT F3', linestyle='-')
            axs1[0, i].set_ylabel(f"F0 {['X', 'Y', 'Z'][i]}")
            axs1[1, i].set_ylabel(f"F1 {['X', 'Y', 'Z'][i]}")
            axs1[2, i].set_ylabel(f"F2 {['X', 'Y', 'Z'][i]}")
            axs1[3, i].set_ylabel(f"F3 {['X', 'Y', 'Z'][i]}")

        return fig0, fig1






























def compute_model_thrust(u_free, v_forward, r_disk, omega, is_ccw_blade, bet_instance: BladeElementTheory):
    if is_ccw_blade:
        f_x, f_y, f_z, v_i = bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
    else:
        f_x, f_y, f_z, v_i = bet_instance.get_rotor_forces(u_free, v_forward, r_disk, -omega, is_ccw_blade) # negative omega for CW rotation, this is an interface mismatch
    return np.array([f_x, f_y, f_z])

def get_loss(x, datasets: list[data_factory.FittingDataset], blade: FittedBlade, is_fine_tune=False, is_in_validation=False):
    params = parameters.PennStateARILab550()
    cl_1, cl_2, cd, alpha_0 = x
    blade.cl_1 = cl_1
    blade.cl_2 = cl_2
    blade.cd = cd
    blade.alpha_0 = alpha_0    
    if is_in_validation or is_fine_tune:
        sample_distance = 500
        num_of_elements = 100
        num_of_rotation_segments = 90
        horizontal_weight = 1.0
    else:
        sample_distance = 100
        num_of_elements = 5
        num_of_rotation_segments = 6
        horizontal_weight = 100
    bet_instance = BladeElementTheory(blade, num_of_elements=num_of_elements, num_of_rotation_segments=num_of_rotation_segments)

    loss = 0.0
    for dataset in datasets:
        if is_in_validation:
            data = {"i": [], "f_0": [], "f_1": [], "f_2": [], "f_3": [], "f_total": [], "f_inertial_frame_frd": []}
        data_len = len(dataset.u_free_0)
        loss_per_data_set = 0.0
        num_of_samples_per_dataset = data_len // sample_distance
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
            f_inertial_frame_frd = utils.FrdFluConverter.flip_vector(dataset.shared_r_disk[i]@f)
            if is_in_validation:
                data["i"].append(i)
                data["f_0"].append(f_0)
                data["f_1"].append(f_1)
                data["f_2"].append(f_2)
                data["f_3"].append(f_3)
                data["f_total"].append(f)
                data["f_inertial_frame_frd"].append(f_inertial_frame_frd)
            f_gt_inertial_frame = -params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) + params.m * dataset.dv[i]
            f_gt_inertial_frame = utils.FrdFluConverter.flip_vector(f_gt_inertial_frame)  # flip to FRD frame
            f_gt_body_frame = dataset.shared_r_disk[i].T@f_gt_inertial_frame
            f_residual = f - f_gt_body_frame
            # loss_f = np.linalg.norm(f_inertial_frame_frd + params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) - params.m * dataset.dv[i])**2
            loss_f = horizontal_weight*(f_residual[0]**2 + f_residual[1]**2) + f_residual[2]**2
            loss_per_data_set += loss_f
        loss_per_data_set = loss_per_data_set / num_of_samples_per_dataset
        loss += loss_per_data_set
        if is_in_validation:
            plot_the_fit(data, dataset)
    loss = loss / len(datasets)
    print(f"Current loss: {loss}")
    return loss

def fit_params(datasets: list[data_factory.FittingDataset], blade: FittedBlade, initial_guess=None, is_fine_tune=False):
    if initial_guess is None:
        # Initial guess for cl_1, cl_2, cd, alpha_0
        initial_guess = np.array([blade.cl_1, blade.cl_2, blade.cd, blade.alpha_0])*1.5
    print("Initial guess:", initial_guess)
    # Define the bounds for each parameter
    bounds = [(2.0, 10.0), (0.0, 5.0), (0.0, 5.0), (np.radians(10), np.radians(40))]
    if is_fine_tune:
        maxiter = 50
    else:
        maxiter = 200
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
        args=(datasets, blade, is_fine_tune),
        bounds=bounds,
        callback=record_and_print,
        method='Nelder-Mead',
        options={'disp': True, 
                 'maxiter': maxiter,
                 'fatol': 1e-1}
    )

    if result.success:
        fitted_params = result.x
        print("Fitted parameters:", fitted_params)
        return fitted_params
    else:
        print("Optimization failed:", result.message)
        return None

def compute_thrust_with_lookup_table(u_free, v_forward, r_disk, omega, is_ccw_blade, lookup_table: propeller_lookup_table.PropellerLookupTable.Reader):
    # The output is in inertial frame
    thrust, _ = lookup_table.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade)
    return thrust

def test_fitted_lookup_table(datasets: list[data_factory.FittingDataset], lookup_table: propeller_lookup_table.PropellerLookupTable.Reader, is_in_validation=True):
    params = parameters.PennStateARILab550()
    loss = 0.0
    loss_x = 0.0
    loss_y = 0.0
    loss_z = 0.0
    sample_distance = 1
    horizontal_weight = 1.0
    for dataset in datasets:
        if is_in_validation:
            data = {"i": [], "f_0": [], "f_1": [], "f_2": [], "f_3": [], "f_total": [], "f_inertial_frame_frd": []}
        data_len = len(dataset.u_free_0)
        loss_per_data_set = 0.0
        loss_x_per_data_set = 0.0
        loss_y_per_data_set = 0.0
        loss_z_per_data_set = 0.0
        for i in range(0, data_len, sample_distance):
            f_0 = compute_thrust_with_lookup_table(dataset.u_free_0[i], 
                                    dataset.v_forward_0[i], 
                                    dataset.shared_r_disk[i], 
                                    dataset.omega_0[i], 
                                    params.is_ccw_blade[0],
                                    lookup_table)

            f_1 = compute_thrust_with_lookup_table(dataset.u_free_1[i], 
                                    dataset.v_forward_1[i], 
                                    dataset.shared_r_disk[i], 
                                    dataset.omega_1[i], 
                                    params.is_ccw_blade[1],
                                    lookup_table)

            f_2 = compute_thrust_with_lookup_table(dataset.u_free_2[i], 
                                    dataset.v_forward_2[i], 
                                    dataset.shared_r_disk[i], 
                                    dataset.omega_2[i], 
                                    params.is_ccw_blade[2],
                                    lookup_table)

            f_3 = compute_thrust_with_lookup_table(dataset.u_free_3[i], 
                                    dataset.v_forward_3[i], 
                                    dataset.shared_r_disk[i], 
                                    dataset.omega_3[i], 
                                    params.is_ccw_blade[3],
                                    lookup_table)

            f_0 = dataset.shared_r_disk[i].T@f_0
            f_1 = dataset.shared_r_disk[i].T@f_1
            f_2 = dataset.shared_r_disk[i].T@f_2
            f_3 = dataset.shared_r_disk[i].T@f_3
            f = f_0 + f_1 + f_2 + f_3
            f_inertial_frame_frd = utils.FrdFluConverter.flip_vector(dataset.shared_r_disk[i]@f)
            if is_in_validation:
                data["i"].append(i)
                data["f_0"].append(f_0)
                data["f_1"].append(f_1)
                data["f_2"].append(f_2)
                data["f_3"].append(f_3)
                data["f_total"].append(f)
                data["f_inertial_frame_frd"].append(f_inertial_frame_frd)
            f_gt_inertial_frame = -params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) + params.m * dataset.dv[i]
            f_gt_inertial_frame = utils.FrdFluConverter.flip_vector(f_gt_inertial_frame)  # flip to FRD frame
            f_gt_body_frame = dataset.shared_r_disk[i].T@f_gt_inertial_frame
            f_residual = f - f_gt_body_frame
            # loss_f = np.linalg.norm(f_inertial_frame_frd + params.m * parameters.Environment.g*np.array([0.0, 0.0, 1.0]) - params.m * dataset.dv[i])**2
            loss_f = horizontal_weight*(f_residual[0]**2 + f_residual[1]**2) + f_residual[2]**2
            loss_f_x = horizontal_weight*f_residual[0]**2
            loss_f_y = horizontal_weight*f_residual[1]**2
            loss_f_z = f_residual[2]**2
            loss_per_data_set += loss_f
            loss_x_per_data_set += loss_f_x
            loss_y_per_data_set += loss_f_y
            loss_z_per_data_set += loss_f_z
        num_of_samples_per_dataset = data_len / sample_distance
        loss_per_data_set = loss_per_data_set / num_of_samples_per_dataset
        loss_x_per_data_set = loss_x_per_data_set / num_of_samples_per_dataset
        loss_y_per_data_set = loss_y_per_data_set / num_of_samples_per_dataset
        loss_z_per_data_set = loss_z_per_data_set / num_of_samples_per_dataset
        loss += loss_per_data_set
        loss_x += loss_x_per_data_set
        loss_y += loss_y_per_data_set
        loss_z += loss_z_per_data_set
        if is_in_validation:
            plot_the_fit(data, dataset)
    loss = loss / len(datasets)
    loss_x = loss_x / len(datasets)
    loss_y = loss_y / len(datasets)
    loss_z = loss_z / len(datasets)
    print(f"loss: {loss}, loss_x: {loss_x}, loss_y: {loss_y}, loss_z: {loss_z}")
    return loss

def plot_the_fit(data, dataset: data_factory.FittingDataset):
    params = parameters.PennStateARILab550()
    f_total_intertial_frame_gt = [
        (-params.m * parameters.Environment.g * np.array([0.0, 0.0, 1.0]) + params.m * dataset.dv[i])
        for i in range(len(dataset.dv))
    ]
    f_0_body_frame_gt = [
        dataset.shared_r_disk[i].T@dataset.rotor_0_f_rotor_inertial_frame[i]    # back to body frame/disk frame
        for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
        ]
    f_1_body_frame_gt = [
        dataset.shared_r_disk[i].T@dataset.rotor_1_f_rotor_inertial_frame[i]    # back to body frame/disk frame
        for i in range(len(dataset.rotor_1_f_rotor_inertial_frame))
        ]
    f_2_body_frame_gt = [
        dataset.shared_r_disk[i].T@dataset.rotor_2_f_rotor_inertial_frame[i]    # back to body frame/disk frame
        for i in range(len(dataset.rotor_2_f_rotor_inertial_frame))
        ]
    f_3_body_frame_gt = [
        dataset.shared_r_disk[i].T@dataset.rotor_3_f_rotor_inertial_frame[i]    # back to body frame/disk frame
        for i in range(len(dataset.rotor_3_f_rotor_inertial_frame))
        ]
    f_total_from_rotor_inertial_frame_gt = [
        utils.FrdFluConverter.flip_vector(dataset.rotor_0_f_rotor_inertial_frame[i] + dataset.rotor_1_f_rotor_inertial_frame[i] + \
        dataset.rotor_2_f_rotor_inertial_frame[i] + dataset.rotor_3_f_rotor_inertial_frame[i])
        for i in range(len(dataset.rotor_0_f_rotor_inertial_frame))
        ]
    
    fig0, axs0 = plt.subplots(3, 1, figsize=(12, 8), sharex=True)

    force_labels = ["Force X", "Force Y", "Force Z"]
    for i in range(3):
        axs0[i].plot(data["i"], [f[i] for f in data["f_inertial_frame_frd"]], label=f'Fitted F{["x","y","z"][i]}', linestyle='None', marker='.')
        axs0[i].plot([f[i] for f in f_total_intertial_frame_gt], label=f'GT F{["x","y","z"][i]}', linestyle='-', marker='.')
        axs0[i].plot([f[i] for f in f_total_from_rotor_inertial_frame_gt], label=f'GT from Rotor F{["x","y","z"][i]}', linestyle='-', marker='.')
        axs0[i].legend()
        axs0[i].set_ylabel(force_labels[i])
    axs0[2].set_xlabel("Sample Index")

    fig0.tight_layout()

    fig1, axs1 = plt.subplots(4, 3, figsize=(12, 8), sharex=True)

    for i in range(3):
        axs1[0, i].plot(data["i"], [f[i] for f in data["f_0"]], label='Fitted F0', linestyle='None', marker='.')
        axs1[0, i].plot([f[i] for f in f_0_body_frame_gt], label='GT F0', linestyle='-')
        axs1[1, i].plot(data["i"], [f[i] for f in data["f_1"]], label='Fitted F1', linestyle='None', marker='.')
        axs1[1, i].plot([f[i] for f in f_1_body_frame_gt], label='GT F1', linestyle='-')
        axs1[2, i].plot(data["i"], [f[i] for f in data["f_2"]], label='Fitted F2', linestyle='None', marker='.')
        axs1[2, i].plot([f[i] for f in f_2_body_frame_gt], label='GT F2', linestyle='-')
        axs1[3, i].plot(data["i"], [f[i] for f in data["f_3"]], label='Fitted F3', linestyle='None', marker='.')
        axs1[3, i].plot([f[i] for f in f_3_body_frame_gt], label='GT F3', linestyle='-')
        axs1[0, i].set_ylabel(f"F0 {['X', 'Y', 'Z'][i]}")
        axs1[1, i].set_ylabel(f"F1 {['X', 'Y', 'Z'][i]}")
        axs1[2, i].set_ylabel(f"F2 {['X', 'Y', 'Z'][i]}")
        axs1[3, i].set_ylabel(f"F3 {['X', 'Y', 'Z'][i]}")

    return fig0, fig1
