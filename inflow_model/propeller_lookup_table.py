import numpy as np
import os
import yaml
from typing import Optional
from scipy.interpolate import RegularGridInterpolator
import matplotlib.pyplot as plt

import bet
import blade_params
import warnings

class PropellerLookupTable:
    class Reader:
        def __init__(self, filename: str):
            self.omega_range = None # rad/s
            self.u_free_x_range = None  # m/s
            self.pitch_range = None # rad
            self.table = None
            self.interp_func = None   
            self.max_allowed_extrapolation = 1e-3  
            self.load_lookup_table(filename) 

        def read_data(self, filename: str):
            file_path = os.path.join(os.path.dirname(__file__), "lookup_table", filename + ".yaml")
            print("[PropellerLookupTable] Reading data from", os.path.relpath(file_path, os.getcwd()))
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File {file_path} not found.")
            with open(file_path, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
                self.omega_range = np.array(data["omega_range"])
                self.u_free_x_range = np.array(data["u_free_x_range"])
                self.pitch_range = np.array(data["pitch_range"])
                self.table = np.array(data["table"])
        
        def get_interpolator(self):
            self.interpolator = RegularGridInterpolator((self.u_free_x_range, self.pitch_range, self.omega_range), self.table)

        def query_rotation_speed(self, u_free_x: float, pitch: float, omega_current:float, thrust_desired: float):
            """From the controller perspective, the lookup table need to query rotation speed given required thrust."""
            # construct a omega-thrust table
            thrust_range = np.zeros(len(self.omega_range))
            for i, omega in enumerate(self.omega_range):
                queried_data = self.query_data_from_table(u_free_x, pitch, omega)
                f_z = queried_data[2]  # f_z is thrust in the lookup table frame
                thrust_range[i] = f_z

            thrust_min = np.min(thrust_range)
            thrust_max = np.max(thrust_range)
            thrust_clipped = np.clip(thrust_desired, thrust_min, thrust_max)  # clip thrust to the range of the table

            # Step 2: Find all bracket intervals and interpolate omega
            candidates = []
            for i in range(len(self.omega_range) - 1):
                thrust_left = thrust_range[i]
                thrust_right = thrust_range[i + 1]
                if (thrust_left - thrust_clipped) * (thrust_right - thrust_clipped) <= 0:
                    if abs(thrust_left - thrust_right) > 1e-3:
                        ratio = (thrust_clipped - thrust_left) / (thrust_right - thrust_left)
                        omega_interp = self.omega_range[i] + ratio * (self.omega_range[i + 1] - self.omega_range[i])
                        candidates.append(omega_interp)
                    else:   # thrust_left and thrust_right are very close, see if we can use omega_current or put the closest one as candidate
                        if omega_current < self.omega_range[i]:
                            candidates.append(self.omega_range[i])
                        elif omega_current > self.omega_range[i + 1]:
                            candidates.append(self.omega_range[i + 1])
                        else:
                            candidates.append(omega_current)

            # Step 3: Choose the one closest to omega_current
            if not candidates:
                raise ValueError(f"Cannot find a valid omega for thrust {thrust_desired} in the range [{thrust_min}, {thrust_max}]")
            closest = min(candidates, key=lambda omega: abs(omega - omega_current))
            return closest

        def get_rotation_speed(self, u_free: np.ndarray, v_forward: np.ndarray, r_disk: np.ndarray, omega_current:float, thrust_desired: float):
            """Get the rotation speed of the rotor given the desired thrust. The lookup table is in the disk frame, so we need to convert the inertial frame to the disk frame.

            Args:
                u_free (np.ndarray): free stream velocity in inertial frame
                v_forward (np.ndarray): forward velocity in inertial frame
                r_disk (np.ndarray): rotor disk pose in inertial frame
                omega_current (float): current rotation speed of the rotor in rad/s
                thrust_desired (float): desired thrust

            Returns:
                float: rotation speed of the rotor in rad/s
            """
            u_relative_wind = u_free - v_forward
            u_relative_norm = np.linalg.norm(u_relative_wind)
            if u_relative_norm < 1e-6: # relative wind is close to zero
                # use the rotor disk frame as the lookup table frame
                matrix_from_inertial_to_lookup_table = r_disk.T
                pitch = 0.0
            else:
                x_axis = u_relative_wind / u_relative_norm
                matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
                pitch = PropellerLookupTable.Reader.get_pitch_angle(r_disk, matrix_from_inertial_to_lookup_table)
            rotation_speed = self.query_rotation_speed(u_relative_norm, pitch, omega_current, thrust_desired)
            return rotation_speed

        def load_lookup_table(self, filename: str):
            self.read_data(filename)
            self.get_interpolator()

        def query_data_from_table(self, u_free_x: float, pitch: float, omega: float):
            """The forces are in the disk frame setup in the lookup table. It's z axis is the rotor axis, x axis is the free stream direction. The actual rotor frame may be different from this frame.

            Args:
                u_free_x (float): _description_
                pitch (float): _description_
                omega (float): _description_

            Returns:
                _type_: _description_
            """
            omega_clipped = np.clip(omega, self.omega_range[0], self.omega_range[-1])
            u_free_x_clipped = np.clip(u_free_x, self.u_free_x_range[0], self.u_free_x_range[-1])
            pitch_clipped = np.clip(pitch, self.pitch_range[0], self.pitch_range[-1])
            if np.abs(omega - omega_clipped) > self.max_allowed_extrapolation or \
            np.abs(u_free_x - u_free_x_clipped) > self.max_allowed_extrapolation or \
            np.abs(pitch - pitch_clipped) > self.max_allowed_extrapolation:
                warnings.warn(f"Warning: Interpolating outside the range:\n"
                              f"u_free_x [m/s]: {u_free_x} (clipped: {u_free_x_clipped}),\n"
                              f"pitch [deg]: {pitch*180/np.pi} (clipped: {pitch_clipped*180/np.pi}),\n"
                              f"omega: {omega} (clipped: {omega_clipped})")
            return self.interpolator((u_free_x_clipped, pitch_clipped, omega_clipped))
        
        def get_rotor_forces(self, u_free: np.ndarray, v_forward: np.ndarray, r_disk: np.ndarray, omega: float, is_ccw_blade: bool) -> tuple[np.ndarray, np.ndarray]:
            """_summary_

            Args:
                u_free (np.ndarray): free stream velocity in inertial frame
                v_forward (np.ndarray): forward velocity in inertial frame
                r_disk (np.ndarray): rotor disk pose in inertial frame
                omega (float): rotation speed of the rotor in rad/s
                is_ccw_blade (bool): True if the blade rotates counter-clockwise from bird view (opposite to z axis body frame)

            Returns:
                tuple[np.ndarray, np.ndarray]: forces in inertial frame and induced velocity in inertial frame
            """
            u_relative_wind = u_free - v_forward
            u_relative_norm = np.linalg.norm(u_relative_wind)
            if u_relative_norm < 1e-6: # relative wind is close to zero
                # use the rotor disk frame as the lookup table frame
                matrix_from_inertial_to_lookup_table = r_disk.T
                matrix_from_lookup_table_to_inertial = r_disk
                pitch = 0.0
            else:
                x_axis = u_relative_wind / u_relative_norm
                matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial = PropellerLookupTable.Reader.get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis, r_disk)
                pitch = PropellerLookupTable.Reader.get_pitch_angle(r_disk, matrix_from_inertial_to_lookup_table)
            queried_data = self.query_data_from_table(u_relative_norm, pitch, omega)
            forces_in_lookup_table_disk_frame = queried_data[:3]  # f_x, f_y, f_z
            v_i_in_lookup_table_disk_frame = np.array([0.0, 0.0, -queried_data[3]])
            if not is_ccw_blade:
                forces_in_lookup_table_disk_frame[1] *= -1  # because wind is always in the x direction, flip y is enough to mirror ccw to cw
            forces_in_lookup_table_frame = PropellerLookupTable.Reader.convert_vector_from_table_disk_to_table(pitch, forces_in_lookup_table_disk_frame)
            forces_in_inertial_frame = matrix_from_lookup_table_to_inertial @ forces_in_lookup_table_frame
            v_i_in_lookup_table_frame = PropellerLookupTable.Reader.convert_vector_from_table_disk_to_table(pitch, v_i_in_lookup_table_disk_frame)
            v_i_in_inertial_frame = matrix_from_lookup_table_to_inertial @ v_i_in_lookup_table_frame
            return forces_in_inertial_frame, v_i_in_inertial_frame
        
        @staticmethod
        def get_rotation_matrix_between_inertial_and_lookup_table_frame(x_axis_by_wind: np.ndarray, r_disk: np.ndarray):
            """compute transformation matrix between the inertial frame and the lookup table frame. 

            Args:
                x_axis_by_wind (np.ndarray): wind direction in the inertial frame defined x axis
                r_disk (np.ndarray): rotor frame in inertial frame

            Returns:
                np.ndarray: transformation matrix between the inertial frame and the lookup table frame. 
            """
            z_disk = r_disk[:, 2]   # this is a 1D array
            y_axis_raw = np.cross(z_disk, x_axis_by_wind)
            y_norm = np.linalg.norm(y_axis_raw)
            if y_norm < 1e-6:   # x_axis and z_disk are parallel
                # use the rotor frame y axis as the lookup table frame y axis
                y_axis = r_disk[:, 1]
            else:
                y_axis = y_axis_raw / y_norm
            z_axis = np.cross(x_axis_by_wind, y_axis)
            matrix_from_inertial_to_lookup_table = np.array([x_axis_by_wind, y_axis, z_axis])
            matrix_from_lookup_table_to_inertial = matrix_from_inertial_to_lookup_table.T
            return matrix_from_inertial_to_lookup_table, matrix_from_lookup_table_to_inertial
                
        @staticmethod
        def get_pitch_angle(r_disk: np.ndarray, matrix_from_inertial_to_lookup_table: np.ndarray):
            """pitch in radian"""
            z_axis_of_disk_in_inertial = r_disk[:, 2]
            z_axis_of_disk_in_lookup_table = matrix_from_inertial_to_lookup_table @ z_axis_of_disk_in_inertial
            pitch = np.arctan2(z_axis_of_disk_in_lookup_table[0], z_axis_of_disk_in_lookup_table[2])
            return pitch
        
        @staticmethod
        def convert_vector_from_table_disk_to_table(pitch: float, force_in_table_disk: np.ndarray):
            """Convert the force from the rotor disk frame to the lookup table frame. 

            Args:
                pitch (float): pitch angle of the rotor disk frame
                force (np.ndarray): force in the rotor disk frame

            Returns:
                np.ndarray: force in the lookup table frame
            """
            matrix_from_disk_to_table = np.array([[np.cos(pitch), 0, -np.sin(pitch)],   # x axis of the table disk 
                                                  [0, 1, 0],                            # y axis of the table disk
                                                  [np.sin(pitch), 0, np.cos(pitch)]]).T # z axis of the table disk
            return matrix_from_disk_to_table @ force_in_table_disk
        
        @staticmethod
        def plot_rotor_force(r_disk: np.ndarray, force: np.ndarray, v_i: np.ndarray):
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.quiver(0, 0, 0, r_disk[0, 0], r_disk[1, 0], r_disk[2, 0], color='r', linestyle='dashed')
            ax.quiver(0, 0, 0, r_disk[0, 1], r_disk[1, 1], r_disk[2, 1], color='g', linestyle='dashed')
            ax.quiver(0, 0, 0, r_disk[0, 2], r_disk[1, 2], r_disk[2, 2], color='b', linestyle='dashed')
            ax.quiver(0, 0, 0, force[0], force[1], force[2], color='orange', label='Force')
            ax.quiver(0, 0, 0, v_i[0], v_i[1], v_i[2], color='purple', label='Induced Velocity')
            ax.legend()
            limit = 2
            ax.set_xlim([-limit, limit])
            ax.set_ylim([-limit, limit])
            ax.set_zlim([-limit, limit])
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            print(f"f_x: {force[0]}, f_y: {force[1]}, f_z: {force[2]}")
            return fig

    class Maker:
        """This class makes a lookup table for propeller forces. This is a 3D table with 
            - u_free_x: free stream velocity in x direction, 
            - pitch: pitch angle of the rotor disk frame, 
            - omega: angular velocity of the rotor.
            Lookup table frame definition: the free stream is always in the x direciton. Direction of free stream and the rotor disk z axis forms the x-z plane.
            The lookup table only store the data for the counter-clockwise blade. The data for the clockwise blade can be obtained by flipping the sign of F_y, while keeping F_x and F_z the same.
        """
        _DEFAULT_U_FREE_X_RANGE = (0, 1, 2, 3, 4, 5, 7, 10, 13, 15, 17, 20) # m/s
        _DEFAULT_PITCH_RANGE = tuple(np.deg2rad([-90, -60, -45, -30, -15, 0, 15, 30, 45, 60, 90]))  # rad
        _DEFAULT_OMEGA_RANGE = (0, 200, 300, 400, 500, 600, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150, 1200, 1300, 1400, 1500, 1600, 1800, 2000, 2600)  # rad/s
        
        def __init__(self):
            self.blade = None
            self.omega_range = None # rad/s
            self.u_free_x_range = None  # m/s
            self.pitch_range = None # rad
            self.table = None
            self.interp_func = None   
            self.max_allowed_extrapolation = 1e-3    

        @staticmethod
        def make_propeller_lookup_table(filename: str, 
                                        blade: Optional[blade_params.Blade] = None, 
                                        omega_range: Optional[np.ndarray] = None, 
                                        u_free_x_range: Optional[np.ndarray] = None, 
                                        pitch_range: Optional[np.ndarray] = None):
            
            blade = blade if blade is not None else blade_params.APC_8x6()
            omega_range = omega_range if omega_range is not None else np.array(PropellerLookupTable.Maker._DEFAULT_OMEGA_RANGE)
            u_free_x_range = u_free_x_range if u_free_x_range is not None else np.array(PropellerLookupTable.Maker._DEFAULT_U_FREE_X_RANGE)
            pitch_range = pitch_range if pitch_range is not None else np.array(PropellerLookupTable.Maker._DEFAULT_PITCH_RANGE)
            table = np.zeros((len(u_free_x_range), len(pitch_range), len(omega_range), 4))   # 4 for f_x, f_y, f_z, v_i

            print("[PropellerLookupTable] Making lookup table:")
            print("Omega range:", omega_range)
            print("u_free_x range:", u_free_x_range)
            print("Pitch range:", pitch_range)        
            v_forward = np.array([0, 0, 0])
            bet_instance = bet.BladeElementTheory(blade)
            r_disk_range = [bet_instance.pitch_rotor_disk_along_y_axis(pitch) for pitch in pitch_range]
            for i, u_free_x in enumerate(u_free_x_range):
                u_free = np.array([u_free_x, 0, 0])
                for j, r_disk in enumerate(r_disk_range):
                    for k, omega in enumerate(omega_range):
                        f_x, f_y, f_z, v_i = bet_instance.get_rotor_forces(u_free, v_forward, r_disk, omega, is_ccw_blade=True)
                        table[i, j, k, :] = [f_x, f_y, f_z, v_i]
                        print(f"u_free_x: {u_free_x:.4f}, pitch: {np.rad2deg(pitch_range[j]):.4f}, omega: {omega:.4f}, forces: ({f_x:.4f}, {f_y:.4f}, {f_z:.4f}), v_i: {v_i:.4f}")
            PropellerLookupTable.Maker.save_data(filename, u_free_x_range, pitch_range, omega_range, table)

        @staticmethod
        def save_data(filename: str, u_free_x_range: np.ndarray, pitch_range: np.ndarray, omega_range: np.ndarray, table: np.ndarray):
            file_path = os.path.join(os.path.dirname(__file__), "lookup_table", filename + ".yaml")
            print("Saving data to", os.path.relpath(file_path, os.getcwd()))
            data = {"omega_range": omega_range.tolist(), "u_free_x_range": u_free_x_range.tolist(), "pitch_range": pitch_range.tolist(), "table": table.tolist()}
            with open(file_path, 'w') as file:
                yaml.dump(data, file)
        
    
