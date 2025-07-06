import numpy as np
import matplotlib.pyplot as plt

import torch
torch.set_default_dtype(torch.float32)

from blade_params import Blade, APC_8x6, APC_8x6_OfficialData
import airfoil.aero_coeff as aero_coeff 
from airfoil.air import Air
import vi_learnable_param

class BladeElementTheory:
    """a hybrid model that combines the BEMT model and the BET model to simulate the propeller loading condition in forward flight.
    Reference:
        Gill, Rajan, and Raffaello D'andrea. "Propeller thrust and drag in forward flight." 
        2017 IEEE Conference on control technology and applications (CCTA). IEEE, 2017.
    """    
    def __init__(self, blade: Blade):
        self.device = None
        self.blade = blade
        self.num_of_elements = 100
        self.num_of_rotation_segments = 90
        self.coeff = aero_coeff.Coeffecients(cl_1=self.blade.cl_1, cl_2=self.blade.cl_2, alpha_0=self.blade.alpha_0, cd=self.blade.cd, cd_0=self.blade.cd_0)
        self.dy = self.blade.y_max/self.num_of_elements
        self.disk_area = torch.pi*(self.blade.y_max**2 - self.blade.y_min**2)  # rotor disk area

    def get_v_flow_disk_frame(self, u_free: torch.Tensor, v_i: torch.Tensor, v_forward: torch.Tensor, r_disk: torch.Tensor):
        """get the relative wind speed in the disk frame. It accounts for the forward speed of the disk.

        Args:
            u_free (torch.Tensor): free stream velocity in the inertial frame
            v_i (torch.Tensor): inflow velocity in rotation disk frame, perpendicular to the disk, positive when flow in -z direction
            v_forward (torch.Tensor): disk velocity in the inertial frame
            r_disk (torch.Tensor): disk rotation matrix, should have a fixed relative pose to drone body, while the drone is using SAE coordinate system, the disk is using ISO coordinate system

        Returns:
            torch.Tensor: 1D array of the relative wind speed in the disk frame
        """
        v_relative_inertial_frame = u_free - v_forward
        v_i_vector_form = torch.cat([torch.zeros(2, device=self.device), v_i.unsqueeze(0)])
        v_flow_disk_frame = r_disk.T@v_relative_inertial_frame + v_i_vector_form
        return v_flow_disk_frame

    def get_relative_v_to_blade_section(self, v_flow_disk_frame: torch.Tensor, psi_blade_angle: torch.Tensor, y: torch.Tensor, omega_blade: torch.Tensor):
        """convention in the blade rotation frame:
        the x axis is along the blade, the y axis is perpendicular to the blade, the z axis is along the blade rotation axis clockwise rotation is positive
        when psi_blade_angle is 0, the blade rotation frame overlaps with the disk frame
        
        Args:
            v_flow_disk_frame (torch.Tensor): wind blowing speed to the disk
            psi_blade_angle (torch.Tensor): rotation angle of blade
            y (torch.Tensor): _description_
            omega_blade (torch.Tensor): rotation speed of the blade, counter clockwise is positive

        Returns:
            u_t: relative wind speed perpendicular to the blade (in y direction), when blade rotates in positive direction without wind, it generates negative u_t
            u_p: relative wind speed in z direction
        """
        cos_psi = torch.cos(psi_blade_angle)
        sin_psi = torch.sin(psi_blade_angle)

        # v_x_blade_frame = v_flow_disk_frame[:2]@torch.tensor([cos_psi, sin_psi])
        direction = torch.stack([-sin_psi, cos_psi])  # shape: (2,)
        v_y_blade_frame = torch.dot(v_flow_disk_frame[:2], direction)
        u_p = v_flow_disk_frame[2]
        u_t = -v_y_blade_frame + omega_blade*y
        alpha_flow = torch.atan2(u_p, u_t)
        return u_t, u_p, alpha_flow

    def get_attack_angle_to_blade(self, alpha_flow, y):
        attack_angle = alpha_flow + self.blade.get_blade_pitch(y)
        return attack_angle

    def get_blade_element_force_in_airfoil_frame(self, u_t: torch.Tensor, u_p: torch.Tensor, attack_angle: torch.Tensor, y: torch.Tensor):
        if attack_angle.item() > torch.pi / 3 or attack_angle.item() < -torch.pi / 3:   # protection from backward flow
            # warnings.warn(f"attack angle is out of range, attack_angle: {torch.rad2deg(attack_angle)} [deg]")
            lift = torch.tensor(0.0, device=self.device)
            drag = torch.tensor(0.0, device=self.device)
        else:
            u = torch.sqrt(u_t**2 + u_p**2)
            chord = self.blade.get_chord(y)
            cl = self.coeff.get_cl(attack_angle)
            cd = self.coeff.get_cd(attack_angle, u, chord)
            lift = 0.5*Air.rho*u**2*cl*chord*self.dy 
            drag = 0.5*Air.rho*u**2*cd*chord*self.dy 
        return lift, drag

    def convert_force_from_airfoil_to_blade_frame(self, lift: torch.Tensor, drag: torch.Tensor, alpha_flow: torch.Tensor):
        f_z = lift*torch.cos(alpha_flow) + drag*torch.sin(alpha_flow)
        f_y = lift*torch.sin(alpha_flow) - drag*torch.cos(alpha_flow)
        return f_z, f_y
    
    def get_force_in_blade_frame_ccw_blade(self, v_flow_disk_frame: torch.Tensor, psi_blade_angle: torch.Tensor, y: torch.Tensor, omega_blade: torch.Tensor):
        """assumes ccw blade"""
        u_t, u_p, alpha_flow = self.get_relative_v_to_blade_section(v_flow_disk_frame, psi_blade_angle, y, omega_blade)
        attack_angle = self.get_attack_angle_to_blade(alpha_flow, y)
        lift, drag = self.get_blade_element_force_in_airfoil_frame(u_t, u_p, attack_angle, y)
        f_z_blade, f_y_blade = self.convert_force_from_airfoil_to_blade_frame(lift, drag, alpha_flow)
        return f_z_blade, f_y_blade

    def get_force_in_blade_frame(self, v_flow_disk_frame: torch.Tensor, psi_blade_angle: torch.Tensor, y: torch.Tensor, omega_blade: torch.Tensor, is_ccw_blade=True):
        if is_ccw_blade:
            f_z_blade, f_y_blade = self.get_force_in_blade_frame_ccw_blade(v_flow_disk_frame, psi_blade_angle, y, omega_blade)
        else:
            # mirror the flow, rotation and the blade direction along x axis to create an equivalent ccw blade problem
            v_flow_mirrored_disk_frame = v_flow_disk_frame.clone()
            v_flow_mirrored_disk_frame[1] = -v_flow_mirrored_disk_frame[1]
            f_z_blade, f_y_blade = self.get_force_in_blade_frame_ccw_blade(v_flow_mirrored_disk_frame, -psi_blade_angle, y, -omega_blade)
            f_y_blade = -f_y_blade  
        return f_z_blade, f_y_blade

    def convert_force_from_blade_to_disk_frame(self, f_z_blade: torch.Tensor, f_y_blade: torch.Tensor, psi_blade_angle: torch.Tensor):
        f_x = -f_y_blade*torch.sin(psi_blade_angle)
        f_y = f_y_blade*torch.cos(psi_blade_angle)
        f_z = f_z_blade
        return f_x, f_y, f_z

    def integrate_element_force_over_one_revolution_per_blade(self, y: torch.Tensor, v_flow_disk_frame: torch.Tensor, omega_blade: torch.Tensor, is_ccw_blade=True):
        f_z_per_rev = torch.tensor(0.0, device=self.device)
        f_y_per_rev = torch.tensor(0.0, device=self.device)
        f_x_per_rev = torch.tensor(0.0, device=self.device)
        for j in range(self.num_of_rotation_segments):
            psi_blade_angle = torch.tensor(j*2*torch.pi/self.num_of_rotation_segments - torch.pi, device=self.device)
            f_z_blade, f_y_blade = self.get_force_in_blade_frame(v_flow_disk_frame, psi_blade_angle, y, omega_blade, is_ccw_blade)
            f_x_disk_frame, f_y_disk_frame, f_z_disk_frame = self.convert_force_from_blade_to_disk_frame(f_z_blade, f_y_blade, psi_blade_angle)
            f_z_per_rev += f_z_disk_frame
            f_y_per_rev += f_y_disk_frame
            f_x_per_rev += f_x_disk_frame
        divisor = torch.tensor(self.num_of_rotation_segments, device=self.device)
        f_z_per_rev /= divisor
        f_y_per_rev /= divisor
        f_x_per_rev /= divisor
        return f_x_per_rev, f_y_per_rev, f_z_per_rev
    
    def integrate_element_force(
        self,
        u_free: torch.Tensor,
        v_i: float,
        v_forward: torch.Tensor,
        r_disk: torch.Tensor,
        omega_blade: torch.Tensor,
        is_ccw_blade: bool = True,
        can_plot: bool = False
    ):

        f_z_total = torch.tensor(0.0, device=self.device)
        f_y_total = torch.tensor(0.0, device=self.device)
        f_x_total = torch.tensor(0.0, device=self.device)

        if can_plot:
            # data[:, 0] = f_x_disk_frame
            # data[:, 1] = f_y_disk_frame
            # data[:, 2] = f_z_disk_frame
            # data[:, 3] = psi_blade_angle
            # data[:, 4] = y
            # data[:, 5] = x_disk_frame
            # data[:, 6] = y_disk_frame
            data = torch.zeros((self.num_of_elements * self.num_of_rotation_segments, 7), device=self.device)
            i_log = 0

        v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        for i in range(self.num_of_elements):
            y = torch.tensor(i * self.dy, device=self.device)
            f_z_per_rev = torch.tensor(0.0, device=self.device)
            f_y_per_rev = torch.tensor(0.0, device=self.device)
            f_x_per_rev = torch.tensor(0.0, device=self.device)

            for j in range(self.num_of_rotation_segments):
                psi_blade_angle = torch.tensor(
                    j * 2 * torch.pi / self.num_of_rotation_segments - torch.pi,
                    device=self.device
                )
                f_z_blade, f_y_blade = self.get_force_in_blade_frame(v_flow_disk_frame, psi_blade_angle, y, omega_blade, is_ccw_blade)
                f_x_disk_frame, f_y_disk_frame, f_z_disk_frame = self.convert_force_from_blade_to_disk_frame(f_z_blade, f_y_blade, psi_blade_angle)
                f_z_per_rev += f_z_disk_frame
                f_y_per_rev += f_y_disk_frame
                f_x_per_rev += f_x_disk_frame
                if can_plot:
                    data[i_log, 0] = f_x_disk_frame
                    data[i_log, 1] = f_y_disk_frame
                    data[i_log, 2] = f_z_disk_frame
                    data[i_log, 3] = psi_blade_angle
                    data[i_log, 4] = y
                    data[i_log, 5] = y*torch.cos(psi_blade_angle)
                    data[i_log, 6] = y*torch.sin(psi_blade_angle)
                    i_log += 1

            divisor = torch.tensor(self.num_of_rotation_segments, device=self.device)
            f_z_total += f_z_per_rev / divisor
            f_y_total += f_y_per_rev / divisor
            f_x_total += f_x_per_rev / divisor
        f_z_total = f_z_total*self.blade.num_of_blades
        f_y_total = f_y_total*self.blade.num_of_blades
        f_x_total = f_x_total*self.blade.num_of_blades
        if can_plot:
            self.plot_force_distribution(data.cpu().numpy())
        return torch.stack([f_x_total, f_y_total, f_z_total])

    def plot_force_distribution(self, data: np.ndarray):
        interval = 5
        fig0 = plt.figure()
        fig1 = plt.figure()
        fig2 = plt.figure()
        ax0 = fig0.add_subplot(111, projection='3d')
        ax1 = fig1.add_subplot(111, projection='3d')
        ax2 = fig2.add_subplot(111, projection='3d')
        ax0.set_title('f_x')
        ax1.set_title('f_y')
        ax2.set_title('f_z')
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax0.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 0])
        ax1.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 1])
        ax2.scatter(data[::interval, 5], data[::interval, 6], data[::interval, 2])
        fig3 = plt.figure()
        interval = 10
        ax3 = fig3.add_subplot(111, projection='3d')
        ax3.quiver(data[::interval, 5], data[::interval, 6], 0, data[::interval, 0], data[::interval, 1], data[::interval, 2], length=0.01, normalize=True)

    def get_dm_dot(self, v_flow_disk_frame: torch.Tensor, y: torch.Tensor):
        # the differential mass flow rate
        # v_flow_disk_frame already considers inflow velocity, thus the equation appears to be different from the paper
        dm_dot = Air.rho*2*torch.pi*y*self.dy*torch.sqrt(v_flow_disk_frame[0]**2 + v_flow_disk_frame[1]**2 + v_flow_disk_frame[2]**2)
        return dm_dot

    def get_differential_thrust_from_momentum(self, v_flow_disk_frame: torch.Tensor, v_i: torch.Tensor, y: torch.Tensor):
        dm_dot = self.get_dm_dot(v_flow_disk_frame, y)
        d_thrust = 2*v_i*dm_dot
        return d_thrust

    def get_thrust_difference(self, v_i: torch.Tensor, y: torch.Tensor, u_free: torch.Tensor, v_forward: torch.Tensor, r_disk: torch.Tensor, omega_blade: float, is_ccw_blade=True):
        v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
        df_bet = self.integrate_element_force_over_one_revolution_per_blade(y, v_flow_disk_frame, omega_blade, is_ccw_blade)
        d_thrust_bet = self.blade.num_of_blades * df_bet[2] 
        d_thrust_bemt = self.get_differential_thrust_from_momentum(v_flow_disk_frame, v_i, y)
        thrust_diff = d_thrust_bet - d_thrust_bemt
        return thrust_diff
    
    def guess_initial_v_i(self, y: float, omega_blade: float, is_ccw_blade=True):
        """Make initial guess based on static thrust model."""
        rpm_to_rad_per_sec = 2*torch.pi/60

        thrust_lbf = np.interp(
            omega_blade/rpm_to_rad_per_sec,
            APC_8x6_OfficialData.OMEGA_APC8X6_OFFICIAL_DATA_RPM,
            APC_8x6_OfficialData.THRUST_APC8X6_OFFICIAL_DATA_LBF
        )
        pound_force_to_newton = 4.44822
        thrust = thrust_lbf *pound_force_to_newton

        inch_to_m = 0.0254
        radius = 8*inch_to_m/2  # 8 inch diameter propeller
        area = torch.pi*radius**2  # rotor disk area
        v_i = torch.sqrt(torch.tensor(thrust/(2*Air.rho*area), dtype=torch.float32))  # static thrust model
        return  v_i

    def solve_v_i(self, y: float, u_free: torch.Tensor, v_forward: torch.Tensor, r_disk: torch.Tensor, omega_blade: torch.Tensor, is_ccw_blade=True):
        v_i_0 = self.guess_initial_v_i(y, omega_blade, is_ccw_blade)
        v_i_model = vi_learnable_param.LearnableInflowSpeed(v_i_0)

        optimizer = torch.optim.Adam(v_i_model.parameters(), lr=1e-2)

        # Adam
        # for _ in range(100):
        #     optimizer.zero_grad()
        #     loss = self.get_thrust_difference(v_i_model(), y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade)**2
        #     loss.backward()
        #     optimizer.step()

        # L-BFGS
        optimizer = torch.optim.LBFGS(
            [v_i_model.v_i],
            lr=1.0,
            max_iter=15,
            line_search_fn='strong_wolfe'
        )

        def closure():
            optimizer.zero_grad()
            loss = self.get_thrust_difference(
                v_i_model(), y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade
            ) ** 2
            loss.backward()
            return loss

        optimizer.step(closure)
        return v_i_model().detach()  # final solved inflow as tensor

    def get_rotor_forces(self, u_free: torch.Tensor, v_forward: torch.Tensor, r_disk: torch.Tensor, omega_blade: torch.Tensor, is_ccw_blade=True):
        """Calculate rotor forces in the disk frame.

        Args:
            u_free (torch.Tensor): Free stream velocity in the inertial frame.
            v_forward (torch.Tensor): Disk velocity in the inertial frame.
            r_disk (torch.Tensor): Disk rotation matrix.
            omega_blade (torch.Tensor): Rotation speed of the blade in rad/s.
            is_ccw_blade (bool, optional): Indicates if the blade is counter-clockwise. Defaults to True.

        Returns:
            tuple: Forces in x, y, and z directions in the disk frame.
        """

        if u_free.device != v_forward.device or u_free.device != r_disk.device or u_free.device != omega_blade.device:
            print("u_free, v_forward, r_disk and omega_blade not on the same device. Using u_free device.")
            v_forward = v_forward.to(u_free.device)
            r_disk = r_disk.to(u_free.device)
            omega_blade = omega_blade.to(u_free.device)
        self.device = u_free.device

        f_x = torch.tensor(0.0, device=self.device)
        f_y = torch.tensor(0.0, device=self.device)
        f_z = torch.tensor(0.0, device=self.device)
        v_i_disk = torch.tensor(0.0, device=self.device)
        for i in range(self.num_of_elements):
            y = torch.tensor(i*self.dy + self.blade.y_min, device=self.device)
            v_i = self.solve_v_i(y, u_free, v_forward, r_disk, omega_blade, is_ccw_blade)
            v_flow_disk_frame = self.get_v_flow_disk_frame(u_free, v_i, v_forward, r_disk)
            df_bet = self.integrate_element_force_over_one_revolution_per_blade(y, v_flow_disk_frame, omega_blade, is_ccw_blade)
            f_x += df_bet[0]
            f_y += df_bet[1]
            f_z += df_bet[2]
            v_i_disk += v_i*self.dy*y*np.pi*2
        v_i_disk = v_i_disk/self.disk_area
        f_z = f_z*self.blade.num_of_blades
        f_y = f_y*self.blade.num_of_blades
        f_x = f_x*self.blade.num_of_blades
        return f_x, f_y, f_z, v_i_disk

    @staticmethod
    def pitch_rotor_disk_along_y_axis(pitch_angle: float):
        r_disk = torch.tensor([[torch.cos(pitch_angle), 0, torch.sin(pitch_angle)],
                           [0, 1, 0],
                           [-torch.sin(pitch_angle), 0, torch.cos(pitch_angle)]])
        return r_disk.to(torch.float32)

