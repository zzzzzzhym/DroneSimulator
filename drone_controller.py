import numpy as np
import drone_utils as utils
import drone_dynamics as dynamics
import drone_parameters as params
import drone_utils as utils
import drone_trajectory as trajectory


class DroneController:
    def __init__(self) -> None:
        self.e_x = np.array([0.0, 0.0, 0.0])
        self.e_v = np.array([0.0, 0.0, 0.0])
        self.e_r = np.array([0.0, 0.0, 0.0])
        self.e_omega = np.array([0.0, 0.0, 0.0])
        self.pose_desired = np.identity(3)
        self.pose_desired_dot = np.identity(3)
        self.pose_desired_dot2 = np.identity(3)
        self.omega_desired = np.array([0.0, 0.0, 0.0])
        self.omega_desired_dot = np.array([0.0, 0.0, 0.0])
        self.b_3d = np.array([0.0, 0.0, 0.0])
        self.b_2d = np.array([0.0, 0.0, 0.0])
        self.f = np.array([0.0, 0.0, 0.0])
        self.f_dot = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        self.f_d = np.array([0.0, 0.0, 0.0])
        self.f_d_dot = np.array([0.0, 0.0, 0.0])

    def step_controller(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference):
        self.step_tracking_error(state, ref)
        self.step_desired_force(ref)
        self.step_tracking_control(state)
        self.step_desired_pose(ref)
        self.step_attitude_error(state)
        self.step_attitude_control(state)

    def step_desired_force(self, ref: trajectory.TrajectoryReference):
        """
        to do: zero protection f_d == 0 
        """
        e3 = np.array([0.0, 0.0, 1.0])
        self.f_d = -(-params.k_x*self.e_x - params.k_v *
                     self.e_v - params.m*params.g*e3 + params.m*ref.x_d_dot2)
        if np.abs(self.f_d@self.f_d) < 0.0001:
            print('Warning: DroneController: f_d too close to 0')
        e_a = self.f/params.m - ref.x_d_dot2
        self.f_d_dot = -(-params.k_x*self.e_v - params.k_v *
                         e_a + params.m*ref.x_d_dot3)
        e_j = self.f_dot/params.m - ref.x_d_dot3
        self.f_d_dot2 = -(-params.k_x*e_a - params.k_v *
                          e_j + params.m*ref.x_d_dot4)

    def step_tracking_control(self, state: dynamics.DroneDynamics):
        """
        project desired force on b3
        """
        self.f = self.f_d@state.pose[:, 2]*np.array([0.0, 0.0, 1.0])
        self.f_dot = self.f_d_dot@state.pose[:, 2]*np.array([0.0, 0.0, 1.0]) + \
            self.f_d@np.cross(state.omega, state.pose[:, 2])

    def step_attitude_control(self, state: dynamics.DroneDynamics):
        self.toruqe = -params.k_r*self.e_r - params.k_omega*self.e_omega + np.cross(state.omega, params.inertia@state.omega) - params.inertia@(
            utils.get_hat_map(state.omega)@state.pose.T@self.pose_desired@self.omega_desired - state.pose.T@self.pose_desired@self.omega_desired_dot)

    def step_desired_pose(self, ref: trajectory.TrajectoryReference):
        (b_3d, b_3d_dot, b_3d_dot2) = utils.get_unit_vector_derivatives(-self.f_d, -
                                                                        self.f_d_dot, -self.f_d_dot2)
        self.b_3d = b_3d
        b_2d_unnormalized = np.cross(self.b_3d, ref.b_1d)
        b_2d_unnormalized_dot = np.cross(
            b_3d_dot, ref.b_1d) + np.cross(self.b_3d, ref.b_1d_dot)
        b_2d_unnormalized_dot2 = np.cross(b_3d_dot2, ref.b_1d) + 2*np.cross(
            b_3d_dot, ref.b_1d_dot) + np.cross(self.b_3d, ref.b_1d_dot2)
        (b_2d, b_2d_dot, b_2d_dot2) = utils.get_unit_vector_derivatives(
            b_2d_unnormalized, b_2d_unnormalized_dot, b_2d_unnormalized_dot2)
        self.b_2d = b_2d
        b2d_x_b3d_unnormalized = np.cross(self.b_2d, self.b_3d)
        b2d_x_b3d_unnormalized_dot = np.cross(
            b_2d_dot, self.b_3d) + np.cross(self.b_2d, b_3d_dot)
        b2d_x_b3d_unnormalized_dot2 = np.cross(b_2d_dot2, self.b_3d) + 2*np.cross(
            b_2d_dot, b_3d_dot) + np.cross(self.b_2d, b_3d_dot2)
        (b2d_x_b3d, b2d_x_b3d_dot, b2d_x_b3d_dot2) = utils.get_unit_vector_derivatives(
            b2d_x_b3d_unnormalized, b2d_x_b3d_unnormalized_dot, b2d_x_b3d_unnormalized_dot2)

        self.pose_desired = np.array([b2d_x_b3d,
                                      self.b_2d,
                                      self.b_3d]).T
        self.pose_desired_dot = np.array([b2d_x_b3d_dot,
                                          b_2d_dot,
                                          b_3d_dot]).T
        self.pose_desired_dot2 = np.array([b2d_x_b3d_dot2,
                                           b_2d_dot2,
                                           b_3d_dot2]).T
        omega_desired_hat = self.pose_desired_dot@self.pose_desired.T
        self.omega_desired = utils.get_vee_map(omega_desired_hat)
        omega_desired_dot_hat = self.pose_desired_dot2@self.pose_desired.T + \
            self.pose_desired_dot@self.pose_desired_dot.T
        self.omega_desired_dot = utils.get_vee_map(omega_desired_dot_hat)

    def step_tracking_error(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference):
        self.e_x = state.position - ref.x_d
        self.e_v = state.v - ref.v_d

    def step_attitude_error(self, state: dynamics.DroneDynamics):
        self.e_r = utils.get_vee_map(
            0.5*(np.transpose(self.pose_desired)@state.pose - np.transpose(state.pose)@self.pose_desired))
        self.e_omega = state.omega - state.pose.T@self.pose_desired@self.omega_desired

if __name__ == "__main__":
    pass