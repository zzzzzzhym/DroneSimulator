import numpy as np

import utils
import dynamics
import parameters as params
import trajectory
import disturbance_estimator

class DroneController:
    def __init__(self, drone: params.Drone) -> None:
        self.params = drone
        self.e_x = np.array([0.0, 0.0, 0.0])
        self.e_v = np.array([0.0, 0.0, 0.0])
        self.e_a = np.array([0.0, 0.0, 0.0])
        self.e_j = np.array([0.0, 0.0, 0.0])
        self.e_r = np.array([0.0, 0.0, 0.0])
        self.e_omega = np.array([0.0, 0.0, 0.0])
        self.pose_desired = np.eye(3)  # intertial frame
        self.pose_desired_dot = np.eye(3)  # intertial frame
        self.pose_desired_dot2 = np.eye(3) # intertial frame
        self.omega_desired = np.array([0.0, 0.0, 0.0])
        self.omega_desired_dot = np.array([0.0, 0.0, 0.0])
        self.b_3d = np.array([0.0, 0.0, 0.0])
        self.b_2d = np.array([0.0, 0.0, 0.0])
        self.f = np.array([0.0, 0.0, 0.0])  # propulsion force, positive in -z direction of body frame. same convention as the paper
        self.f_dot = np.array([0.0, 0.0, 0.0])
        self.torque = np.array([0.0, 0.0, 0.0])
        self.f_d = np.array([0.0, 0.0, 0.0])    # desired force in inertial frame
        self.f_d_dot = np.array([0.0, 0.0, 0.0])
        self.psi_r_rd = 0.0 # debug use only
        self.force_motor = np.array([0.0, 0.0, 0.0, 0.0])
        self.disturbance_estimator = disturbance_estimator.DisturbanceEstimator("wall_effect", 0.01)
        self.f_disturb = np.array([0.0, 0.0, 0.0])
        self.torque_disturb = np.array([0.0, 0.0, 0.0])        
        self.baseline_disturbance_estimator = disturbance_estimator.BaselineDisturbanceEstimator(0.01)
        self.f_disturb_base = np.array([0.0, 0.0, 0.0])
        self.torque_disturb_base = np.array([0.0, 0.0, 0.0])

    def step_controller(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference):
        self.step_tracking_error(state, ref)
        self.step_disturbance_estimator(state)
        self.step_desired_force(state, ref)
        self.step_tracking_control(state)
        self.step_desired_pose(ref)
        self.step_attitude_error(state)
        self.step_attitude_control(state)
        self.step_error_function_so3(state)
        self.step_motor_force()

    def step_disturbance_estimator(self, state: dynamics.DroneDynamics):
        # tracking_error = np.zeros(6)  # assume no tracking error term in disturbance estimator
        tracking_error = np.hstack((self.e_v, np.zeros(3)))
        self.disturbance_estimator.step_disturbance(state.state.v, state.state.q, self.force_motor, self.get_disturbance(state), tracking_error)
        self.f_disturb = np.array([self.disturbance_estimator.f_x.disturbance, self.disturbance_estimator.f_y.disturbance, self.disturbance_estimator.f_z.disturbance])
        self.torque_disturb = np.array([self.disturbance_estimator.tq_x.disturbance, self.disturbance_estimator.tq_y.disturbance, self.disturbance_estimator.tq_z.disturbance])
        self.baseline_disturbance_estimator.step_disturbance(self.get_disturbance(state), tracking_error)
        self.f_disturb_base = np.array([self.baseline_disturbance_estimator.f_x.disturbance, self.baseline_disturbance_estimator.f_y.disturbance, self.baseline_disturbance_estimator.f_z.disturbance])
        self.torque_disturb_base = np.array([self.baseline_disturbance_estimator.tq_x.disturbance, self.baseline_disturbance_estimator.tq_y.disturbance, self.baseline_disturbance_estimator.tq_z.disturbance])

    def get_disturbance(self, state: dynamics.DroneDynamics):
        """Distrubance force in body frame"""
        f_disturb = self.f + state.state.pose.T@(state.v_dot*self.params.m - self.params.m*params.Environment.g*np.array([0.0, 0.0, 1.0]))
        tq_disturbance = self.params.inertia@state.omega_dot + utils.get_hat_map(state.state.omega)@self.params.inertia@state.state.omega - self.torque
        return np.hstack((f_disturb, tq_disturbance))

    def step_desired_force(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference, can_sense_jerk: bool=False, can_plan_jerk: bool=True):
        """
        to do: zero protection f_d == 0 
        """
        e3 = np.array([0.0, 0.0, 1.0])
        self.f_d = (-params.Control.k_x*self.e_x - params.Control.k_v *
                     self.e_v - self.params.m*params.Environment.g*e3 + self.params.m*ref.x_d_dot2)
        self.f_d += -state.state.pose@self.f_disturb # add disturbance force
        if np.abs(self.f_d@self.f_d) < 0.0001:
            print('Warning: DroneController: f_d too close to 0')
            self.f_d = -0.0001*e3    # z positive points down
        self.e_a = (state.state.pose@(-self.f) + self.params.m*params.Environment.g*e3)/self.params.m - ref.x_d_dot2
        if can_plan_jerk:
            self.f_d_dot = (-params.Control.k_x*self.e_v - params.Control.k_v *
                            self.e_a + self.params.m*ref.x_d_dot3)
        else:
            self.f_d_dot = (-params.Control.k_x*self.e_v - params.Control.k_v *
                            self.e_a)
        if can_sense_jerk:
            self.e_j = state.state.pose@(-self.f_dot)/self.params.m - ref.x_d_dot3
        else:
            self.e_j = np.array([0.0, 0.0, 0.0])
        if can_plan_jerk:
            self.f_d_dot2 = (-params.Control.k_x*self.e_a - params.Control.k_v *
                            self.e_j + self.params.m*ref.x_d_dot4)
        else:
            self.f_d_dot2 = (-params.Control.k_x*self.e_a - params.Control.k_v *
                            self.e_j)

    def step_tracking_control(self, state: dynamics.DroneDynamics):
        """
        project desired force on b3 to generate control input force
        """
        self.f = self.f_d@(-state.state.pose[:, 2])*np.array([0.0, 0.0, 1.0])
        self.f_dot = self.f_d_dot@(-state.state.pose[:, 2])*np.array([0.0, 0.0, 1.0]) + \
            self.f_d@np.cross(state.state.pose@state.state.omega, -state.state.pose[:, 2])*np.array([0.0, 0.0, 1.0]) + \
            self.f_d@(-state.state.pose[:, 2])*np.cross(state.state.omega, np.array([0.0, 0.0, 1.0]))
        '''
        Or equivalently
        self.f_dot = state.pose.T@(self.f_d_dot@(-state.pose[:, 2])*(state.pose[:, 2]) + \
            self.f_d@np.cross(state.pose@state.omega, -state.pose[:, 2])*(state.pose[:, 2]) + \
            self.f_d@(-state.pose[:, 2])*np.cross(state.pose@state.omega, state.pose[:, 2]))
        '''

    def step_attitude_control(self, state: dynamics.DroneDynamics):
        self.torque = -params.Control.k_r*self.e_r - params.Control.k_omega*self.e_omega + np.cross(state.state.omega, self.params.inertia@state.state.omega) - self.params.inertia@(
            utils.get_hat_map(state.state.omega)@state.state.pose.T@self.omega_desired - state.state.pose.T@self.omega_desired_dot)

    def step_desired_pose(self, ref: trajectory.TrajectoryReference):
        b_3d, b_3d_dot, b_3d_dot2 = utils.get_unit_vector_derivatives(-self.f_d,
                                                                      -self.f_d_dot, -self.f_d_dot2)
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
        self.e_x = state.state.position - ref.x_d
        self.e_v = state.state.v - ref.v_d

    def step_attitude_error(self, state: dynamics.DroneDynamics):
        self.e_r = utils.get_vee_map(
            0.5*(self.pose_desired.T@state.state.pose - state.state.pose.T@self.pose_desired))
        self.e_omega = state.state.omega - state.state.pose.T@self.omega_desired

    def step_error_function_so3(self, state: dynamics.DroneDynamics):
        '''
        debug purpose only, does not need to use in controller
        '''
        self.psi_r_rd = 0.5*(1 - self.pose_desired[:,0]@state.state.pose[:,0] +
                        1 - self.pose_desired[:,1]@state.state.pose[:,1] +
                        1 - self.pose_desired[:,2]@state.state.pose[:,2])
    
    def step_motor_force(self, can_saturate=True):
        f_motor_raw = self.params.m_wrench_to_thrust@np.hstack((self.f[2], self.torque))
        if can_saturate:
            self.force_motor = np.clip(f_motor_raw, a_max=self.params.f_motor_max, a_min=self.params.f_motor_min)
            state = self.params.m_thrust_to_wrench@self.force_motor
            self.f[2] = state[0]
            self.torque = state[1:]
        else:
            self.force_motor = f_motor_raw

    def step_motor_force_wip(self):
        """this could be formulated to a linear programming problem TBD"""
        f_motor_raw = self.params.m_wrench_to_thrust@np.hstack((self.f[2], self.torque))
        if np.max(np.abs(f_motor_raw)) > self.params.f_motor_max:
            print("Full commanded motor force saturated!")
            f_torque_only = self.params.m_wrench_to_thrust@np.hstack((0.0, self.torque))
            f_torque_only = self.get_saturated_motor_force(f_torque_only)
            df = self.params.f_motor_max - np.max(np.abs(f_torque_only))
            self.force_motor = f_torque_only + df
        else:
            self.force_motor = f_motor_raw
        state = self.params.m_thrust_to_wrench@self.force_motor
        self.f[2] = state[0]
        self.torque = state[1:]

    def get_saturated_motor_force(self, f_raw):
        f_max = np.max(np.abs(f_raw))
        if f_max > self.params.f_motor_max:
            coeff = self.params.f_motor_max/f_max
            f_saturated = coeff*f_raw
        else:
            f_saturated = f_raw
        return f_saturated

if __name__ == "__main__":
    pass
