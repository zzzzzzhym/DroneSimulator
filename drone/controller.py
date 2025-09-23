import numpy as np
import warnings

import utils
import dynamics
import parameters as params
import trajectory
import disturbance_estimator
import inflow_model.propeller_lookup_table as propeller_lookup_table
import propeller

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
        self.force_motor_desired = np.array([0.0, 0.0, 0.0, 0.0])
        self.force_motor_available = np.array([0.0, 0.0, 0.0, 0.0])
        self.rotation_speed = np.array([0.0, 0.0, 0.0, 0.0])
        # self.disturbance_estimator = disturbance_estimator.DisturbanceEstimator("wind_near_wall_bemt_in_control_train_xz_wind", 0.01)
        self.disturbance_estimator = disturbance_estimator.DisturbanceEstimator("wind_near_wall_wo_bemt_in_control_far_from_wall", 0.01)
        self.f_disturb = np.array([0.0, 0.0, 0.0])
        self.torque_disturb = np.array([0.0, 0.0, 0.0])        
        self.baseline_disturbance_estimator = disturbance_estimator.BaselineDisturbanceEstimator(0.01)
        self.f_disturb_base = np.array([0.0, 0.0, 0.0])
        self.torque_disturb_base = np.array([0.0, 0.0, 0.0])
        # self.propeller_force_table = propeller_lookup_table.PropellerLookupTable.Reader("apc_8x6_with_trail")
        self.propeller_force_table = propeller_lookup_table.PropellerLookupTable.Reader("apc_8x6_fitted2")
        self.propeller = propeller.apc_8x6
        self.is_warmed_up = False
        self.warm_up_count = 0
        self.warm_up_count_max = 0
        self.is_using_baseline_disturbance_estimator = True   
        self.is_using_any_disturbance_estimator = True
        self.is_using_inflow_model = True
        print("DroneController: using inflow model: ", self.is_using_inflow_model)
        print("DroneController: using disturbance estimator: ", self.is_using_any_disturbance_estimator)
        print("DroneController: using baseline disturbance estimator: ", self.is_using_baseline_disturbance_estimator)

        # saturation parameters
        self.max_f_feedback = 6.0  # feedback force saturation
        self.max_f_disturb_compensation = 6.0  # disturbance compensation force saturation
        self.b_2d_norm_min = 1e-3   # minimum norm to avoid singularity
        self.max_torque = 2.0  # maximum torque
        
        # internal var logging
        self.torque_feedback = np.array([0.0, 0.0, 0.0])
        self.torque_coriolis = np.array([0.0, 0.0, 0.0])
        self.torque_feedforward = np.array([0.0, 0.0, 0.0])
        self.f_feedback = np.array([0.0, 0.0, 0.0])
        self.f_feedforward = np.array([0.0, 0.0, 0.0])
        self.f_disturb_compensation = np.array([0.0, 0.0, 0.0])
        self.f_disturb_sensed_raw = np.array([0.0, 0.0, 0.0])

    def step_controller(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference):
        self.step_tracking_error(state, ref)
        self.step_disturbance_estimator(state)
        # if not self.is_warmed_up:
        #     self.warm_up_count += 1
        #     # skip disturbance estimator at the first cycle to prevent large initial deviation
        #     self.f_disturb = np.array([0.0, 0.0, 0.0])
        #     self.torque_disturb = np.array([0.0, 0.0, 0.0]) 
        #     if self.warm_up_count > self.warm_up_count_max:
        #         self.is_warmed_up = True
        if self.is_warmed_up:
            # skip disturbance estimator at the first cycle to prevent large initial deviation
            self.step_disturbance_estimator(state)
        else:
            self.is_warmed_up = True        
        self.step_desired_force(state, ref)
        self.step_tracking_control(state)
        self.step_desired_pose(ref)
        self.step_attitude_error(state)
        self.step_attitude_control(state)
        self.step_error_function_so3(state)
        self.step_motor_output(state)

    def step_disturbance_estimator(self, state: dynamics.DroneDynamics):
        # tracking_error = np.zeros(6)  # assume no tracking error term in disturbance estimator
        tracking_error = np.hstack((self.e_v, np.zeros(3)))
        self.f_disturb_sensed_raw = self.get_disturbance(state)[0:3]
        self.disturbance_estimator.step_disturbance(
            state.state.position,
            state.state.v,
            state.state.q,
            state.state.omega,
            self.f[2],
            self.torque,
            np.array([state.rotors.rotors[0].rotation_speed, 
                      state.rotors.rotors[1].rotation_speed, 
                      state.rotors.rotors[2].rotation_speed, 
                      state.rotors.rotors[3].rotation_speed]),
            self.get_disturbance(state),
            tracking_error
        )
        self.f_disturb = np.array([
            self.disturbance_estimator.f_x.disturbance,
            self.disturbance_estimator.f_y.disturbance,
            self.disturbance_estimator.f_z.disturbance
        ])
        self.torque_disturb = np.array([
            self.disturbance_estimator.tq_x.disturbance,
            self.disturbance_estimator.tq_y.disturbance,
            self.disturbance_estimator.tq_z.disturbance
        ])
        self.baseline_disturbance_estimator.step_disturbance(
            self.get_disturbance(state),
            tracking_error
        )
        self.f_disturb_base = np.array([
            self.baseline_disturbance_estimator.f_x.disturbance,
            self.baseline_disturbance_estimator.f_y.disturbance,
            self.baseline_disturbance_estimator.f_z.disturbance
        ])
        self.torque_disturb_base = np.array([
            self.baseline_disturbance_estimator.tq_x.disturbance,
            self.baseline_disturbance_estimator.tq_y.disturbance,
            self.baseline_disturbance_estimator.tq_z.disturbance
        ])

    def get_disturbance(self, state: dynamics.DroneDynamics):
        """Disturbance force in body frame"""
        f_disturb = (
            self.f
            + state.state.pose.T @ (
            state.v_dot * self.params.m
            - self.params.m * params.Environment.g * np.array([0.0, 0.0, 1.0])
            )
        )

        tq_disturbance = (
            self.params.inertia @ state.omega_dot
            + utils.get_hat_map(state.state.omega) @ self.params.inertia @ state.state.omega
            - self.torque
        )

        # # Add noise
        # f_noise_std = np.array([0.1, 0.1, 0.1])*2
        # tq_noise_std = np.array([0.01, 0.01, 0.01])
        # f_noise = np.random.normal(0, f_noise_std)
        # tq_noise = np.random.normal(0, tq_noise_std)        
        # return np.hstack((f_disturb + f_noise, tq_disturbance + tq_noise))
    
        return np.hstack((f_disturb, tq_disturbance))

    def step_desired_force(self, state: dynamics.DroneDynamics, ref: trajectory.TrajectoryReference, can_sense_jerk: bool=False, can_plan_jerk: bool=False):
        f_feedback = -params.Control.k_x*self.e_x - params.Control.k_v*self.e_v
        f_feedback = utils.saturate_vector_norm(f_feedback, self.max_f_feedback)

        f_disturb_compensation = np.array([0.0, 0.0, 0.0])
        if self.is_using_any_disturbance_estimator:
            if self.is_using_baseline_disturbance_estimator:
                f_disturb_compensation = -state.state.pose@self.f_disturb_base # add disturbance force
            else:
                f_disturb_compensation = -state.state.pose@self.f_disturb # add disturbance force
        f_disturb_compensation = utils.saturate_vector_norm(f_disturb_compensation, self.max_f_disturb_compensation)

        e3 = np.array([0.0, 0.0, 1.0])
        f_feedforward = -self.params.m*params.Environment.g*e3 + self.params.m*ref.x_d_dot2
        self.f_d = f_feedback + f_feedforward + f_disturb_compensation
        if np.abs(self.f_d@self.f_d) < 0.0001:
            warnings.warn("DroneController: f_d too close to 0")
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
            
        # logging
        self.f_feedback = f_feedback
        self.f_feedforward = f_feedforward
        self.f_disturb_compensation = f_disturb_compensation

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

    def step_attitude_control(self, state: dynamics.DroneDynamics, can_use_omega_desired_dot: bool=False):
        yaw_weight = 0.2
        w = np.diag([1.0, 1.0, yaw_weight]) # Mz is weaker
        torque_feedback = -params.Control.k_r*self.e_r - params.Control.k_omega*self.e_omega
        torque_feedback = w@torque_feedback
        torque_coriolis = np.cross(state.state.omega, self.params.inertia@state.state.omega)
        if can_use_omega_desired_dot:
            torque_feedforward = - self.params.inertia@(
                utils.get_hat_map(state.state.omega)@state.state.pose.T@self.omega_desired - state.state.pose.T@self.omega_desired_dot)
        else:
            torque_feedforward = - self.params.inertia@(
                utils.get_hat_map(state.state.omega)@state.state.pose.T@self.omega_desired)
        torque_feedback = utils.saturate_vector_norm(torque_feedback, self.max_torque)
        torque_feedforward = utils.saturate_vector_norm(torque_feedforward, self.max_torque)
        self.torque = torque_feedback + torque_coriolis + torque_feedforward

        # torque saturation
        self.torque = utils.saturate_vector_norm(self.torque, self.max_torque)

        # logging
        self.torque_feedback = torque_feedback
        self.torque_coriolis = torque_coriolis
        self.torque_feedforward = torque_feedforward

    def step_desired_pose(self, ref: trajectory.TrajectoryReference):
        b_3d, b_3d_dot, b_3d_dot2 = utils.get_unit_vector_derivatives(-self.f_d,
                                                                      -self.f_d_dot, -self.f_d_dot2)
        self.b_3d = b_3d
        b_2d_unnormalized = np.cross(self.b_3d, ref.b_1d)
        norm_proj = np.linalg.norm(b_2d_unnormalized)
        if norm_proj < self.b_2d_norm_min:
            print('Warning: DroneController: b_2d_unnormalized too close to 0, use previous value')
            b_2d = self.b_2d
            b_2d_dot = np.zeros(3)
            b_2d_dot2 = np.zeros(3)
        else:
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
        '''The error function on SO(3) is defined as the angle between the desired and actual rotation matrix.
        debug purpose only, does not need to use in controller
        '''
        self.psi_r_rd = 0.5*(1 - self.pose_desired[:,0]@state.state.pose[:,0] +
                        1 - self.pose_desired[:,1]@state.state.pose[:,1] +
                        1 - self.pose_desired[:,2]@state.state.pose[:,2])
    
    def step_motor_output(self, state: dynamics.DroneDynamics):
        self.force_motor_desired = self.params.m_wrench_to_thrust@np.hstack((self.f[2], self.torque))
        if self.is_using_inflow_model:
            # with inflow model
            for i, thrust in enumerate(self.force_motor_desired):
                self.rotation_speed[i] = self.propeller_force_table.get_rotation_speed(
                    state.rotors.rotors[i].local_wind_velocity,
                    state.rotors.rotors[i].velocity_inertial_frame,
                    state.rotors.rotors[i].pose,
                    state.rotors.rotors[i].rotation_speed,
                    thrust
                )
        else:
            # without inflow model
            for i, thrust in enumerate(self.force_motor_desired):
                self.rotation_speed[i] = self.propeller_force_table.get_rotation_speed(
                    np.zeros(3),
                    np.zeros(3),
                    state.rotors.rotors[i].pose,
                    state.rotors.rotors[i].rotation_speed,
                    thrust
                )

    def get_control_output(self):
        """controller provides yaw torque and rotation speed because rotor yaw torque is not modeled"""
        return self.rotation_speed, self.torque[2]
