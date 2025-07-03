import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import plot_utils
import drone.utils as utils

class Plotter:
    def __init__(self, t_span, dt):
        self.t_span = t_span
        self.dt = dt

    def make_plots(self, logger_output: np.ndarray, has_animation=False):
        # plot
        # self.plot_position_and_derivatives(logger_output)
        # self.plot_omega_and_derivatives(logger_output)
        # self.plot_pose_and_derivatives(logger_output)
        # self.plot_quaternion(logger_output)
        self.plot_trajectory(logger_output)
        self.plot_force_and_torque(logger_output)
        self.plot_position_tracking_error(logger_output)
        self.plot_pose_tracking_error(logger_output)
        # self.plot_desired_force(logger_output)
        # self.plot_control_input_force(logger_output)
        self.plot_pose_desired(logger_output)
        self.plot_omega_desired(logger_output)
        self.plot_rotor(logger_output)
        self.plot_disturbance_force(logger_output)
        self.plot_3d_trace(logger_output)
        if has_animation:
            self.ani = self.animate_pose(logger_output)

    def plot_position_and_derivatives(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_and_derivatives')
        axs[0, 0].plot(self.t_span, logger["position"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["position"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["position"][:, 2], marker='x')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, logger["v"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["v"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["v"][:, 2], marker='x')
        t_diff, position_diff = utils.get_signal_derivative(self.t_span, logger["position"], self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].set_ylabel('v_z')
        axs[0, 2].plot(self.t_span, logger["dv"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, logger["dv"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, logger["dv"][:, 2], marker='x')
        t_diff, v_diff = utils.get_signal_derivative(self.t_span, logger["v"], self.dt)
        axs[0, 2].plot(t_diff, v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, v_diff[:, 2], marker='.')
        axs[0, 2].set_ylabel('a_x')
        axs[1, 2].set_ylabel('a_y')
        axs[2, 2].set_ylabel('a_z')        
        axs[0, 2].legend(['accel from dynamics', 'v_diff as accel to check numerical precision'])
        t_diff, a_diff = utils.get_signal_derivative(self.t_span, logger["dv"], self.dt)
        axs[0, 3].plot(t_diff, a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, a_diff[:, 2], marker='.')
        axs[0, 3].legend(['a_diff as jerk'])
        axs[0, 2].set_ylabel('j_x')
        axs[1, 2].set_ylabel('j_y')
        axs[2, 2].set_ylabel('j_z')  

    def plot_quaternion(self, logger: np.ndarray):
        fig, axs = plt.subplots(5, 1, sharex=True)
        fig.suptitle('quaternion')
        axs[0].plot(self.t_span, logger["q"][:, 0], marker='x')
        axs[1].plot(self.t_span, logger["q"][:, 1], marker='x')
        axs[2].plot(self.t_span, logger["q"][:, 2], marker='x')
        axs[3].plot(self.t_span, logger["q"][:, 3], marker='x')
        axs[0].set_ylabel('q0')
        axs[1].set_ylabel('q1')
        axs[2].set_ylabel('q2')
        axs[3].set_ylabel('q3')
        axs[4].plot(self.t_span, logger["q"][:, 0]**2 +
                    logger["q"][:, 1]**2 +
                    logger["q"][:, 2]**2 +
                    logger["q"][:, 3]**2, marker='x')
        axs[4].set_ylabel('norm of q (should be 1)')

    def plot_omega_and_derivatives(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('omega_and_derivatives')
        axs[0, 0].plot(self.t_span, logger["omega"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["omega"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["omega"][:, 2], marker='x')
        axs[0, 0].set_ylabel("omega x")
        axs[1, 0].set_ylabel("omega y")
        axs[2, 0].set_ylabel("omega z")
        axs[0, 1].plot(self.t_span, logger["omega_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["omega_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["omega_dot"][:, 2], marker='x')
        t_diff, omega_diff = utils.get_signal_derivative(self.t_span, logger["omega"], self.dt)
        axs[0, 1].plot(t_diff, omega_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, omega_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, omega_diff[:, 2], marker='.')
        axs[0, 1].set_ylabel("omega dot x")
        axs[1, 1].set_ylabel("omega dot y")
        axs[2, 1].set_ylabel("omega dot z")

    def plot_pose_and_derivatives(self, logger: np.ndarray):
        fig1, axs1 = plt.subplots(3, 6, sharex=True)
        fig1.suptitle('pose (left) and pose_dot (right)')
        axs1[0, 0].plot(self.t_span, logger["pose"][:, 0, 0], marker='x')
        axs1[1, 0].plot(self.t_span, logger["pose"][:, 1, 0], marker='x')
        axs1[2, 0].plot(self.t_span, logger["pose"][:, 2, 0], marker='x')
        axs1[0, 1].plot(self.t_span, logger["pose"][:, 0, 1], marker='x')
        axs1[1, 1].plot(self.t_span, logger["pose"][:, 1, 1], marker='x')
        axs1[2, 1].plot(self.t_span, logger["pose"][:, 2, 1], marker='x')
        axs1[0, 2].plot(self.t_span, logger["pose"][:, 0, 2], marker='x')
        axs1[1, 2].plot(self.t_span, logger["pose"][:, 1, 2], marker='x')
        axs1[2, 2].plot(self.t_span, logger["pose"][:, 2, 2], marker='x')
        for i in range(3):
            for j in range(3):
                axs1[i, j].set_ylim(-1, 1)

        axs1[0, 3].plot(self.t_span, logger["pose_dot"][:, 0, 0], marker='x')
        axs1[1, 3].plot(self.t_span, logger["pose_dot"][:, 1, 0], marker='x')
        axs1[2, 3].plot(self.t_span, logger["pose_dot"][:, 2, 0], marker='x')
        axs1[0, 4].plot(self.t_span, logger["pose_dot"][:, 0, 1], marker='x')
        axs1[1, 4].plot(self.t_span, logger["pose_dot"][:, 1, 1], marker='x')
        axs1[2, 4].plot(self.t_span, logger["pose_dot"][:, 2, 1], marker='x')
        axs1[0, 5].plot(self.t_span, logger["pose_dot"][:, 0, 2], marker='x')
        axs1[1, 5].plot(self.t_span, logger["pose_dot"][:, 1, 2], marker='x')
        axs1[2, 5].plot(self.t_span, logger["pose_dot"][:, 2, 2], marker='x')

    def plot_trajectory(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('trajectory')
        axs[0, 0].plot(self.t_span, logger["x_d"][:, 0], marker='.')
        axs[0, 0].set_ylabel('x')
        axs[1, 0].plot(self.t_span, logger["x_d"][:, 1], marker='.')
        axs[1, 0].set_ylabel('y')
        axs[2, 0].plot(self.t_span, logger["x_d"][:, 2], marker='.')
        axs[2, 0].set_ylabel('z')
        axs[0, 1].plot(self.t_span, logger["v_d"][:, 0], marker='.')
        axs[0, 1].set_ylabel('v_x')
        axs[1, 1].plot(self.t_span, logger["v_d"][:, 1], marker='.')
        axs[1, 1].set_ylabel('v_y')
        axs[2, 1].plot(self.t_span, logger["v_d"][:, 2], marker='.')
        axs[2, 1].set_ylabel('v_z')

        axs[0, 2].plot(self.t_span, logger["x_d_dot2"][:, 0], marker='.')
        axs[1, 2].plot(self.t_span, logger["x_d_dot2"][:, 1], marker='.')
        axs[2, 2].plot(self.t_span, logger["x_d_dot2"][:, 2], marker='.')
        axs[0, 2].set_ylabel('x_ddot')
        axs[1, 2].set_ylabel('y_ddot')
        axs[2, 2].set_ylabel('z_ddot')

        axs[0, 3].plot(self.t_span, logger["x_d_dot3"][:, 0], marker='.')
        axs[1, 3].plot(self.t_span, logger["x_d_dot3"][:, 1], marker='.')
        axs[2, 3].plot(self.t_span, logger["x_d_dot3"][:, 2], marker='.')
        axs[0, 3].set_ylabel('x_dddot')
        axs[1, 3].set_ylabel('y_dddot')
        axs[2, 3].set_ylabel('z_dddot')

    def plot_force_and_torque(self, logger: np.ndarray):
        fig, axs = plt.subplots(4, 2, sharex=True)
        fig.suptitle('force_and_torque')
        axs[0, 0].plot(self.t_span, logger["f_d"][:, 0], marker='x', label="f_d_x")
        axs[0, 0].plot(self.t_span, logger["f_ctrl_input"][:, 0], marker='.', label="f_ctrl_input_x")
        axs[0, 0].plot(self.t_span, logger["f_feedback"][:, 0], label="f_feedback_x")
        axs[0, 0].plot(self.t_span, logger["f_feedforward"][:, 0], label="f_feedforward_x")
        axs[0, 0].plot(self.t_span, logger["f_disturb_compensation"][:, 0], label="f_disturb_compensation_x")
        axs[1, 0].plot(self.t_span, logger["f_d"][:, 1], marker='x', label="f_d_y")
        axs[1, 0].plot(self.t_span, logger["f_ctrl_input"][:, 1], marker='.', label="f_ctrl_input_y")
        axs[1, 0].plot(self.t_span, logger["f_feedback"][:, 1], label="f_feedback_y")
        axs[1, 0].plot(self.t_span, logger["f_feedforward"][:, 1], label="f_feedforward_y")
        axs[1, 0].plot(self.t_span, logger["f_disturb_compensation"][:, 1], label="f_disturb_compensation_y")        
        axs[2, 0].plot(self.t_span, logger["f_d"][:, 2], marker='x', label="f_d_z")
        axs[2, 0].plot(self.t_span, logger["f_ctrl_input"][:, 2], marker='.', label="f_ctrl_input_z")
        axs[2, 0].plot(self.t_span, logger["f_feedback"][:, 2], label="f_feedback_z")
        axs[2, 0].plot(self.t_span, logger["f_feedforward"][:, 2], label="f_feedforward_z")
        axs[2, 0].plot(self.t_span, logger["f_disturb_compensation"][:, 2], label="f_disturb_compensation_z")
        axs[0, 0].legend()
        axs[1, 0].legend()
        axs[2, 0].legend()
        axs[3, 0].plot(self.t_span, np.sqrt(logger["f_d"][:, 0]**2 + 
            logger["f_d"][:, 1]**2 +
            logger["f_d"][:, 2]**2), marker='x', label="f_d_norm")
        axs[3, 0].plot(self.t_span, np.sqrt(logger["f_ctrl_input"][:, 0]**2 + 
            logger["f_ctrl_input"][:, 1]**2 +
            logger["f_ctrl_input"][:, 2]**2), marker='.', label="f_ctrl_input_norm")
        axs[3, 0].plot(self.t_span, np.sqrt(logger["f_feedback"][:, 0]**2 + 
            logger["f_feedback"][:, 1]**2 +
            logger["f_feedback"][:, 2]**2), label="f_feedback_norm")
        axs[3, 0].plot(self.t_span, np.sqrt(logger["f_feedforward"][:, 0]**2 + 
            logger["f_feedforward"][:, 1]**2 +
            logger["f_feedforward"][:, 2]**2), label="f_feedforward_norm")
        axs[3, 0].plot(self.t_span, np.sqrt(logger["f_disturb_compensation"][:, 0]**2 + 
            logger["f_disturb_compensation"][:, 1]**2 +
            logger["f_disturb_compensation"][:, 2]**2), label="f_disturb_compensation_norm")
        axs[3, 0].legend()
        axs[0, 0].set_ylabel('f_x')
        axs[1, 0].set_ylabel('f_y')
        axs[2, 0].set_ylabel('f_z')
        axs[3, 0].set_ylabel('f_norm')

        axs[0, 1].plot(self.t_span, logger["torque_ctrl_input"][:, 0], marker='x', label="torque_ctrl_input_x")
        axs[0, 1].plot(self.t_span, logger["torque_feedback"][:, 0], label="torque_feedback_x")
        axs[0, 1].plot(self.t_span, logger["torque_coriolis"][:, 0], label="torque_coriolis_x")
        axs[0, 1].plot(self.t_span, logger["torque_feedforward"][:, 0], label="torque_feedforward_x")
        axs[1, 1].plot(self.t_span, logger["torque_ctrl_input"][:, 1], marker='x', label="torque_ctrl_input_y")
        axs[1, 1].plot(self.t_span, logger["torque_feedback"][:, 1], label="torque_feedback_y")
        axs[1, 1].plot(self.t_span, logger["torque_coriolis"][:, 1], label="torque_coriolis_y")
        axs[1, 1].plot(self.t_span, logger["torque_feedforward"][:, 1], label="torque_feedforward_y")
        axs[2, 1].plot(self.t_span, logger["torque_ctrl_input"][:, 2], marker='x', label="torque_ctrl_input_z")
        axs[2, 1].plot(self.t_span, logger["torque_feedback"][:, 2], label="torque_feedback_z")
        axs[2, 1].plot(self.t_span, logger["torque_coriolis"][:, 2], label="torque_coriolis_z")
        axs[2, 1].plot(self.t_span, logger["torque_feedforward"][:, 2], label="torque_feedforward_z")
        axs[0, 1].legend()
        axs[1, 1].legend()
        axs[2, 1].legend()
        axs[3, 1].plot(self.t_span, np.sqrt(logger["torque_ctrl_input"][:, 0]**2 + 
            logger["torque_ctrl_input"][:, 1]**2 +
            logger["torque_ctrl_input"][:, 2]**2), marker='x', label="torque_ctrl_input_norm")
        axs[3, 1].plot(self.t_span, np.sqrt(logger["torque_feedback"][:, 0]**2 + 
            logger["torque_feedback"][:, 1]**2 +
            logger["torque_feedback"][:, 2]**2), label="torque_feedback_norm")
        axs[3, 1].plot(self.t_span, np.sqrt(logger["torque_coriolis"][:, 0]**2 + 
            logger["torque_coriolis"][:, 1]**2 +
            logger["torque_coriolis"][:, 2]**2), label="torque_coriolis_norm")
        axs[3, 1].plot(self.t_span, np.sqrt(logger["torque_feedforward"][:, 0]**2 + 
            logger["torque_feedforward"][:, 1]**2 +
            logger["torque_feedforward"][:, 2]**2), label="torque_feedforward_norm")
        axs[3, 1].legend()
        axs[0, 1].set_ylabel('M_x')
        axs[1, 1].set_ylabel('M_y')
        axs[2, 1].set_ylabel('M_z')
        axs[3, 1].set_ylabel('M_norm')

    def plot_position_tracking_error(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 4, sharex=True)
        fig.suptitle('position_tracking_error')
        axs[0, 0].set_title("position error x")
        axs[1, 0].set_title("position error y")
        axs[2, 0].set_title("position error z")
        axs[0, 0].plot(self.t_span, logger["e_x"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["e_x"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["e_x"][:, 2], marker='x')

        axs[0, 1].set_title("v error x")
        axs[1, 1].set_title("v error y")
        axs[2, 1].set_title("v error z")
        axs[0, 1].plot(self.t_span, logger["e_v"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["e_v"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["e_v"][:, 2], marker='x')
        t_diff, e_x_diff = utils.get_signal_derivative(
            self.t_span, logger["e_x"], self.dt)
        axs[0, 1].plot(t_diff, e_x_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, e_x_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, e_x_diff[:, 2], marker='.')
        t_diff, position_diff = utils.get_signal_derivative(self.t_span, logger["position"], self.dt)
        t_diff, x_d_diff = utils.get_signal_derivative(self.t_span, logger["x_d"], self.dt)
        axs[0, 1].plot(t_diff, position_diff[:, 0] - x_d_diff[:, 0], marker='.')
        axs[1, 1].plot(t_diff, position_diff[:, 1] - x_d_diff[:, 1], marker='.')
        axs[2, 1].plot(t_diff, position_diff[:, 2] - x_d_diff[:, 2], marker='.')
        axs[0, 1].plot(self.t_span, logger["v"][:, 0] - logger["v_d"][:, 0])
        axs[1, 1].plot(self.t_span, logger["v"][:, 1] - logger["v_d"][:, 1])
        axs[2, 1].plot(self.t_span, logger["v"][:, 2] - logger["v_d"][:, 2])

        axs[0, 2].set_title("a error x")
        axs[1, 2].set_title("a error y")
        axs[2, 2].set_title("a error z")
        axs[0, 2].plot(self.t_span, logger["e_a"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, logger["e_a"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, logger["e_a"][:, 2], marker='x')
        t_diff, e_v_diff = utils.get_signal_derivative(
            self.t_span, logger["e_v"], self.dt)
        axs[0, 2].plot(t_diff, e_v_diff[:, 0], marker='.')
        axs[1, 2].plot(t_diff, e_v_diff[:, 1], marker='.')
        axs[2, 2].plot(t_diff, e_v_diff[:, 2], marker='.')

        axs[0, 3].set_title("j error x")
        axs[1, 3].set_title("j error y")
        axs[2, 3].set_title("j error z")
        axs[0, 3].plot(self.t_span, logger["e_j"][:, 0], marker='x')
        axs[1, 3].plot(self.t_span, logger["e_j"][:, 1], marker='x')
        axs[2, 3].plot(self.t_span, logger["e_j"][:, 2], marker='x')
        t_diff, e_a_diff = utils.get_signal_derivative(
            self.t_span, logger["e_a"], self.dt)
        axs[0, 3].plot(t_diff, e_a_diff[:, 0], marker='.')
        axs[1, 3].plot(t_diff, e_a_diff[:, 1], marker='.')
        axs[2, 3].plot(t_diff, e_a_diff[:, 2], marker='.')

    def plot_pose_tracking_error(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('pose_tracking_error')
        axs[0, 0].plot(self.t_span, logger["e_r"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["e_r"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["e_r"][:, 2], marker='x')
        axs[0, 0].set_ylabel("e_r_x")
        axs[1, 0].set_ylabel("e_r_y")
        axs[2, 0].set_ylabel("e_r_z")        
        axs[0, 1].plot(self.t_span, logger["e_omega"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["e_omega"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["e_omega"][:, 2], marker='x')
        axs[0, 1].set_ylabel("e_omega_x")
        axs[1, 1].set_ylabel("e_omega_y")
        axs[2, 1].set_ylabel("e_omega_z")
        axs[0, 2].plot(self.t_span, logger["psi_r_rd"], marker='x')
        axs[0, 2].set_ylabel("psi_r_rd")

    def plot_desired_force(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 3, sharex=True)
        fig.suptitle('f_desired, f_desired_dot, f_desired_dot2')
        axs[0, 0].plot(self.t_span, logger["f_d"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["f_d"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["f_d"][:, 2], marker='x')
        axs[0, 0].set_ylabel("f_d_x")
        axs[1, 0].set_ylabel("f_d_y")
        axs[2, 0].set_ylabel("f_d_z")
        axs[0, 1].plot(self.t_span, logger["f_d_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["f_d_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["f_d_dot"][:, 2], marker='x')
        axs[0, 1].set_ylabel("f_d_dot_x")
        axs[1, 1].set_ylabel("f_d_dot_y")
        axs[2, 1].set_ylabel("f_d_dot_z")
        axs[0, 2].plot(self.t_span, logger["f_d_dot2"][:, 0], marker='x')
        axs[1, 2].plot(self.t_span, logger["f_d_dot2"][:, 1], marker='x')
        axs[2, 2].plot(self.t_span, logger["f_d_dot2"][:, 2], marker='x')
        axs[0, 2].set_ylabel("f_d_dot2_x")
        axs[1, 2].set_ylabel("f_d_dot2_y")
        axs[2, 2].set_ylabel("f_d_dot2_z")        

    def plot_control_input_force(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 2, sharex=True)
        fig.suptitle('force, force dot')
        axs[0, 0].plot(self.t_span, logger["f_ctrl_input"][:, 0], marker='x')
        axs[1, 0].plot(self.t_span, logger["f_ctrl_input"][:, 1], marker='x')
        axs[2, 0].plot(self.t_span, logger["f_ctrl_input"][:, 2], marker='x')
        axs[0, 0].set_ylabel("f_x")
        axs[1, 0].set_ylabel("f_y")
        axs[2, 0].set_ylabel("f_z")
        axs[0, 1].plot(self.t_span, logger["f_ctrl_input_dot"][:, 0], marker='x')
        axs[1, 1].plot(self.t_span, logger["f_ctrl_input_dot"][:, 1], marker='x')
        axs[2, 1].plot(self.t_span, logger["f_ctrl_input_dot"][:, 2], marker='x')
        axs[0, 1].set_ylabel("f_dot_x")
        axs[1, 1].set_ylabel("f_dot_y")
        axs[2, 1].set_ylabel("f_dot_z")        

    def plot_pose_desired(self, logger: np.ndarray):
        fig1, axs1 = plt.subplots(3, 9, sharex=True)
        fig1.suptitle('pose_desired, pose_desired_dot, pose_desired_dot2')
        axs1[0, 0].plot(self.t_span, logger["pose_desired"][:, 0, 0], marker='x', label="pose_desired[0, 0]")
        axs1[1, 0].plot(self.t_span, logger["pose_desired"][:, 1, 0], marker='x', label="pose_desired[1, 0]")
        axs1[2, 0].plot(self.t_span, logger["pose_desired"][:, 2, 0], marker='x', label="pose_desired[2, 0]")
        axs1[0, 1].plot(self.t_span, logger["pose_desired"][:, 0, 1], marker='x', label="pose_desired[0, 1]")
        axs1[1, 1].plot(self.t_span, logger["pose_desired"][:, 1, 1], marker='x', label="pose_desired[1, 1]")
        axs1[2, 1].plot(self.t_span, logger["pose_desired"][:, 2, 1], marker='x', label="pose_desired[2, 1]")
        axs1[0, 2].plot(self.t_span, logger["pose_desired"][:, 0, 2], marker='x', label="pose_desired[0, 2]")
        axs1[1, 2].plot(self.t_span, logger["pose_desired"][:, 1, 2], marker='x', label="pose_desired[1, 2]")
        axs1[2, 2].plot(self.t_span, logger["pose_desired"][:, 2, 2], marker='x', label="pose_desired[2, 2]")
        axs1[0, 0].plot(self.t_span, logger["pose"][:, 0, 0], label="pose[0, 0]")
        axs1[1, 0].plot(self.t_span, logger["pose"][:, 1, 0], label="pose[1, 0]")
        axs1[2, 0].plot(self.t_span, logger["pose"][:, 2, 0], label="pose[2, 0]")
        axs1[0, 1].plot(self.t_span, logger["pose"][:, 0, 1], label="pose[0, 1]")
        axs1[1, 1].plot(self.t_span, logger["pose"][:, 1, 1], label="pose[1, 1]")
        axs1[2, 1].plot(self.t_span, logger["pose"][:, 2, 1], label="pose[2, 1]")
        axs1[0, 2].plot(self.t_span, logger["pose"][:, 0, 2], label="pose[0, 2]")
        axs1[1, 2].plot(self.t_span, logger["pose"][:, 1, 2], label="pose[1, 2]")
        axs1[2, 2].plot(self.t_span, logger["pose"][:, 2, 2], label="pose[2, 2]")
        axs1[0, 0].legend()
        axs1[1, 0].legend()
        axs1[2, 0].legend()
        axs1[0, 1].legend()
        axs1[1, 1].legend()
        axs1[2, 1].legend()
        axs1[0, 2].legend()
        axs1[1, 2].legend()
        axs1[2, 2].legend()
        for i in range(3):
            for j in range(3):
                axs1[i, j].set_ylim(-1, 1)
        
        axs1[0, 3].plot(self.t_span, logger["pose_desired_dot"][:, 0, 0], marker='x', label="pose_desired_dot[0, 0]")
        axs1[1, 3].plot(self.t_span, logger["pose_desired_dot"][:, 1, 0], marker='x', label="pose_desired_dot[1, 0]")
        axs1[2, 3].plot(self.t_span, logger["pose_desired_dot"][:, 2, 0], marker='x', label="pose_desired_dot[2, 0]")
        axs1[0, 4].plot(self.t_span, logger["pose_desired_dot"][:, 0, 1], marker='x', label="pose_desired_dot[0, 1]")
        axs1[1, 4].plot(self.t_span, logger["pose_desired_dot"][:, 1, 1], marker='x', label="pose_desired_dot[1, 1]")
        axs1[2, 4].plot(self.t_span, logger["pose_desired_dot"][:, 2, 1], marker='x', label="pose_desired_dot[2, 1]")
        axs1[0, 5].plot(self.t_span, logger["pose_desired_dot"][:, 0, 2], marker='x', label="pose_desired_dot[0, 2]")
        axs1[1, 5].plot(self.t_span, logger["pose_desired_dot"][:, 1, 2], marker='x', label="pose_desired_dot[1, 2]")
        axs1[2, 5].plot(self.t_span, logger["pose_desired_dot"][:, 2, 2], marker='x', label="pose_desired_dot[2, 2]")
        axs1[0, 3].plot(self.t_span, logger["pose_dot"][:, 0, 0], label="pose_dot[0, 0]")
        axs1[1, 3].plot(self.t_span, logger["pose_dot"][:, 1, 0], label="pose_dot[1, 0]")
        axs1[2, 3].plot(self.t_span, logger["pose_dot"][:, 2, 0], label="pose_dot[2, 0]")
        axs1[0, 4].plot(self.t_span, logger["pose_dot"][:, 0, 1], label="pose_dot[0, 1]")
        axs1[1, 4].plot(self.t_span, logger["pose_dot"][:, 1, 1], label="pose_dot[1, 1]")
        axs1[2, 4].plot(self.t_span, logger["pose_dot"][:, 2, 1], label="pose_dot[2, 1]")
        axs1[0, 5].plot(self.t_span, logger["pose_dot"][:, 0, 2], label="pose_dot[0, 2]")
        axs1[1, 5].plot(self.t_span, logger["pose_dot"][:, 1, 2], label="pose_dot[1, 2]")
        axs1[2, 5].plot(self.t_span, logger["pose_dot"][:, 2, 2], label="pose_dot[2, 2]")
        axs1[0, 3].legend()
        axs1[1, 3].legend()
        axs1[2, 3].legend()
        axs1[0, 4].legend()
        axs1[1, 4].legend()
        axs1[2, 4].legend()
        axs1[0, 5].legend()
        axs1[1, 5].legend()
        axs1[2, 5].legend()

        axs1[0, 6].plot(self.t_span, logger["pose_desired_dot2"][:, 0, 0], marker='x', label="pose_desired_dot2[0, 0]")
        axs1[1, 6].plot(self.t_span, logger["pose_desired_dot2"][:, 1, 0], marker='x', label="pose_desired_dot2[1, 0]")
        axs1[2, 6].plot(self.t_span, logger["pose_desired_dot2"][:, 2, 0], marker='x', label="pose_desired_dot2[2, 0]")
        axs1[0, 7].plot(self.t_span, logger["pose_desired_dot2"][:, 0, 1], marker='x', label="pose_desired_dot2[0, 1]")
        axs1[1, 7].plot(self.t_span, logger["pose_desired_dot2"][:, 1, 1], marker='x', label="pose_desired_dot2[1, 1]")
        axs1[2, 7].plot(self.t_span, logger["pose_desired_dot2"][:, 2, 1], marker='x', label="pose_desired_dot2[2, 1]")
        axs1[0, 8].plot(self.t_span, logger["pose_desired_dot2"][:, 0, 2], marker='x', label="pose_desired_dot2[0, 2]")
        axs1[1, 8].plot(self.t_span, logger["pose_desired_dot2"][:, 1, 2], marker='x', label="pose_desired_dot2[1, 2]")
        axs1[2, 8].plot(self.t_span, logger["pose_desired_dot2"][:, 2, 2], marker='x', label="pose_desired_dot2[2, 2]")

    def plot_disturbance_force(self, logger: np.ndarray):
        fig, axs = plt.subplots(4, 2, sharex=True)
        fig.suptitle('disturbance')
        f_norm = np.sqrt(
            logger["f_disturb"][:, 0]**2 +
            logger["f_disturb"][:, 1]**2 +
            logger["f_disturb"][:, 2]**2
        )
        f_est_norm = np.sqrt(
            logger["f_disturb_est"][:, 0]**2 +
            logger["f_disturb_est"][:, 1]**2 +
            logger["f_disturb_est"][:, 2]**2
        )
        f_base_norm = np.sqrt(
            logger["f_disturb_est_base"][:, 0]**2 +
            logger["f_disturb_est_base"][:, 1]**2 +
            logger["f_disturb_est_base"][:, 2]**2
        )
        torque_norm = np.sqrt(
            logger["torque_disturb"][:, 0]**2 +
            logger["torque_disturb"][:, 1]**2 +
            logger["torque_disturb"][:, 2]**2
        )
        torque_est_norm = np.sqrt(
            logger["torque_disturb_est"][:, 0]**2 +
            logger["torque_disturb_est"][:, 1]**2 +
            logger["torque_disturb_est"][:, 2]**2
        )
        torque_base_norm = np.sqrt(
            logger["torque_disturb_est_base"][:, 0]**2 +
            logger["torque_disturb_est_base"][:, 1]**2 +
            logger["torque_disturb_est_base"][:, 2]**2
        )
        
        axs[0, 0].plot(self.t_span, logger["f_disturb"][:, 0], marker='.', markersize=9, linewidth=1.5, label='f_disturb')
        axs[0, 0].plot(self.t_span, logger["f_disturb_est"][:, 0], marker='.', markersize=7, linewidth=1, label='f_disturb_est')
        axs[0, 0].plot(self.t_span, logger["f_disturb_est_base"][:, 0], marker='.', markersize=5, linewidth=0.5, label='f_disturb_est_base')
        axs[0, 0].plot(self.t_span, logger["f_disturb_sensed_raw"][:, 0], marker='.', markersize=5, linewidth=0.3, label='f_disturb_sensed_raw')
        axs[0, 0].plot(self.t_span, logger["f_propeller"][:, 0], marker='.', markersize=5, linewidth=0.2, label='f_propeller')
        axs[0, 0].plot(self.t_span, logger["f_body"][:, 0], marker='.', markersize=5, linewidth=0.1, label='f_body')
        axs[0, 0].set_ylabel("f_x")
        axs[0, 0].legend()
        
        axs[1, 0].plot(self.t_span, logger["f_disturb"][:, 1], marker='.', markersize=9, linewidth=1.5, label='f_disturb')
        axs[1, 0].plot(self.t_span, logger["f_disturb_est"][:, 1], marker='.', markersize=7, linewidth=1, label='f_disturb_est')
        axs[1, 0].plot(self.t_span, logger["f_disturb_est_base"][:, 1], marker='.', markersize=5, linewidth=0.5, label='f_disturb_est_base')
        axs[1, 0].plot(self.t_span, logger["f_disturb_sensed_raw"][:, 1], marker='.', markersize=5, linewidth=0.5, label='f_disturb_sensed_raw')
        axs[1, 0].plot(self.t_span, logger["f_propeller"][:, 1], marker='.', markersize=5, linewidth=0.2, label='f_propeller')
        axs[1, 0].plot(self.t_span, logger["f_body"][:, 1], marker='.', markersize=5, linewidth=0.1, label='f_body')        
        axs[1, 0].set_ylabel("f_y")
        axs[1, 0].legend()
        
        axs[2, 0].plot(self.t_span, logger["f_disturb"][:, 2], marker='.', markersize=9, linewidth=1.5, label='f_disturb')
        axs[2, 0].plot(self.t_span, logger["f_disturb_est"][:, 2], marker='.', markersize=7, linewidth=1, label='f_disturb_est')
        axs[2, 0].plot(self.t_span, logger["f_disturb_est_base"][:, 2], marker='.', markersize=5, linewidth=0.5, label='f_disturb_est_base')
        axs[2, 0].plot(self.t_span, logger["f_disturb_sensed_raw"][:, 2], marker='.', markersize=5, linewidth=0.5, label='f_disturb_sensed_raw')
        axs[2, 0].plot(self.t_span, logger["f_propeller"][:, 2], marker='.', markersize=5, linewidth=0.2, label='f_propeller')
        axs[2, 0].plot(self.t_span, logger["f_body"][:, 2], marker='.', markersize=5, linewidth=0.1, label='f_body')        
        axs[2, 0].set_ylabel("f_z")
        axs[2, 0].legend()
        
        axs[3, 0].plot(self.t_span, f_norm, marker='.', markersize=9, linewidth=1.5, label='f_norm')
        axs[3, 0].plot(self.t_span, f_est_norm, marker='.', markersize=7, linewidth=1, label='f_est_norm')
        axs[3, 0].plot(self.t_span, f_base_norm, marker='.', markersize=5, linewidth=0.5, label='f_base_norm')
        axs[3, 0].set_ylabel("f_norm")
        axs[3, 0].legend()
        
        axs[0, 1].plot(self.t_span, logger["torque_disturb"][:, 0], marker='.')
        axs[0, 1].plot(self.t_span, logger["torque_disturb_est"][:, 0], marker='.')
        axs[0, 1].plot(self.t_span, logger["torque_disturb_est_base"][:, 0], marker='.')
        axs[0, 1].set_ylabel("torque_x")
        
        axs[1, 1].plot(self.t_span, logger["torque_disturb"][:, 1], marker='.')
        axs[1, 1].plot(self.t_span, logger["torque_disturb_est"][:, 1], marker='.')
        axs[1, 1].plot(self.t_span, logger["torque_disturb_est_base"][:, 1], marker='.')
        axs[1, 1].set_ylabel("torque_y")
        
        axs[2, 1].plot(self.t_span, logger["torque_disturb"][:, 2], marker='.')
        axs[2, 1].plot(self.t_span, logger["torque_disturb_est"][:, 2], marker='.')
        axs[2, 1].plot(self.t_span, logger["torque_disturb_est_base"][:, 2], marker='.')
        axs[2, 1].set_ylabel("torque_z")
        
        axs[3, 1].plot(self.t_span, torque_norm, marker='.')
        axs[3, 1].plot(self.t_span, torque_est_norm, marker='.')
        axs[3, 1].plot(self.t_span, torque_base_norm, marker='.')
        axs[3, 1].set_ylabel("torque_norm")

    def plot_rotor(self, logger: np.ndarray):
        # fig, axs = plt.subplots(2, 1, sharex=True)
        # fig.suptitle('rotor force, rotor speed')
        # axs[0].plot(self.t_span, logger["f_motor"][:, 0], label="Rotor 0 Force")
        # axs[0].plot(self.t_span, logger["f_motor"][:, 1], label="Rotor 1 Force")
        # axs[0].plot(self.t_span, logger["f_motor"][:, 2], label="Rotor 2 Force")
        # axs[0].plot(self.t_span, logger["f_motor"][:, 3], label="Rotor 3 Force")
        # axs[0].set_ylabel("Rotor Forces")
        # axs[0].legend()
        # axs[1].plot(self.t_span, logger["rotor_0_rotation_spd"], label="Rotor 0 Speed")
        # axs[1].plot(self.t_span, logger["rotor_1_rotation_spd"], label="Rotor 1 Speed")
        # axs[1].plot(self.t_span, logger["rotor_2_rotation_spd"], label="Rotor 2 Speed")
        # axs[1].plot(self.t_span, logger["rotor_3_rotation_spd"], label="Rotor 3 Speed")
        # axs[1].plot(self.t_span, logger["rotor_0_rotation_spd_delayed"], label="Rotor 0 Speed delayed")
        # axs[1].plot(self.t_span, logger["rotor_1_rotation_spd_delayed"], label="Rotor 1 Speed delayed")
        # axs[1].plot(self.t_span, logger["rotor_2_rotation_spd_delayed"], label="Rotor 2 Speed delayed")
        # axs[1].plot(self.t_span, logger["rotor_3_rotation_spd_delayed"], label="Rotor 3 Speed delayed")
        # axs[1].set_ylabel("rotor_spd [RPM]")
        # axs[1].legend()
        fig, axs = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
        fig.suptitle('Rotor Force & Rotor Speed')

        # === Rotor Force Plot ===
        force_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
        line_width_base = 1.5
        line_width_gap = 0.8
        force_line_width = [line_width_base + (4-i) * line_width_gap for i in range(4)]
        for i in range(4):
            axs[0].plot(
                self.t_span, 
                logger["f_motor"][:, i], 
                label=f"Rotor {i} Force", 
                linestyle='-', 
                linewidth=force_line_width[i], 
                color=force_colors[i], 
                zorder=3+i  # ensure later lines go on top
            )
        axs[0].set_ylabel("Rotor Forces")
        axs[0].legend(loc='upper right')

        # === Rotor Speed Plot ===
        speed_colors = force_colors
        speed_line_width = force_line_width
        delay_styles = ['--', '--', '--', '--']  # dashed for delayed
        for i in range(4):
            axs[1].plot(
                self.t_span, 
                logger[f"rotor_{i}_rotation_spd"], 
                label=f"Rotor {i} Speed", 
                color=speed_colors[i], 
                linestyle='-', 
                linewidth=speed_line_width[i], 
                zorder=2*i
            )
            axs[1].plot(
                self.t_span, 
                logger[f"rotor_{i}_rotation_spd_delayed"], 
                label=f"Rotor {i} Speed delayed", 
                color=speed_colors[i], 
                linestyle=delay_styles[i], 
                linewidth=speed_line_width[i], 
                zorder=2*i + 1
            )
        axs[1].set_ylabel("Rotor Speed [RPM]")
        axs[1].legend(loc='upper right')

        # Optional: Grid and layout
        for ax in axs:
            ax.grid(True)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    def plot_omega_desired(self, logger: np.ndarray):
        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.suptitle('omega_desired vs omega')
        axs[0].plot(self.t_span, logger["omega_desired"][:, 0], marker='x')
        axs[0].plot(self.t_span, logger["omega"][:, 0], marker='.')
        axs[1].plot(self.t_span, logger["omega_desired"][:, 1], marker='x')
        axs[1].plot(self.t_span, logger["omega"][:, 1], marker='.')
        axs[2].plot(self.t_span, logger["omega_desired"][:, 2], marker='x')
        axs[2].plot(self.t_span, logger["omega"][:, 2], marker='.')
        axs[0].legend(['omega desired', 'omega'])

    def plot_3d_trace(self, logger: np.ndarray):
        fig9, axs9 = plt.subplots(1, 1, sharex=True)
        axs9 = fig9.add_subplot(111, projection='3d')
        axs9.plot3D(logger["x_d"][:, 0],
                    logger["x_d"][:, 1],
                    logger["x_d"][:, 2], '.', c='blue', label='Points')
        axs9.plot3D(logger["position"][:, 0],
                    logger["position"][:, 1],
                    logger["position"][:, 2], 'green')
        b1b2, b3 = plot_utils.generate_drone_profile(
            logger["position"][-1, :], logger["pose"][-1, :, :])
        axs9.plot3D(b1b2[:, 0],
                    b1b2[:, 1],
                    b1b2[:, 2], 'red')
        axs9.plot3D(b3[:, 0],
                    b3[:, 1],
                    b3[:, 2], 'red')
        b1b2, b3 = plot_utils.generate_drone_profile(
             logger["position"][-1, :], logger["pose_desired"][-1, :, :])
        axs9.plot3D(b1b2[:, 0],
                    b1b2[:, 1],
                    b1b2[:, 2], 'orange')
        axs9.plot3D(b3[:, 0],
                    b3[:, 1],
                    b3[:, 2], 'orange')
        b1 = np.vstack((logger["x_d"][-1, :],
                        logger["x_d"][-1, :] + 0.5*logger["b_1d"][-1, :]))
        axs9.plot3D(b1[:, 0],
                    b1[:, 1],
                    b1[:, 2], 'purple')
        b1 = np.vstack(( logger["position"][-1, :],
                         logger["position"][-1, :] + 0.5*logger["b_1d"][-1, :]))
        axs9.plot3D(b1[:, 0],
                    b1[:, 1],
                    b1[:, 2], 'purple')
        axs9.set_title('3D Plot')
        axs9.set_xlabel('X')
        axs9.set_ylabel('Y')
        axs9.set_zlabel('Z')
        axs9.axis('equal')
        axs9.invert_zaxis()
        axs9.invert_yaxis()

    def plot_pose_in_given_time(self, logger: np.ndarray, t: float):
        idx = int(t/self.dt)
        idx = np.clip(idx, 0, len(self.t_span)-1)
        fig9, axs9 = plt.subplots(1, 1, sharex=True)
        axs9 = fig9.add_subplot(111, projection='3d')
        b1b2, b3 = plot_utils.generate_drone_profile(
            np.zeros(3), logger["pose"][idx, :, :])
        axs9.plot3D(b1b2[:, 0],
                b1b2[:, 1],
                b1b2[:, 2], 'red', label='Pose')
        axs9.plot3D(b3[:, 0],
                b3[:, 1],
                b3[:, 2], 'red')
        b1b2, b3 = plot_utils.generate_drone_profile(
            np.zeros(3), logger["pose_desired"][idx, :, :])
        axs9.plot3D(b1b2[:, 0],
                b1b2[:, 1],
                b1b2[:, 2], 'orange', label='Pose Desired')
        axs9.plot3D(b3[:, 0],
                b3[:, 1],
                b3[:, 2], 'orange')
        axs9.set_title('3D Plot')
        axs9.set_xlabel('X')
        axs9.set_ylabel('Y')
        axs9.set_zlabel('Z')
        axs9.axis('equal')
        axs9.invert_zaxis()
        axs9.invert_yaxis()
        axs9.legend()
        
    def animate_pose(self, logger: np.ndarray):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(logger["x_d"][:, 0],
                    logger["x_d"][:, 1],
                    logger["x_d"][:, 2], '.', c='blue', label='Points')
        ax.plot3D(logger["position"][:, 0],
                    logger["position"][:, 1],
                    logger["position"][:, 2], 'green')     
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axis('equal')           
        ax.invert_zaxis()
        ax.invert_yaxis()   
        text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes)   
        delta_frame = 100
        pos = logger["position"][::delta_frame, :]
        pose_desire = logger["pose_desired"][::delta_frame, :, :]
        pose = logger["pose"][::delta_frame, :, :]
        t = self.t_span[::delta_frame]
        ani = FuncAnimation(fig, lambda n: update_frame(ax, text, pos[n, :], pose_desire[n, :, :], pose[n, :, :], t[n]), frames=int(len(logger["pose_desired"])/delta_frame), interval=50, blit=True, repeat=False) 
        return ani

def update_frame(ax: plt.Axes, text, pos: np.ndarray, pose_ref: np.ndarray, pose_meas: np.ndarray, t: float):
    b1b2, b3 = plot_utils.generate_drone_profile(pos, pose_ref)
    ax.plot(b1b2[:, 0],
            b1b2[:, 1],
            b1b2[:, 2], 'orange')
    ax.plot(b3[:, 0],
            b3[:, 1],
            b3[:, 2], 'orange')
    b1b2, b3 = plot_utils.generate_drone_profile(pos, pose_meas)
    ax.plot(b1b2[:, 0],
            b1b2[:, 1],
            b1b2[:, 2], 'red')
    ax.plot(b3[:, 0],
            b3[:, 1],
            b3[:, 2], 'red')
    text.set_text(f'timestamp: {t}')
    return ax, text